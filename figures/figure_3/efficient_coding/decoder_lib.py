"""The model-free decoder: response tables, bootstrap populations, and the MLP read-out.

Nothing here knows that a retinotopic or a relational model exists. It turns a set of recorded
neurons into a simulated population — a kernel-regression response table per unit, resampled
with random receptive-field rotations and read out under independent Poisson noise — and trains
an MLP to recover the two stimulus angles from it.

The two halves, in order:

* **response tables** — `precompute_unit_tables` builds one `[n_grid, n_grid]` table of mean rate
  over the (theta0, theta1) grid per unit, which is the only expensive step;
  `sample_bootstrap_table_inputs` draws a simulated population from them, and
  `make_table_neural_fn` turns that into the `(theta0, theta1) -> rates` closure the task
  wants. Rotating a unit by `k` grid steps is a diagonal shift of its table, so the rotation
  replicas cost a re-index rather than a re-fit.
* **the task** — `stimulus_grid`, `sample_stimuli` and `target_bump` define what is decoded;
  `StimulusRepresentationModel`, `train_model` and `evaluate_error` are the read-out, trained on
  MSE against a target bump over a ring of output units and scored by finding the two peaks.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks


def _bump_weights_2obj(t0, t1, query_t0, query_t1, sigma_deg):
    """Swap-invariant circular-Gaussian weights of each recorded trial for each query,
    `[n_query, n_trials]`, rows summing to 1.

    The distance between a query and a trial is the smaller of the direct and
    object-swapped per-object arc-distance sums, so the estimate does not depend on which
    object is labeled 0 — matching the task's own order symmetry."""
    query_t0 = np.atleast_1d(np.asarray(query_t0, float))
    query_t1 = np.atleast_1d(np.asarray(query_t1, float))
    d_direct = circ_dist(query_t0[:, None], t0[None, :]) + circ_dist(
        query_t1[:, None], t1[None, :]
    )
    d_swap = circ_dist(query_t0[:, None], t1[None, :]) + circ_dist(
        query_t1[:, None], t0[None, :]
    )
    e = -0.5 * (np.minimum(d_direct, d_swap) / np.radians(sigma_deg)) ** 2
    e -= e.max(axis=1, keepdims=True)  # stable per query row
    w = np.exp(e)
    return w / w.sum(axis=1, keepdims=True)


def precompute_unit_tables(units_2obj_trials, sample_sigma_deg, n_grid):
    """One kernel-regression response table per unit over the `n_grid × n_grid` `(θ0, θ1)`
    grid: `[n_units, n_grid, n_grid]` float32.

    This is the whole expensive real-data step, done once for the entire candidate pool.
    Afterwards a unit rotated by any grid-aligned orientation and queried at any grid
    stimulus is a plain shifted read of its own table."""
    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    g0, g1 = np.meshgrid(grid, grid, indexing="ij")
    q0, q1 = g0.ravel(), g1.ravel()
    tables = np.empty(
        (len(units_2obj_trials), n_grid, n_grid), dtype=np.float32
    )
    for u, (t0, t1, fr) in enumerate(units_2obj_trials):
        w = _bump_weights_2obj(t0, t1, q0, q1, sample_sigma_deg)
        with np.errstate(all="ignore"):
            tables[u] = (w @ fr).reshape(n_grid, n_grid)
    return tables


def sample_bootstrap_table_inputs(pool_positions, n_samples, n_grid, rng):
    """Draw `n_samples` bootstrap inputs: a unit picked **with replacement** from
    `pool_positions` plus an orientation drawn uniformly on [0, 2π) and snapped to the grid.
    Returns `(unit_rows, shifts)`."""
    idx = rng.integers(0, len(pool_positions), size=n_samples)
    unit_rows = np.asarray(pool_positions)[idx]
    offsets = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    shifts = np.round(offsets / (2 * np.pi / n_grid)).astype(int) % n_grid
    return unit_rows, shifts


def make_table_neural_fn(tables, unit_rows, shifts, firing_rate_scale):
    """The `neural_fn(theta0, theta1, rng) -> [n, n_samples]` closure the task wants, built
    purely from precomputed tables — no kernel math per call.

    Input `j` is unit `unit_rows[j]` rotated by `shifts[j]` grid steps: reading its table
    `k` cells over in **both** stimulus axes is evaluating that unit at the stimulus rotated
    by `k·dθ`, i.e. a copy of the neuron whose tuning is rigidly rotated by `−k·dθ`.
    `firing_rate_scale` multiplies every rate before the Poisson draw: < 1 makes Poisson noise
    dominate, > 1 makes it relatively smaller."""
    n_grid = tables.shape[1]
    dtheta = 2 * np.pi / n_grid
    unit_rows = np.asarray(unit_rows)
    shifts = np.asarray(shifts)

    def neural_fn(theta0, theta1, rng):
        theta0 = np.atleast_1d(np.asarray(theta0, float))
        theta1 = np.atleast_1d(np.asarray(theta1, float))
        i0 = np.round(theta0 / dtheta).astype(int) % n_grid
        i1 = np.round(theta1 / dtheta).astype(int) % n_grid
        a0 = (i0[:, None] + shifts[None, :]) % n_grid  # [n_query, n_samples]
        a1 = (i1[:, None] + shifts[None, :]) % n_grid
        rates = tables[unit_rows[None, :], a0, a1] * firing_rate_scale
        return rng.poisson(rates).astype(np.float32)

    return neural_fn


def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def stimulus_grid(n_grid):
    """`n_grid` angles evenly spaced around [0, 2π) — the decode grid."""
    return np.linspace(0, 2 * np.pi, n_grid, endpoint=False)


def sample_stimuli(n, min_sep_deg, rng):
    """`(theta0, theta1)` for `n` trials: the first object uniform on the ring, the second
    at least `min_sep_deg` away (uniform on the allowed arc, either side)."""
    theta0 = rng.uniform(0, 2 * np.pi, n)
    min_sep = np.radians(min_sep_deg)
    delta = rng.uniform(
        min_sep, 2 * np.pi - min_sep, n
    )  # circ-dist(delta) >= min_sep
    return theta0, (theta0 + delta) % (2 * np.pi)


def target_bump(theta0, theta1, grid, sigma_deg):
    """`[n, len(grid)]` target: the **sum** of two circular-Gaussian bumps (peak 1 each) at
    the two object angles — symmetric under swapping them."""
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    theta1 = np.atleast_1d(np.asarray(theta1, float))
    sigma = np.radians(sigma_deg)
    b0 = np.exp(
        -0.5 * (circ_dist(theta0[:, None], grid[None, :]) / sigma) ** 2
    )
    b1 = np.exp(
        -0.5 * (circ_dist(theta1[:, None], grid[None, :]) / sigma) ** 2
    )
    return (b0 + b1).astype(np.float32)


def make_batch(
    neural_fn, batch_size, min_sep_deg, grid, target_sigma_deg, rng
):
    """One training batch: `(theta0, theta1, neural, target)`."""
    theta0, theta1 = sample_stimuli(batch_size, min_sep_deg, rng)
    return (
        theta0,
        theta1,
        neural_fn(theta0, theta1, rng),
        target_bump(theta0, theta1, grid, target_sigma_deg),
    )


def _mlp(in_dim, hidden_sizes, out_dim):
    layers, d = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, out_dim))


class StimulusRepresentationModel(nn.Module):
    """Feed-forward readout: population response → activity bump at *both* object
    locations. The only trainable component; `hidden_sizes` is a list of layer widths.
    """

    def __init__(self, n_neurons, n_angles, hidden_sizes):
        super().__init__()
        self.mlp = _mlp(n_neurons, hidden_sizes, n_angles)

    def forward(self, neural):
        return self.mlp(neural)


def model_output(model, neural_fn, theta0, theta1, rng):
    """The output bump for one stochastic population sample per stimulus."""
    model.eval()
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    theta1 = np.atleast_1d(np.asarray(theta1, float))
    # `model.eval()` only switches layer behaviour; the graph is still built without this, and
    # `.numpy()` on a tensor that requires grad raises. Evaluation never backpropagates.
    with torch.no_grad():
        out = model(torch.as_tensor(neural_fn(theta0, theta1, rng))).numpy()
    return out.astype(float)


def decode_two_peaks(output, grid, min_peak_sep_deg):
    """The two most prominent peaks of each output curve, at least `min_peak_sep_deg` apart.

    The grid is circular, so each curve is tiled three times and peaks are kept only from
    the middle copy — a peak straddling the 0°/360° seam is then found correctly. If fewer
    than two peaks exist, the single global maximum is repeated. Returns `(peak_a, peak_b)`
    in radians, unordered."""
    output = np.atleast_2d(output)
    n_grid = len(grid)
    distance = max(1, int(round(min_peak_sep_deg / (360.0 / n_grid))))
    peak_a = np.empty(len(output))
    peak_b = np.empty(len(output))
    for i, row in enumerate(output):
        idx, _ = find_peaks(np.concatenate([row, row, row]), distance=distance)
        idx = idx[(idx >= n_grid) & (idx < 2 * n_grid)] - n_grid
        if not len(idx):
            idx = np.array([int(np.argmax(row))])
        idx = idx[np.argsort(row[idx])[::-1]]  # most active first
        top = np.concatenate([idx, idx])[:2]  # repeat if only one found
        peak_a[i], peak_b[i] = grid[top[0]], grid[top[1]]
    return peak_a, peak_b


def train_model(model, neural_fn, cfg, rng):
    """Train the MLP with Adam on the MSE against the target bump. `cfg` needs
    `n_angles, min_sep_deg, target_sigma_deg, batch_size, n_steps, lr`."""
    grid = stimulus_grid(cfg["n_angles"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for _ in range(cfg["n_steps"]):
        _, _, neural, target = make_batch(
            neural_fn,
            cfg["batch_size"],
            cfg["min_sep_deg"],
            grid,
            cfg["target_sigma_deg"],
            rng,
        )
        out = model(torch.as_tensor(neural))
        loss = ((out - torch.as_tensor(target)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate_error(model, neural_fn, cfg, rng, n_test, min_peak_sep_deg):
    """Per-trial mean per-object decode error (degrees) on a fresh random test set of
    `n_test` stimuli: the two decoded peaks matched to the two true angles by whichever pairing
    has lower total error (the task has no object order)."""
    grid = stimulus_grid(cfg["n_angles"])
    theta0, theta1 = sample_stimuli(n_test, cfg["min_sep_deg"], rng)
    out = model_output(model, neural_fn, theta0, theta1, rng)
    peak_a, peak_b = decode_two_peaks(
        out, grid, min_peak_sep_deg=min_peak_sep_deg
    )
    direct = np.degrees(circ_dist(peak_a, theta0)) + np.degrees(
        circ_dist(peak_b, theta1)
    )
    swapped = np.degrees(circ_dist(peak_a, theta1)) + np.degrees(
        circ_dist(peak_b, theta0)
    )
    return np.minimum(direct, swapped) / 2.0
