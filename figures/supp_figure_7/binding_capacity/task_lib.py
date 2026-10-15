"""Cued-response task on a synthetic retinotopic/relational population — the
``binding_capacity`` model.

On each trial two objects are on screen (at least ``MIN_SEP_DEG`` apart) and the
model must report the location of the **first** object as a bump of activity over
a decode grid. The population code is symmetric under swapping the two objects —
it says *where* the objects are but not *which one is first* — so the ordering has
to arrive through a separate, deliberately low-capacity **cue** channel, and the
readout must fuse the two. That is the binding problem this directory measures.

The model is :class:`CueMatrixModel`: the cue is a genuine Poisson spike channel
whose spikes are projected into the population through a non-negative weight
matrix with a fixed mean weight. See its docstring for the full pipeline.
"""

import numpy as np
import torch
import torch.nn as nn

# torch takes every core for CPU ops unless told otherwise, which makes the machine
# unusable while a sweep trains. Cap it at import; raise it deliberately if a run is the
# only thing on the machine.
N_THREADS = 6
torch.set_num_threads(min(N_THREADS, torch.get_num_threads()))


# ── Circular helpers ──────────────────────────────────────────────────────────
def circ_dist(a, b):
    """Circular arc distance in [0, pi]; broadcasts. Wraps around the circle."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def pair_vector_angle(t0, t1):
    """Angle of the vector from object 0 to object 1 (their unit-circle
    positions). Swapping the two objects flips this angle by exactly pi — which is
    why the relational tuning below is *double*-frequency (period pi), and so is
    invariant to object order."""
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    dx = np.cos(t1) - np.cos(t0)
    dy = np.sin(t1) - np.sin(t0)
    return np.arctan2(dy, dx) % (2 * np.pi)


def stimulus_grid(n_grid):
    """``n_grid`` candidate stimulus positions, evenly spaced around [0, 2pi)."""
    return np.linspace(0, 2 * np.pi, n_grid, endpoint=False)


# ── Synthetic population (ground-truth params, no fitting) ────────────────────
def make_population(n_retinotopic, n_relational, baseline, amplitude):
    """Ground-truth population as a dict of ``[n_neurons]`` arrays. Preferred
    directions are evenly spaced around the circle *within* each kind (the two
    kinds tile the circle independently). Retinotopic and relational neurons
    share identical 1-object cosine tuning ``(baseline, amplitude, mean)`` and
    differ only in how they combine two objects — see
    :func:`expected_rates_2obj`."""
    prefs_ret = 2 * np.pi * np.arange(n_retinotopic) / max(n_retinotopic, 1)
    prefs_rel = 2 * np.pi * np.arange(n_relational) / max(n_relational, 1)
    kind = np.array(
        ["retinotopic"] * n_retinotopic + ["relational"] * n_relational
    )
    return dict(
        kind=kind,
        is_relational=(kind == "relational"),
        baseline=np.full(len(kind), float(baseline)),
        amplitude=np.full(len(kind), float(amplitude)),
        mean=np.concatenate([prefs_ret, prefs_rel]),
        n_retinotopic=n_retinotopic,
        n_relational=n_relational,
    )


def n_neurons(pop):
    return len(pop["mean"])


def expected_rates_2obj(theta0, theta1, pop):
    """``[n_query, n_neurons]`` expected 2-object firing rate, the synthetic
    ground truth: retinotopic neurons sum each object's own cosine drive;
    relational neurons apply a double-frequency cosine to the segment angle
    ``pair_vector_angle(t0, t1)``, oriented along the neuron's own preferred
    direction."""
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    theta1 = np.atleast_1d(np.asarray(theta1, float))
    mu = pop["mean"][None, :]
    base, amp = pop["baseline"][None, :], pop["amplitude"][None, :]
    retino = base + amp * (
        np.cos(theta0[:, None] - mu) + np.cos(theta1[:, None] - mu)
    )
    phi = pair_vector_angle(theta0, theta1)  # [n_query]
    relational = base + amp * np.cos(2 * (phi[:, None] - mu))
    rates = np.where(pop["is_relational"][None, :], relational, retino)
    return np.clip(rates, 0.0, None)


# ── Task: stimuli, cue features, neural samples, target bumps ─────────────────
def sample_stimuli(n, min_sep_deg, rng):
    """``(theta0, theta1)`` for ``n`` trials: ``theta0`` uniform on the circle,
    the second object placed a circular distance of at least ``min_sep_deg``
    away (uniformly on the allowed arc, either side)."""
    theta0 = rng.uniform(0, 2 * np.pi, n)
    min_sep = np.radians(min_sep_deg)
    delta = rng.uniform(
        min_sep, 2 * np.pi - min_sep, n
    )  # circ-dist(delta) >= min_sep
    theta1 = (theta0 + delta) % (2 * np.pi)
    return theta0, theta1


def cue_features(theta0, theta1):
    """``[n, 4]`` ordered cue input ``[cos θ₀, sin θ₀, cos θ₁, sin θ₁]`` — a
    periodic encoding of both object angles that preserves their order (θ₀
    first), which is exactly the information the symmetric population code
    lacks."""
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    theta1 = np.atleast_1d(np.asarray(theta1, float))
    return np.stack(
        [np.cos(theta0), np.sin(theta0), np.cos(theta1), np.sin(theta1)],
        axis=-1,
    ).astype(np.float32)


def neural_sample(theta0, theta1, pop):
    """``[n, n_neurons]`` neural input: the population's exact expected 2-object
    rate (:class:`CueMatrixModel` Poisson-samples it itself). This is fixed input
    *data* — the population is not trained."""
    return expected_rates_2obj(theta0, theta1, pop).astype(np.float32)


def target_bump(theta0, grid, sigma_deg):
    """``[n, len(grid)]`` circular-Gaussian target bump (peak 1) centered on
    each trial's first-object angle ``theta0``."""
    theta0 = np.atleast_1d(np.asarray(theta0, float))
    d = circ_dist(theta0[:, None], grid[None, :])
    return np.exp(-0.5 * (d / np.radians(sigma_deg)) ** 2).astype(np.float32)


# ── Model ────────────────────────────────────────────────────────────────────
def _mlp(in_dim, hidden_sizes, out_dim):
    """Feed-forward ReLU MLP from a list of hidden layer sizes (empty list => a
    single linear layer)."""
    layers, d = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class CueMatrixModel(nn.Module):
    """Cue-response model where a **Poisson-sampled** cue representation is
    projected through a **non-negative weight matrix** into an additive delta on
    the neural state.

    Pipeline (``neural`` is the exact population rate, from
    :func:`neural_sample`)::

        cue_rate   = cue_max_rate * sigmoid(cue_mlp(cue_feats))   # [., cue_output_neurons]
        cue_sample = Poisson(cue_rate)                            # straight-through
        delta      = cue_sample @ W.T                             # W >= 0, mean(W) = cue_influence
        combined   = ReLU(neural + delta)                        # [., n_neurons]
        noisy      = Poisson(combined)                           # straight-through
        out        = readout(noisy)                              # [., n_angles]

    So the cue is its own noisy spike channel — ``cue_output_neurons`` units
    firing Poisson spikes at rates up to ``cue_max_rate`` — whose spikes are
    synaptically weighted into the population by ``W`` and then read out under
    the population's own Poisson noise (a second, separate draw).

    ``W`` is ``[n_neurons, cue_output_neurons]``, **entrywise non-negative**, and
    its mean entry is fixed at ``cue_influence``. See :meth:`cue_weight_matrix`
    for the exact (softmax-based) parameterization, which enforces both at every
    step. ``cue_hidden`` / ``readout_hidden`` are lists of hidden sizes. The cue
    nonlinearity is a sigmoid (a Poisson rate must be non-negative), scaled by
    ``cue_max_rate``. The cue input is the 4-dimensional :func:`cue_features`.
    """

    def __init__(
        self,
        n_neurons,
        n_angles,
        cue_hidden,
        readout_hidden,
        cue_output_neurons,
        cue_max_rate,
        cue_influence,
    ):
        super().__init__()
        self.cue_mlp = _mlp(4, cue_hidden, cue_output_neurons)
        self.readout = _mlp(n_neurons, readout_hidden, n_angles)
        self.cue_max_rate = float(cue_max_rate)
        self.cue_influence = float(cue_influence)
        self.n_neurons = int(n_neurons)
        self.cue_output_neurons = int(cue_output_neurons)
        # Unconstrained logits -> a non-negative W (softmax base) rescaled to the
        # requested mean (see cue_weight_matrix). Zero-init => uniform.
        self.cue_weight_logits = nn.Parameter(
            torch.zeros(n_neurons * cue_output_neurons)
        )

    def cue_weight_matrix(self):
        """Non-negative ``[n_neurons, cue_output_neurons]`` cue→state weights:
        a softmax base ``p`` (positive, summing to 1) scaled so the mean entry
        equals ``cue_influence``."""
        p = torch.softmax(self.cue_weight_logits, dim=0)
        w = self.cue_influence * p * p.numel()
        return w.view(self.n_neurons, self.cue_output_neurons)

    def cue_rate(self, cue_feats):
        """Latent cue *rate* (pre-Poisson) in ``(0, cue_max_rate)``,
        ``[., cue_output_neurons]``."""
        return self.cue_max_rate * torch.sigmoid(self.cue_mlp(cue_feats))

    def cue_delta(self, cue_feats):
        """The additive delta on the neural state: the Poisson-sampled cue
        representation projected through the non-negative weight matrix,
        ``[., n_neurons]``."""
        rate = self.cue_rate(cue_feats)
        noisy = torch.poisson(rate)
        rate = rate + (noisy - rate).detach()  # straight-through
        return rate @ self.cue_weight_matrix().t()

    def forward(self, neural, cue_feats):
        combined = torch.relu(neural + self.cue_delta(cue_feats))
        noisy = torch.poisson(combined)
        # Straight-through on the population draw.
        combined = combined + (noisy - combined).detach()
        return self.readout(combined)


# ── Training ─────────────────────────────────────────────────────────────────
def make_batch(
    pop,
    batch_size,
    min_sep_deg,
    grid,
    target_sigma_deg,
    rng,
):
    """One training batch: returns ``(theta0, theta1, neural, cue, target)`` as
    numpy arrays (float32 for the tensors)."""
    theta0, theta1 = sample_stimuli(batch_size, min_sep_deg, rng)
    neural = neural_sample(theta0, theta1, pop)
    cue = cue_features(theta0, theta1)
    target = target_bump(theta0, grid, target_sigma_deg)
    return theta0, theta1, neural, cue, target


def train_model(model, pop, cfg, rng, log_every=200):
    """Train the readout + cue MLPs with Adam on MSE loss. ``cfg`` is a dict
    with keys ``n_angles, min_sep_deg, target_sigma_deg, batch_size, n_steps,
    lr``. Returns ``(losses, grid)``."""
    grid = stimulus_grid(cfg["n_angles"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    losses = []
    model.train()
    for step in range(cfg["n_steps"]):
        _, _, neural, cue, target = make_batch(
            pop,
            cfg["batch_size"],
            cfg["min_sep_deg"],
            grid,
            cfg["target_sigma_deg"],
            rng,
        )
        neural_t = torch.as_tensor(neural)
        cue_t = torch.as_tensor(cue)
        target_t = torch.as_tensor(target)

        out = model(neural_t, cue_t)
        loss = ((out - target_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == cfg["n_steps"] - 1):
            print(f'step {step:5d}/{cfg["n_steps"]}  MSE={loss.item():.4f}')
    return losses, grid
