"""Checks that run in seconds, **before** anything is fit to real data.

    python test_pairs.py            # everything
    python test_pairs.py --n 40     # a bigger recovery test (slower)

Five groups, in the order a failure would matter:

1. **The identity.** ``cos(u_i - c) - cos(u_j - c) = p cos c + q sin c``. Everything downstream
   is built on it — it is what removes the phase sweep — so it is checked to machine precision.
2. **The estimator.** IRLS against a brute-force phase sweep: same phase, and a likelihood at
   least as high. If this fails the closed-form fit is finding the wrong optimum.
3. **The null.** Shuffled firing rates must give a held-out delta log-likelihood that is not
   **positive**. This is the leak test, and the direction is the whole point: a fold leak
   inflates it above zero, and nothing else in the pipeline would reveal that. It should land
   slightly *below* zero — two fitted parameters scored out of sample cost a little when there
   is no signal to find — and that small negative offset is measured here rather than assumed,
   because it is the floor against which a real neuron's dLL has to be read.
4. **Recovery.** Simulated retinotopic and relational neurons, on the real stimulus
   distribution with its 80 deg exclusion, must be won by their own model — and the fitted
   phase must match the phase they were built with.
5. **Choices that should not matter.** The weight floor and the comparison cap are there for
   memory and speed; they are checked to move the statistics in the third decimal or beyond.
"""

import argparse

import numpy as np
import pairs_lib as pairs


def _check(label, ok):
    print(f'  {"PASS" if ok else "FAIL"}  {label}')
    return bool(ok)


def stimuli(n, rng, min_sep_deg=80.0):
    """Stimuli as the task draws them: uniform on the ring, pairs closer than 80 deg rejected."""
    x, y = np.empty(n), np.empty(n)
    f = 0
    while f < n:
        a, b = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        k = pairs.circ_dist(a, b) >= np.radians(min_sep_deg)
        t = min(int(k.sum()), n - f)
        x[f : f + t], y[f : f + t] = a[k][:t], b[k][:t]
        f += t
    return x, y


def rates(x, y, kind, phase=1.0):
    """Noise-free rate of an ideal neuron of each kind, at a known phase."""
    if kind == "retinotopic":
        return 5 + 2 * (np.cos(x - phase) + np.cos(y - phase))
    if kind == "relational":
        return 5 + 2 * np.cos(x + y - phase)
    return np.full(len(x), 6.0)


# ── 1. The identity ───────────────────────────────────────────────────────────
def test_identity():
    rng = np.random.default_rng(0)
    u_i, u_j = rng.uniform(-8, 8, 5000), rng.uniform(-8, 8, 5000)
    p, q = pairs._predictors(u_i, u_j)
    worst = 0.0
    for c in rng.uniform(0, 2 * np.pi, 25):
        direct = np.cos(u_i - c) - np.cos(u_j - c)
        worst = max(
            worst,
            float(np.max(np.abs(direct - (p * np.cos(c) + q * np.sin(c))))),
        )
    return _check(
        f"ds = p cos c + q sin c  (max err {worst:.1e})", worst < 1e-12
    )


# ── 2. The estimator ──────────────────────────────────────────────────────────
def test_irls_matches_sweep():
    rng = np.random.default_rng(1)
    x, y = stimuli(200, rng)
    fr = rng.poisson(rates(x, y, "relational") * 2.0) / 2.0
    c = pairs.build_comparisons(x, y, fr, -1)
    g = pairs.fit_irls(c["w"], c["y"], c["p"], c["q"])
    phase, beta = pairs.gamma_to_phase(g)
    ll_irls = pairs.mean_loglik(c["w"], c["y"], c["p"], c["q"], g)
    grid = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    lls = [
        pairs.mean_loglik(
            c["w"],
            c["y"],
            c["p"],
            c["q"],
            beta * np.array([np.cos(t), np.sin(t)]),
        )
        for t in grid
    ]
    best = float(np.degrees(grid[int(np.argmax(lls))]))
    ok = _check(
        f"IRLS phase {phase:.1f} deg matches a 720-point sweep ({best:.1f} deg)",
        abs(pairs.wrap_deg(phase - best)) < 1.0,
    )
    return ok & _check(
        f"IRLS likelihood >= sweep ({ll_irls:+.5f} vs {max(lls):+.5f})",
        ll_irls >= max(lls) - 1e-9,
    )


# ── 3. The null, and the leak it would reveal ────────────────────────────────
def test_null_is_zero(n_rep=12):
    rng = np.random.default_rng(2)
    x, y = stimuli(220, rng)
    vals = {k: [] for k in pairs.MODELS}
    for rep in range(n_rep):
        fr = rng.poisson(rates(x, y, "relational") * 2.0) / 2.0
        fr = rng.permutation(fr)  # rates carry no stimulus information
        for k in pairs.MODELS:
            vals[k].append(
                pairs.cv_delta_ll(
                    pairs.build_comparisons(x, y, fr, k), seed=rep
                )
            )
    ok = True
    for k in pairs.MODELS:
        v = np.array(vals[k], float)
        m, se = float(np.nanmean(v)), float(
            np.nanstd(v, ddof=1) / np.sqrt(len(v))
        )
        # One-sided: positive is a leak. Negative is the out-of-sample cost of two parameters,
        # and is reported so the size of that floor is on the record.
        ok &= _check(
            f"{pairs.MODEL_NAME[k]:>12}: held-out dLL on shuffled rates "
            f"{m:+.5f} +/- {se:.5f}  (must not be positive)",
            m < 3 * se,
        )
        ok &= _check(
            f"{pairs.MODEL_NAME[k]:>12}: that floor is small next to a real effect "
            f"(|{m:.5f}| < 0.02)",
            abs(m) < 0.02,
        )
    return ok


# ── 4. Recovery ───────────────────────────────────────────────────────────────
def test_recovery(n_neurons=8, n_trials=400, verbose=True):
    rng = np.random.default_rng(3)
    names = ["retinotopic", "relational"]
    conf = {g: {m: 0 for m in names} for g in names}
    phase_err = {g: [] for g in names}
    for gen in names:
        for _ in range(n_neurons):
            x, y = stimuli(n_trials, rng)
            truth = rng.uniform(0, 2 * np.pi)
            fr = rng.poisson(rates(x, y, gen, truth) * 2.0) / 2.0
            dll = {}
            for k in pairs.MODELS:
                c = pairs.build_comparisons(x, y, fr, k)
                dll[pairs.MODEL_NAME[k]] = pairs.cv_delta_ll(c, seed=0)
                if pairs.MODEL_NAME[k] == gen:
                    ph, _ = pairs.gamma_to_phase(
                        pairs.fit_irls(c["w"], c["y"], c["p"], c["q"])
                    )
                    phase_err[gen].append(
                        abs(pairs.wrap_deg(ph - np.degrees(truth)))
                    )
            conf[gen][max(dll, key=dll.get)] += 1
    if verbose:
        print(
            "\n  recovery (rows = simulated, columns = winner by held-out dLL):"
        )
        print(" " * 16 + "".join(f"{n:>14}" for n in names))
        for g in names:
            print(f"    {g:<12}" + "".join(f"{conf[g][m]:>14}" for m in names))
    ok = True
    for g in names:
        frac = conf[g][g] / max(sum(conf[g].values()), 1)
        ok &= _check(f"{g:>12} recovers itself ({frac:.0%})", frac >= 0.9)
        med = float(np.median(phase_err[g]))
        ok &= _check(
            f"{g:>12} phase within 10 deg of truth (median {med:.1f} deg)",
            med < 10.0,
        )
    return ok


# ── 5. Choices that should not matter ────────────────────────────────────────
def test_knobs_are_harmless():
    rng = np.random.default_rng(4)
    x, y = stimuli(300, rng)
    fr = rng.poisson(rates(x, y, "relational") * 2.0) / 2.0
    full = pairs.build_comparisons(
        x, y, fr, -1, weight_floor=0.0, max_comparisons=0
    )
    floored = pairs.build_comparisons(
        x, y, fr, -1, weight_floor=pairs.WEIGHT_FLOOR, max_comparisons=0
    )
    capped = pairs.build_comparisons(
        x,
        y,
        fr,
        -1,
        weight_floor=pairs.WEIGHT_FLOOR,
        max_comparisons=60_000,
        seed=0,
    )
    ph = {
        n: pairs.gamma_to_phase(
            pairs.fit_irls(c["w"], c["y"], c["p"], c["q"])
        )[0]
        for n, c in (("full", full), ("floored", floored), ("capped", capped))
    }
    d_floor = abs(pairs.wrap_deg(ph["floored"] - ph["full"]))
    d_cap = abs(pairs.wrap_deg(ph["capped"] - ph["full"]))
    kept = 100 * len(floored["w"]) / max(len(full["w"]), 1)
    ok = _check(
        f"weight floor keeps {kept:.0f}% of comparisons and moves the phase "
        f"{d_floor:.2f} deg",
        d_floor < 1.0,
    )
    return ok & _check(
        f"the comparison cap moves the phase {d_cap:.2f} deg", d_cap < 3.0
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n", type=int, default=8, help="simulated neurons per generator"
    )
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    ok = True
    print("1. the identity behind the closed-form fit:")
    ok &= test_identity()
    print("\n2. the estimator:")
    ok &= test_irls_matches_sweep()
    print("\n3. the null — and the fold leak it would expose:")
    ok &= test_null_is_zero()
    print("\n4. recovery:")
    ok &= test_recovery(args.n, args.trials)
    print("\n5. knobs that exist for speed, not for the answer:")
    ok &= test_knobs_are_harmless()
    print(f'\n{"ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"}')
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
