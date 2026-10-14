r"""Figure 3's motor decoder, fit once and cached to disk.

`figure_3/motor_decoding/decoder_v2.ipynb` memoises its decoders **in the kernel only**
(``_DECODERS``), so every fresh kernel refits all ``len(SUBJECTS) × N_BOOTSTRAPS`` of them. This
module fits the *same* decoders and writes each to ``cache/figures/supp_figure_4/decoders/`` so nothing here ever
refits one.

Nothing about the decoder is re-implemented:

* **The hyperparameters** are read from `decoder_v2.ipynb`'s own CONFIG cell at call time
  (:func:`decoder_config`), so an edit there reaches this directory and changes the cache key,
  instead of the two silently drifting apart.
* **The fit** is `decoder_v2`'s ``build_decoder``, line for line, calling the same
  `population_v2` / `population_lib` / `decoder_lib` functions, which are imported from
  `figure_3/motor_decoding/` and not copied.

What is cached is what the fit *produces* and nothing that can be cheaply recomputed: the
drawn neurons (subject/session/unit_id, in draw order, with repeats), the joint fit's
amplitudes and phases, the 1-object cosine phases it started from, and the fit report. The
decode matrix ``W`` (180 × 12,800 floats, ~18 MB per decoder) is rebuilt from those in
milliseconds by :func:`load_decoder`, and the neurons' trials are re-read from the shared
firing-rate cache.

A cache file is keyed by a hash of every hyperparameter the fit depends on, plus the drawn
neurons, so a changed CONFIG or a changed population makes it stale rather than wrong.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
_ROOT = _FIGURES.parent
MOTOR_DIR = _FIGURES / "figure_3" / "motor_decoding"
DECODER_NB = MOTOR_DIR / "decoder_v2.ipynb"
for _p in (str(MOTOR_DIR), str(_FIGURES / "figure_3"), str(_FIGURES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import population_v2 as pv  # noqa: E402  figure 3's v2 pool; re-exports pl, dl

pl, dl = pv.pl, pv.dl

#: In the paper-wide cache, beside every other figure's, not in this directory.
CACHE_ROOT = _ROOT / "cache" / "figures" / "supp_figure_4"
CACHE_DIR = CACHE_ROOT / "decoders"

SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}

#: The CONFIG names the fit depends on — `decoder_v2`'s ``_hyperparameters()``, by name.
FIT_KEYS = (
    "NUM_NEURONS",
    "SEED",
    "NUM_REPLICAS",
    "SAMPLE_SIGMA_DEG",
    "LOBE_WIDTH_DEG",
    "FIT_N_SAMPLES",
    "FIT_K_MAX",
    "FIT_PHASES",
    "FIT_ITERS",
    "EXCLUSION",
)


# ── The hyperparameters, from decoder_v2.ipynb itself ─────────────────────────
def decoder_config():
    """`decoder_v2.ipynb`'s CONFIG cell, executed, as a dict of its upper-case names.

    The cell is plain assignments, so executing it is exactly what the notebook does. Read
    on every call — nothing is frozen at import — so an edit to that notebook is picked up.
    """
    import nbformat

    nb = nbformat.read(DECODER_NB, as_version=4)
    cells = [
        c
        for c in nb.cells
        if c.cell_type == "code" and "# ── CONFIG" in c.source
    ]
    if len(cells) != 1:
        raise RuntimeError(
            f"expected one CONFIG cell in {DECODER_NB}, found {len(cells)}"
        )
    ns = {}
    exec(cells[0].source, {}, ns)
    return {k: v for k, v in ns.items() if k.isupper()}


def _key(cfg, subject, boot):
    blob = json.dumps(
        {
            "subject": subject,
            "boot": int(boot),
            **{k: cfg[k] for k in FIT_KEYS},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _path(subject, boot):
    return CACHE_DIR / f"{subject}_boot{int(boot)}.npz"


# ── The fit: decoder_v2's build_decoder ───────────────────────────────────────
_POOLS = {}
_SESSIONS = {}


def _pool(subject, exclusion):
    if (subject, exclusion) not in _POOLS:
        _POOLS[(subject, exclusion)] = pv.candidate_units(
            subject=subject, exclusion=exclusion, verbose=False
        )
    return _POOLS[(subject, exclusion)]


def _sessions_for(units_df):
    missing = pd.DataFrame(
        [
            r
            for _, r in units_df[["subject", "session"]]
            .drop_duplicates()
            .iterrows()
            if (r["subject"], r["session"]) not in _SESSIONS
        ]
    )
    if len(missing):
        _SESSIONS.update(pl._session_trials(missing))
    return _SESSIONS


def _fit(subject, boot, cfg, verbose=True):
    """`decoder_v2.build_decoder`, unchanged except that it returns the fit's parameters
    rather than a closure over them."""
    n_copies = cfg["NUM_REPLICAS"] * 2
    pool_df = _pool(subject, cfg["EXCLUSION"])
    units_df, counts = pl.sample_units(
        pool_df,
        np.arange(len(pool_df)),
        cfg["NUM_NEURONS"],
        np.random.default_rng(cfg["SEED"] + boot),
    )
    sessions = _sessions_for(units_df)
    units_1obj = pl.load_units_1obj_trials(units_df, session_cache=sessions)

    tune = dl.fit_tuning(units_1obj)
    rng = np.random.default_rng(cfg["SEED"])
    theta_train = rng.uniform(0, 2 * np.pi, cfg["FIT_N_SAMPLES"])
    R_train = pl.population_sample_1obj(
        units_1obj,
        theta_train,
        cfg["NUM_REPLICAS"],
        cfg["SAMPLE_SIGMA_DEG"],
        rng,
        True,
    )
    amps, psi, info = dl.joint_fit(
        R_train,
        theta_train,
        len(units_df),
        n_copies,
        cfg["NUM_REPLICAS"],
        True,
        dl._lobe_weights,
        cfg["LOBE_WIDTH_DEG"],
        psi_init=tune["mu"],
        k_max=cfg["FIT_K_MAX"],
        n_iter=cfg["FIT_ITERS"],
        fit_phases=cfg["FIT_PHASES"],
        verbose=False,
    )

    dpsi = np.degrees(np.abs((psi - tune["mu"] + np.pi) % (2 * np.pi) - np.pi))
    report = (
        f'{subject:6s} boot {boot}: {cfg["NUM_NEURONS"]} draws from {len(pool_df)} '
        f"candidates hit {int((counts > 0).sum())} distinct "
        f'({units_df["session"].nunique()} sessions)\n'
        f'         joint fit: unexplained {info["unexplained"][0]:.3f} -> '
        f'{info["unexplained"][-1]:.3f} | {int((amps < 0).sum())}/{len(amps)} amplitudes '
        f"negative | phase moved {np.median(dpsi):.0f}° (median)"
    )
    if verbose:
        print(report, flush=True)
    return dict(
        units_df=units_df[["subject", "session", "unit_id"]],
        amps=amps,
        psi=psi,
        mu_init=tune["mu"],
        unexplained=np.asarray(info["unexplained"], float),
        n_pool=len(pool_df),
        report=report,
    )


def _save(path, fit, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    u = fit["units_df"]
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        key=key,
        subject=u["subject"].to_numpy(str),
        session=u["session"].to_numpy(str),
        unit_id=u["unit_id"].to_numpy(int),
        amps=fit["amps"],
        psi=fit["psi"],
        mu_init=fit["mu_init"],
        unexplained=fit["unexplained"],
        n_pool=fit["n_pool"],
        report=fit["report"],
    )
    tmp.replace(
        path
    )  # atomic: an interrupted fit never leaves a half-written file


def _read(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["units_df"] = pd.DataFrame(
        {
            "subject": d.pop("subject"),
            "session": d.pop("session"),
            "unit_id": d.pop("unit_id"),
        }
    )
    d["key"], d["report"] = str(d["key"]), str(d["report"])
    return d


def ensure_decoder(subject, boot, cfg=None, verbose=True):
    """Fit and cache one decoder if its cache file is missing or stale; return the fit."""
    cfg = decoder_config() if cfg is None else cfg
    key, path = _key(cfg, subject, boot), _path(subject, boot)
    if path.exists():
        d = _read(path)
        if d["key"] == key:
            return d
    t = time.time()
    fit = _fit(subject, boot, cfg, verbose=verbose)
    _save(path, fit, key)
    if verbose:
        print(
            f"         fit in {time.time() - t:.0f} s, cached to {path.name}",
            flush=True,
        )
    return _read(path)


def ensure_all(cfg=None, verbose=True):
    """Every ``(subject, boot)`` decoder `decoder_v2` uses, fitting only what is not cached."""
    cfg = decoder_config() if cfg is None else cfg
    for s in cfg["SUBJECTS"]:
        for b in range(cfg["N_BOOTSTRAPS"]):
            ensure_decoder(s, b, cfg, verbose=verbose)


def load_decoder(subject, boot, cfg=None, verbose=False):
    """A ready-to-use decoder: the cached fit, the neurons' trials, and the decode matrix.

    Returns a dict with ``units_1obj``, ``units_2obj``, ``W`` and ``decode`` — the same
    things `decoder_v2.build_decoder` returns, plus the fit's parameters and report.
    """
    cfg = decoder_config() if cfg is None else cfg
    d = ensure_decoder(subject, boot, cfg, verbose=verbose)
    n_copies = cfg["NUM_REPLICAS"] * 2
    sessions = _sessions_for(d["units_df"])
    d["units_1obj"] = pl.load_units_1obj_trials(
        d["units_df"], session_cache=sessions
    )
    d["units_2obj"] = pl.load_units_2obj_trials(
        d["units_df"], session_cache=sessions
    )
    mu_eff = pl.replica_preferred_directions(
        d["psi"], cfg["NUM_REPLICAS"], True
    )
    W = dl.decoder_weights_from_means(
        mu_eff,
        dl.decode_grid(),
        width_deg=cfg["LOBE_WIDTH_DEG"],
        amplitudes=np.repeat(d["amps"], n_copies),
    )
    d["W"] = W
    d["decode"] = lambda r, W=W: dl.decode(W, r, rectify=False)
    return d


# ── 1-object read-out ─────────────────────────────────────────────────────────
def response_1obj(dec, theta, cfg):
    """``[n_theta, n_inputs]`` expected population response — `decoder_v2.response_1obj`."""
    return pl.population_response_1obj(
        dec["units_1obj"],
        theta,
        cfg["NUM_REPLICAS"],
        cfg["SAMPLE_SIGMA_DEG"],
        True,
    )


def sample_1obj(dec, theta, cfg, rng):
    """``[n_theta, n_inputs]`` single-trial population responses — what the decoder is fit on."""
    return pl.population_sample_1obj(
        dec["units_1obj"],
        theta,
        cfg["NUM_REPLICAS"],
        cfg["SAMPLE_SIGMA_DEG"],
        rng,
        True,
    )


def _refine(grid, values, phi):
    """Sub-grid location of the maximum at grid angle `phi`: the vertex of the parabola
    through it and its two (circular) neighbours. The decode grid is 2 deg, so without this a
    maximum can only be reported to the nearest 2 deg and every error is a multiple of it.
    """
    n = len(grid)
    i = int(np.argmin(np.abs((grid - phi + np.pi) % (2 * np.pi) - np.pi)))
    y0, y1, y2 = values[(i - 1) % n], values[i], values[(i + 1) % n]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return grid[i] + np.clip(off, -0.5, 0.5) * (2 * np.pi / n)


def errors_1obj(dec, theta, R, cfg, refine=False):
    """Signed error (deg, in (-180, 180]) of the decode's **nearest local maximum** to the
    object, per row of ``R`` — `nearest_max_errors`' convention, with one object. NaN where
    the decode has no maximum. Positive = the decode sits counter-clockwise of the object.

    ``refine=False`` is `decoder_v2`'s convention exactly (a maximum is a grid point);
    ``refine=True`` places it between grid points by :func:`_refine`.
    """
    grid = dl.decode_grid()
    out = np.full(len(theta), np.nan)
    for i, (t, r) in enumerate(zip(theta, R)):
        D = dec["decode"](r)
        peaks = dl.local_maxima(
            grid, D, min_height_frac=cfg["PEAK_MIN_HEIGHT_FRAC"]
        )
        if len(peaks):
            if refine:
                peaks = np.array([_refine(grid, D, ph) for ph in peaks])
            d = (peaks - t + np.pi) % (2 * np.pi) - np.pi
            out[i] = np.degrees(d[np.argmin(np.abs(d))])
    return out


# ── The 1-object error sweep, cached per decoder ──────────────────────────────
SWEEP_DIR = CACHE_ROOT / "errors_1obj"


def error_sweep(
    subject, boot, step_deg, n_samples, seed, refine, cfg=None, verbose=True
):
    """1-object errors around the ring for one decoder, as a long table.

    Two kinds of row: ``kind='mean'``, the decode of the population's **expected** response at
    each angle (what `decoder_v2`'s ring panels show); and ``kind='trial'``, decodes of
    **single-trial** population responses (what the decoder is fit on), ``n_samples`` per
    angle. Sample ``k`` is drawn with ``default_rng([seed, boot, k, subject code])``, so raising
    `n_samples` computes only the new samples: the cache holds a ``sample`` column rather than
    being keyed on how many there are.
    """
    cfg = decoder_config() if cfg is None else cfg
    dec_key = _key(cfg, subject, boot)
    tag = f"{subject}_boot{int(boot)}_step{step_deg:g}_seed{seed}_refine{int(refine)}"
    path = SWEEP_DIR / f"{tag}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old) and (old["decoder_key"].iloc[0] != dec_key):
        old = pd.DataFrame()  # the decoder changed: start over
    theta_deg = np.arange(0.0, 360.0, step_deg)
    theta = np.radians(theta_deg)
    have_mean = len(old) and (old["kind"] == "mean").any()
    have = (
        set(old.loc[old["kind"] == "trial", "sample"].unique())
        if len(old)
        else set()
    )
    todo = [k for k in range(n_samples) if k not in have]
    if have_mean and not todo:
        return old[
            (old["kind"] == "mean") | (old["sample"] < n_samples)
        ].reset_index(drop=True)

    dec = load_decoder(subject, boot, cfg)
    parts = [old] if len(old) else []
    if not have_mean:
        e = errors_1obj(
            dec, theta, response_1obj(dec, theta, cfg), cfg, refine
        )
        parts.append(
            pd.DataFrame(
                dict(kind="mean", sample=-1, theta_deg=theta_deg, err_deg=e)
            )
        )
    code = sum(map(ord, subject))
    for k in todo:
        rng = np.random.default_rng([seed, int(boot), k, code])
        e = errors_1obj(
            dec, theta, sample_1obj(dec, theta, cfg, rng), cfg, refine
        )
        parts.append(
            pd.DataFrame(
                dict(kind="trial", sample=k, theta_deg=theta_deg, err_deg=e)
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df["subject"], df["boot"], df["decoder_key"] = subject, int(boot), dec_key
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    if verbose:
        print(
            f'  {subject} boot {boot}: {"mean + " if not have_mean else ""}{len(todo)} new '
            f"single-trial sweeps cached",
            flush=True,
        )
    return df[(df["kind"] == "mean") | (df["sample"] < n_samples)].reset_index(
        drop=True
    )
