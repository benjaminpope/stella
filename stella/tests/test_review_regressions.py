"""Regression tests added after the 2026-11 review (docs/review_stella.md in benjaminpope/srsc).

Fixture: AU Mic (TIC 441420236), TESS sector 1, SPOC 2-min PDCSAP, public data,
stored in tests/data/aumic_s1.npz (time, flux, flux_err, quality). Expected
probabilities in aumic_s1_expected.npz were computed from the ORIGINAL TF2 .h5
weights (final_models.zip) with an independent NumPy/JAX forward pass.
None of these tests needs the network.
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[2]


def _aumic(invert=False):
    d = np.load(DATA / "aumic_s1.npz")
    exp = np.load(DATA / "aumic_s1_expected.npz")
    t, f, e, q = d["time"], d["flux"].astype(float), d["flux_err"].astype(float), d["quality"]
    m = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & ((q & int(exp["bitmask"])) == 0)
    t, f, e = t[m], f[m], e[m]
    if invert:
        f = 2 * np.nanmedian(f) - f
    return t, f, e


def _h5_arrays(fobj):
    h5py = pytest.importorskip("h5py")
    out = []
    with h5py.File(fobj, "r") as f:
        f.visititems(lambda n, o: out.append((n, o[()])) if isinstance(o, h5py.Dataset) else None)
    return out


# ---------------------------------------------------------------- weights
def test_keras_weights_equal_original_h5():
    from stella import models as sm

    zp = ROOT / "final_models.zip"
    if not zp.exists():
        pytest.skip("final_models.zip not in this checkout")
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.startswith("run01/") and n.endswith(".h5")]
        assert len(names) == 10
        for n in names:
            seed = n.split("_")[1]
            orig = _h5_arrays(io.BytesIO(z.read(n)))
            orig = [a for k, a in sorted(orig, key=lambda t: (t[0].split("/")[1], 0 if "kernel" in t[0] else 1))
                    if k.startswith("model_weights/")]
            kp = sm.get_model_path(f"ensemble_{seed}_i0350_b0.73_savedmodel.keras")
            with zipfile.ZipFile(kp) as kz:
                new = [a for k, a in _h5_arrays(io.BytesIO(kz.read("model.weights.h5"))) if np.ndim(a) > 0]
            assert len(orig) == len(new) == 8
            for a, b in zip(orig, new):
                assert a.shape == b.shape and np.array_equal(a, b), seed


# ---------------------------------------------------------------- NumPy 2
def test_no_removed_numpy_functions_in_source():
    import stella

    src = Path(stella.__file__).parent
    for p in src.glob("*.py"):
        text = p.read_text()
        for bad in ("np.trapz(", "np.float_(", "np.product(", "np.in1d(", "np.row_stack("):
            assert bad not in text, f"{bad} in {p.name}"


def test_fitflares_fits_a_synthetic_flare():
    """Exercises the equivalent-duration integral (np.trapz crashed under NumPy >= 2.4)."""
    from stella.mark_flares import FitFlares
    from stella.utils import flare_lightcurve

    rng = np.random.default_rng(0)
    t = np.arange(3000) * 2 / 1440
    model, _ = flare_lightcurve(t, 1500, 0.05, 0.0008, 0.005)
    f = model + rng.normal(0, 1e-3, t.size)
    p = np.zeros_like(t)
    p[1498:1503] = 0.95
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[p])
    ff.identify_flare_peaks(threshold=0.5)
    assert len(ff.flare_table) == 1
    row = ff.flare_table[0]
    assert abs(row["tpeak"] - t[1500]) < 1e-6
    true_ed = 0.05 * (np.sqrt(np.pi / 2) * 0.0008 + 0.005) * 86400  # analytic integral, about 25.9 s
    assert row["ed_s"] == pytest.approx(true_ed, rel=0.3)  # a no-op fit returns about 4.5 s
    assert row["fall"] > 0.003  # injected 0.005 d; the starting point is 0.001 d


def test_fitflares_no_duplicate_rows():
    """Two probability groups within 50 cadences resolved to the same peak and were reported twice."""
    from stella.mark_flares import FitFlares
    from stella.utils import flare_lightcurve

    rng = np.random.default_rng(1)
    t = np.arange(3000) * 2 / 1440
    model, _ = flare_lightcurve(t, 1500, 0.05, 0.0008, 0.005)
    f = model + rng.normal(0, 1e-3, t.size)
    p = np.zeros_like(t)
    p[1498:1503] = 0.95
    p[1520:1523] = 0.95  # second group in the decay
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[p])
    ff.identify_flare_peaks(threshold=0.5)
    assert len(ff.flare_table) == len(np.unique(ff.flare_table["tpeak"])) == 1


# ---------------------------------------------------------------- windows and gaps
def test_make_windows_alignment_and_gaps():
    from stella.neural_network import make_windows

    t = np.arange(1000) * 2 / 1440
    t[600:] += 1.0  # one gap between cadences 599 and 600
    lc = np.arange(1000, dtype=float)
    X, scored = make_windows(t, lc, 200)
    i = 300
    assert scored[i] and X[i][100] == lc[i] and X[i][0] == lc[i - 100] and X[i][-1] == lc[i + 99]
    assert not scored[:100].any() and not scored[-99:].any()
    # windows that would include both cadence 599 and 600 are not scored
    crossing = [k for k in range(100, 901) if k - 100 <= 599 and k + 99 >= 600]
    assert not scored[crossing].any()
    assert scored[500] and scored[700]


def test_predict_gap_near_end_and_ragged_inputs():
    from stella import models as sm
    from stella.neural_network import ConvNN

    t, f, e = _aumic()
    t2 = t.copy()
    t2[-50:] += 1.0  # gap inside the last 100 cadences: IndexError before the fix
    cnn = ConvNN(output_dir=".")
    cnn.predict(sm.get_model_path(), [t2, t[:5000]], [f, f[:5000]], [e, e[:5000]], verbose=False)
    p0, p1 = cnn.predictions
    assert len(p0) == len(t2) and len(p1) == 5000
    assert np.all(np.isnan(p0[-149:]))  # every window touching the gap or the end is unscored
    from stella.neural_network import make_windows

    _, scored1 = make_windows(t[:5000], f[:5000], 200)
    assert np.isnan(p1[:100]).all() and scored1.mean() > 0.5
    np.testing.assert_array_equal(np.isfinite(p1), scored1)  # NaN exactly where unscored


# ---------------------------------------------------------------- fixed probabilities
def test_fixed_probabilities_one_sector():
    from stella import models as sm
    from stella.neural_network import ConvNN

    t, f, e = _aumic()
    exp = np.load(DATA / "aumic_s1_expected.npz")
    paths = sm.list_model_paths()
    assert [p.split("ensemble_")[1][:5] for p in paths] == list(exp["seeds"])
    cnn = ConvNN(output_dir=".")
    for k, path in enumerate(paths):
        cnn.predict(path, t, f, e, verbose=False)
        got = np.asarray(cnn.predictions[0])[exp["idx"]]
        np.testing.assert_allclose(got, exp["probs"][k], atol=1e-5, equal_nan=True)


def test_ensemble_median_runs_and_is_float():
    from stella import models as sm
    from stella import pipeline

    t, f, e = _aumic()
    t, f, e = t[:3000], f[:3000], e[:3000]
    tt, ff, ee, agg, per = pipeline.predict_ensemble(sm.models[:2], t, f, e, aggregate="median", verbose=False)
    assert per.dtype == float and agg.dtype == float and per.shape == (2, len(tt))


# ---------------------------------------------------------------- science
def test_recovers_known_flare():
    from stella import models as sm
    from stella import pipeline

    t, f, e = _aumic()
    exp = np.load(DATA / "aumic_s1_expected.npz")
    tp = float(exp["flare_tpeak"])
    tt, ff, ee, preds, table = pipeline.predict_and_mark(sm.models, t, f, e, threshold=0.5,
                                                        aggregate="median", verbose=False)
    assert np.nanmax(preds[np.abs(tt - tp) < 0.01]) > 0.9
    assert np.any(np.abs(np.asarray(table["tpeak"]) - tp) < 0.01)


def test_inverted_lightcurve_gives_no_accepted_flares():
    """Flux-inverted data: every real flare becomes a dip, so any accepted flare is false."""
    from stella import models as sm
    from stella import pipeline

    t, f, e = _aumic(invert=True)
    *_, table = pipeline.predict_and_mark(sm.models, t, f, e, threshold=0.5, aggregate="median", verbose=False)
    assert len(table) == 0


def test_eclipsing_binary_no_crash_and_flag():
    from stella import models as sm
    from stella import pipeline
    from stella.diagnostics import dip_diagnostics

    rng = np.random.default_rng(2)
    t = 1400 + np.arange(18000) * 2 / 1440
    P, dur, depth = 1.268, 0.05, 0.4  # CM Dra-like
    ph = ((t - 1400.3) / P + 0.5) % 1 - 0.5
    f = 1 - depth * np.clip(1 - np.abs(ph * P) / (dur / 2), 0, None)
    ph2 = ((t - 1400.3 - P / 2) / P + 0.5) % 1 - 0.5
    f -= 0.9 * depth * np.clip(1 - np.abs(ph2 * P) / (dur / 2), 0, None)
    f = f + rng.normal(0, 1e-3, t.size)
    e = np.full_like(t, 1e-3)
    tt, ff, ee, preds, table = pipeline.predict_and_mark(sm.models[:3], t, f, e, threshold=0.5,
                                                        aggregate="median", verbose=False)
    diag = dip_diagnostics(tt, ff)
    assert diag["eclipse_like"] and diag["dips_periodic"]
    assert table.meta["eclipse_like"] and table.meta["n_dips"] == diag["n_dips"]
    assert "near_dip" in table.colnames  # present even when the table is empty
    # every candidate this light curve produces lies at an eclipse
    near = np.asarray(table["near_dip"])
    assert near.size == 0 or near.mean() > 0.5


# ---------------------------------------------------------------- quality mask and columns
def test_flare_safe_bitmask_keeps_impulsive_outlier_cadences():
    from lightkurve import LightCurve
    from stella.pipeline import FLARE_SAFE_BITMASK, _extract_series

    assert FLARE_SAFE_BITMASK == 17087 & ~512 and FLARE_SAFE_BITMASK & 512 == 0
    t = np.arange(10, dtype=float)
    q = np.zeros(10, dtype=int)
    q[3] = 512  # impulsive outlier: the peak of a flare, keep
    q[5] = 1  # attitude tweak: drop
    q[7] = 32  # reaction wheel desaturation: drop
    lc = LightCurve(time=t, flux=np.ones(10), flux_err=np.full(10, 0.01), quality=q)
    tt, ff, ee = _extract_series(lc)
    assert list(tt) == [0, 1, 2, 3, 4, 6, 8, 9]


def test_fitflares_vetting_columns_and_min_amp():
    from stella.mark_flares import FitFlares
    from stella.utils import flare_lightcurve

    rng = np.random.default_rng(3)
    t = np.arange(3000) * 2 / 1440
    model, _ = flare_lightcurve(t, 1500, 0.05, 0.0008, 0.005)
    f = model + rng.normal(0, 1e-3, t.size)
    p = np.zeros_like(t)
    p[1498:1503] = 0.95
    kw = dict(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[p])
    ff = FitFlares(**kw)
    ff.identify_flare_peaks(threshold=0.5)
    row = ff.flare_table[0]
    assert row["prob_max"] == pytest.approx(0.95) and row["snr_peak"] > 20 and row["n_consec_3sigma"] >= 3
    assert row["dist_gap"] > 100 and not row["fit_at_bound"]
    ff = FitFlares(**kw, min_amp=10.0)  # the amplitude cut is a parameter
    ff.identify_flare_peaks(threshold=0.5)
    assert len(ff.flare_table) == 0 and "snr_peak" in ff.flare_table.colnames


def test_fitflares_flare_near_end_and_start_does_not_crash():
    from stella.mark_flares import FitFlares
    from stella.utils import flare_lightcurve

    t = np.arange(600) * 2 / 1440
    f = np.ones_like(t)
    for k in (3, 598):
        m, _ = flare_lightcurve(t, k, 0.05, 0.0008, 0.005)
        f = f + (m - 1)
    p = np.zeros_like(t)
    p[[3, 598]] = 0.9
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[p])
    ff.identify_flare_peaks(threshold=0.5)
    assert ff.n_skipped == 2 and len(ff.flare_table) == 0


# ---------------------------------------------------------------- review round 2 (PR #7)
def _flare_lc(seed=0, rise=0.0008, fall=0.005, n=3000):
    from stella.utils import flare_lightcurve

    rng = np.random.default_rng(seed)
    t = np.arange(n) * 2 / 1440
    model, _ = flare_lightcurve(t, 1500, 0.05, rise, fall)
    return t, model, model + rng.normal(0, 1e-3, t.size)


def test_fitflares_weak_precursor_does_not_replace_the_flare_group():
    """A 0.15-probability group 30 cadences before a 0.98 flare resolves to the same peak at
    threshold 0.1; the candidate must carry the flare's prob_max, not the precursor's."""
    from stella.mark_flares import FitFlares

    t, _, f = _flare_lc(1)
    p = np.zeros_like(t)
    p[1465:1468] = 0.15
    p[1498:1503] = 0.98
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[p])
    ff.identify_flare_peaks(threshold=0.1)
    assert len(ff.flare_table) == 1
    row = ff.flare_table[0]
    assert row["prob_max"] == pytest.approx(0.98) and row["n_above"] == 5


def test_fitflares_group_metadata_stays_aligned_with_peaks():
    from stella.mark_flares import FitFlares

    t, _, f = _flare_lc(2)
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[np.full_like(t, 1e-3)], predictions=[np.zeros_like(t)])
    ff.threshold = 0.5
    p = np.zeros_like(t)
    p[1498:1503] = 0.9
    groups = ff.group_inds(np.where(p >= 0.5)[0])
    assert groups.dtype == object and groups.ndim == 1 and len(groups) == 1
    guesses = ff.get_init_guesses(groups, t, f, np.full_like(t, 1e-3), p, 2, 50)
    assert len(guesses) == 1 and guesses[0][2:] == (5, pytest.approx(0.9))


@pytest.mark.parametrize("err_kind", ["good", "zeros"])
def test_fitflares_fit_matches_the_injected_flare(err_kind):
    """The fit must move away from its starting point (fall 0.001 d), with or without flux errors."""
    from stella.mark_flares import FitFlares

    t, model, f = _flare_lc(3)
    e = np.full_like(t, 1e-3) if err_kind == "good" else np.zeros_like(t)
    p = np.zeros_like(t)
    p[1498:1503] = 0.95
    ff = FitFlares(id=[1], time=[t], flux=[f], flux_err=[e], predictions=[p])
    ff.identify_flare_peaks(threshold=0.5)
    row = ff.flare_table[0]
    true_ed = np.trapezoid(model - 1, t) * 86400 if hasattr(np, "trapezoid") else np.trapz(model - 1, t) * 86400
    assert row["ed_s"] == pytest.approx(true_ed, rel=0.3)
    assert row["fall"] > 0.003  # the injected fall is 0.005 d; the starting point is 0.001 d


def test_dips_periodic_has_positive_degrees_of_freedom():
    from stella.diagnostics import _dips_periodic

    assert not _dips_periodic([1.0, 2.718, 13.142])  # any three times used to pass
    assert not _dips_periodic([0.0, 3.0, 6.0])  # equal spacing of three is not evidence either
    rng = np.random.default_rng(0)
    for n in (3, 4, 5, 6, 8):  # random dip times: a few per cent at most
        assert sum(_dips_periodic(np.sort(rng.uniform(0, 25, n))) for _ in range(200)) <= 20
    assert _dips_periodic([0.0, 3.0, 6.0, 9.0])  # one phase, four cycles
    assert _dips_periodic([0.0, 1.4, 3.0, 4.4, 6.0, 7.4])  # primary and secondary
    assert _dips_periodic([0.4, 3.0, 4.4, 6.0, 7.4, 9.0])  # an isolated dropout comes first


def test_dip_diagnostics_and_baseline_do_not_read_a_step_across_a_gap_as_a_dip():
    from stella.diagnostics import dip_diagnostics

    rng = np.random.default_rng(0)
    t = np.arange(4000) * 2 / 1440
    t[2000:] += 1.0  # downlink gap
    f = 1 + rng.normal(0, 1e-3, t.size)
    f[2000:] += 0.02  # the star's rotation moved it by 2% across the gap
    assert dip_diagnostics(t, f)["n_dips"] == 0
    from stella.diagnostics import local_baseline

    b = local_baseline(f, time=t)
    assert abs(b[1990] - 1.0) < 2e-3 and abs(b[2010] - 1.02) < 2e-3


def test_n_consec_counts_the_rise_cadence_before_the_peak():
    from stella.diagnostics import candidate_significance

    rng = np.random.default_rng(1)
    t = np.arange(3000) * 2 / 1440
    f = 1 + rng.normal(0, 1e-3, t.size)
    f[1999], f[2000] = f[1999] + 0.008, f[2000] + 0.01  # a two-cadence flare, peak on the second
    snr, n = candidate_significance(t, f, [t[2000]])
    assert snr[0] > 5 and n[0] == 2


def test_predict_without_errors_and_with_raw_counts():
    from stella import models as sm
    from stella.neural_network import ConvNN

    t, f, e = _aumic()
    t, f, e = t[:1500], f[:1500], e[:1500]
    cnn = ConvNN(output_dir=".")
    cnn.predict(sm.get_model_path(), t, f, None, verbose=False)  # errs=None crashed with IndexError
    assert np.all(np.asarray(cnn.predict_err[0]) == 0)
    cnn.predict(sm.get_model_path(), t, f * 1e4, e * 1e4, verbose=False)  # raw-count scale
    assert np.nanmedian(cnn.predict_err[0]) < 0.05  # relative, like the normalised flux
    np.testing.assert_allclose(np.nanmedian(cnn.predict_flux[0]), 1.0)


def test_rotations_three_sectors_long_period_flags_without_crashing():
    from astropy.table import Table
    from stella.rotations import MeasureProt

    mp = MeasureProt.__new__(MeasureProt)
    mp.IDs = np.array([7, 7, 7])
    tab = Table({"Target_ID": [7, 7, 7], "period_days": [12.5, 12.5, 12.5],
                 "secondary_period_days": [6.25, 6.25, 6.25], "oflag1": [0, 0, 0], "oflag2": [0, 0, 0]})
    out = mp.averaged_per_sector(tab)  # np.full(2, 3) used to fail on the flag assignment
    assert list(out["Flags"]) == [2, 2, 2]
    np.testing.assert_allclose(out["avg_period_days"], 12.5)
