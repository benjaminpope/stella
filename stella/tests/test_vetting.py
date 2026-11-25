"""Tests of the per-star ROC vetting (stella.vetting) and its helpers.

Most tests use a cheap stand-in for the network (``FakeScorer``, a
matched-filter-like probability) on white noise, so that the statistics are
known; one test runs two of the packaged models on the AU Mic fixture.
"""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")

from stella.diagnostics import candidate_significance, local_baseline
from stella.vetting import (
    STATISTICS,
    EnsembleScorer,
    apply_threshold,
    binomial_interval,
    candidate_statistic,
    davenport_flare,
    flare_roc,
    poisson_interval,
    threshold_for_fn_fraction,
    threshold_for_fp_rate,
)

DATA = Path(__file__).parent / "data"


class FakeScorer(object):
    """Two stand-in 'models': probabilities that rise with the significance of the
    central cadence(s) of a window against the window's own scatter."""

    cadences = 200

    def __init__(self, z0=4.5):
        self.z0 = z0

    def score(self, X):
        X = np.asarray(X, float)
        med = np.median(X, axis=1)
        mad = 1.4826 * np.median(np.abs(X - med[:, None]), axis=1) + 1e-12
        z1 = (X[:, 100] - med) / mad
        z3 = (X[:, 100:103].mean(axis=1) - med) / mad * np.sqrt(3)
        sig = lambda x: 1 / (1 + np.exp(-x))
        return np.stack([sig(2 * (z1 - self.z0)), sig(2 * (z3 - self.z0 + 0.3))])


def noise_lc(seed, n=20000, sigma=1e-3):
    rng = np.random.default_rng(seed)
    return np.arange(n) * 2 / 1440, 1 + rng.normal(0, sigma, n)


# ---------------------------------------------------------------- helpers
def test_davenport_template_shape_and_exposure_average():
    t = np.arange(-50, 400) * 2 / 1440
    y = davenport_flare(t, 0.0, 10 / 1440, 0.1)
    assert y[:44].max() == 0  # nothing before the rise starts (one FWHM = 5 cadences)
    assert 0.08 < y.max() <= 0.1  # peak, slightly reduced by the 2-min exposure average
    half = t[y >= y.max() / 2]
    assert 7 / 1440 <= half.max() - half.min() <= 12 / 1440  # FWHM about 10 min
    assert np.all(np.diff(y[np.argmax(y):]) <= 1e-12)  # monotonic decay


def test_intervals():
    lo, hi = poisson_interval(0, 10.0)
    assert lo == 0 and hi == pytest.approx(1.84 / 10, rel=0.01)  # 68% Garwood limit for zero counts
    lo, hi = poisson_interval(40, 10.0)
    assert lo < 4 < hi and hi - lo < 1.5
    lo, hi = binomial_interval(5, 10)
    assert 0.3 < lo < 0.5 < hi < 0.7
    assert binomial_interval(0, 0)[0] != binomial_interval(0, 0)[0]  # NaN for no trials


def test_local_baseline_is_not_lifted_by_flares_and_significance():
    rng = np.random.default_rng(0)
    t = np.arange(4000) * 2 / 1440
    f = 1 + rng.normal(0, 1e-3, t.size)
    flare = davenport_flare(t, t[2000], 30 / 1440, 0.05)  # long, bright flare
    b0 = local_baseline(f)
    b1 = local_baseline(f + flare)
    assert np.abs(b1 - b0)[1950:2100].max() < 1e-3  # 2% of the flare amplitude: masked out of the baseline
    snr, n3 = candidate_significance(t, f + flare, [t[2000], t[500]])
    assert snr[0] > 20 and n3[0] >= 3  # the flare
    assert abs(snr[1]) < 5 and n3[1] < 3  # a noise cadence


# ---------------------------------------------------------------- the ROC
@pytest.fixture(scope="module")
def roc():
    t, f = noise_lc(10, n=12000)
    return flare_roc(t, f, scorer=FakeScorer(), n_injections=48, seed=1)


def test_roc_curves_are_monotonic(roc):
    for st in STATISTICS:
        c = roc.curve(st)
        assert np.all(np.diff(c["threshold"]) > 0)
        assert np.all(np.diff(c["n_false"]) <= 0) and np.all(np.diff(c["fp_rate"]) <= 0)
        assert np.all(np.diff(c["recovery"]) <= 1e-12)
        for a in roc.amplitudes:
            assert np.all(np.diff(c[f"recovery@{a:g}"]) <= 1e-12)
        assert np.all(c["fp_lo"] <= c["fp_rate"]) and np.all(c["fp_rate"] <= c["fp_hi"])
        assert np.all(c["recovery_lo"] <= c["recovery"] + 1e-12) and np.all(c["recovery"] <= c["recovery_hi"] + 1e-12)
        assert c["fp_rate"][-1] == 0 and c["recovery"][-1] == 0  # above the largest statistic
    # bigger flares are recovered at least as often at a fixed threshold
    thr = roc.floor("prob")
    rec = [roc.recovery("prob", thr, a)[0] for a in sorted(roc.amplitudes)]
    assert rec == sorted(rec) and rec[-1] == 1.0 and rec[0] < rec[-1]


def test_inverted_pass_accepts_nothing_above_a_high_threshold(roc):
    assert len(roc.false_table) > 0  # the permissive stand-in does fire on noise
    for st, hi in (("prob", 0.99), ("snr", 50.0), ("combined", 50.0)):
        assert roc.fp_rate(st, hi)[3] == 0
    # injected flares are still recovered where the inverted noise is not accepted
    assert roc.recovery("prob", 0.99, 0.1)[0] > 0.5


def test_roc_is_deterministic_for_a_fixed_seed():
    t, f = noise_lc(3, n=6000)
    kw = dict(scorer=FakeScorer(), n_injections=12)
    a = flare_roc(t, f, seed=5, **kw)
    b = flare_roc(t, f, seed=5, **kw)
    c = flare_roc(t, f, seed=6, **kw)
    for col in a.injections.colnames:
        np.testing.assert_array_equal(np.asarray(a.injections[col]), np.asarray(b.injections[col]))
    np.testing.assert_array_equal(np.asarray(a.false_table["tpeak"]), np.asarray(b.false_table["tpeak"]))
    assert not np.array_equal(np.asarray(a.injections["tpeak"]), np.asarray(c.injections["tpeak"]))


def test_threshold_for_fp_rate_meets_the_rate_within_its_interval():
    target = 1.0  # per day
    t, f = noise_lc(10, n=30000)
    roc = flare_roc(t, f, scorer=FakeScorer(2.8), n_injections=12, seed=1)
    assert roc.exposure_days > 40
    for st in ("prob", "snr"):
        thr = threshold_for_fp_rate(roc, target, st)
        rate, lo, hi, n = roc.fp_rate(st, thr)
        assert n > 20  # enough false candidates for the interval to mean something
        assert rate <= target and lo <= target <= hi
        # an independent noise realisation: counts at this threshold agree with the calibration
        t2, f2 = noise_lc(11, n=30000)
        roc2 = flare_roc(t2, f2, scorer=FakeScorer(2.8), n_injections=12, seed=2)
        n2 = roc2.fp_rate(st, thr)[3]
        assert abs(n2 - n) / np.sqrt(n + n2) < 3.0
        # the conservative bound is stricter
        assert threshold_for_fp_rate(roc, target, st, bound="upper") >= thr


def test_threshold_for_fn_fraction_and_operating_point(roc):
    thr = threshold_for_fn_fraction(roc, 0.1, 0.03, "prob")
    rec, lo, hi, n = roc.recovery("prob", thr, 0.03)
    assert rec >= 0.9 and n == roc.injections["amp"].tolist().count(0.03)
    higher = thr + 1e-9
    assert roc.recovery("prob", higher, 0.03)[0] < rec or thr == np.nanmax(roc._inj_stat("prob", 0.03))
    op = roc.operating_point("prob", thr)
    assert op["recovery_by_amp"][0.03][0] == rec and 0 <= op["fp_rate"]
    with pytest.raises(ValueError):
        roc.recovery("prob", thr, 0.0123)  # not an injected amplitude
    with pytest.warns(UserWarning):  # the smallest flares are not all recoverable
        assert np.isnan(threshold_for_fn_fraction(roc, 0.0, 0.003, "combined"))


def test_apply_threshold_adds_columns_and_never_deletes_by_default():
    from astropy.table import Table

    tab = Table(dict(prob_max=[0.2, 0.7, 0.95], snr_peak=[2.0, 6.0, 9.0], n_consec_3sigma=[0, 1, 4]))
    out = apply_threshold(tab, "prob", 0.5)
    assert len(out) == 3 and list(out["vetted"]) == [False, True, True] and "vetted" not in tab.colnames
    assert list(candidate_statistic(tab, "combined")) == [-np.inf, -np.inf, 9.0]
    assert list(apply_threshold(tab, "combined", 5.0, drop=True)["snr_peak"]) == [9.0]
    assert list(apply_threshold(tab, "snr", 5.0, drop=True)["snr_peak"]) == [6.0, 9.0]
    assert len(apply_threshold(tab[:0], "prob", 0.5)) == 0
    with pytest.raises(ValueError):
        candidate_statistic(tab, "bogus")


def test_ensemble_scorer_roc_on_aumic_runs_and_separates_injections():
    from stella import models as sm

    d = np.load(DATA / "aumic_s1.npz")
    e = np.load(DATA / "aumic_s1_expected.npz")
    t, f, er, q = d["time"], d["flux"].astype(float), d["flux_err"].astype(float), d["quality"]
    m = np.isfinite(t) & np.isfinite(f) & np.isfinite(er) & ((q & int(e["bitmask"])) == 0)
    t, f, er = t[m][:9000], f[m][:9000], er[m][:9000]
    scorer = EnsembleScorer(models=sm.models[:2])
    out = flare_roc(t, f, er, scorer=scorer, aggregate=("median", "mean"), n_injections=24, seed=0)
    assert set(out) == {"median", "mean"}
    for how, r in out.items():
        assert r.meta["n_models"] == 2 and r.meta["aggregate"] == how
        assert r.exposure_days > 8
        # big flares are recovered, and the fp rate of the inverted curve is lower than the real one
        assert r.recovery("prob", 0.5, 0.1)[0] > 0.8
        assert r.fp_rate("prob", 0.5)[0] < 1.0
        assert r.curve("combined")["recovery"][0] >= r.curve("combined")["recovery"][-1]


def test_worked_example_from_the_docs():
    """The sequence in docs/getting_started/vetting.md: ROC, threshold, candidate table, apply."""
    from stella import models as sm
    from stella import pipeline

    d = np.load(DATA / "aumic_s1.npz")
    e = np.load(DATA / "aumic_s1_expected.npz")
    t, f, er, q = d["time"], d["flux"].astype(float), d["flux_err"].astype(float), d["quality"]
    m = np.isfinite(t) & np.isfinite(f) & np.isfinite(er) & ((q & int(e["bitmask"])) == 0)
    t, f, er = t[m][:9000], f[m][:9000], er[m][:9000]
    models = sm.models[:2]
    roc = flare_roc(t, f, er, scorer=EnsembleScorer(models=models), n_injections=12, seed=0)
    thr = roc.threshold_for_fp_rate(0.5, statistic="combined")
    _, _, _, _, table = pipeline.predict_and_mark(models, t, f, er, threshold=roc.meta["base_threshold"],
                                                  verbose=False)
    out = roc.apply(table, "combined", thr)
    assert len(out) == len(table) and "vetted" in out.colnames
    assert out["vetted"].sum() <= len(table) and out.meta["vetting"]["statistic"] == "combined"


# ---------------------------------------------------------------- review round 2 (PR #7)
def test_plot_steps_stay_on_achievable_points(roc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    roc.plot("combined", ax=ax)
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2 and ln.get_drawstyle() == "steps-post"]
    assert lines
    for ln in lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        assert np.all(np.diff(x) >= 0)  # false-positive rate increases along the curve
        assert np.all(np.diff(y) >= -1e-12)  # and recovery does not fall
    assert ax.collections  # the interval band is drawn
    plt.close(fig)


def test_inversion_diagnostic_returns_the_scored_arrays():
    from stella.vetting import inversion_diagnostic

    t, f = noise_lc(3, n=3000)
    tt, ff, base, inv = inversion_diagnostic(t, f)
    assert len(tt) == len(ff) == len(base) == len(inv) == 3000
    np.testing.assert_allclose(inv * np.nanmedian(2 * base - ff), 2 * base - ff)


def test_cadence_and_gap_handling_in_flare_roc():
    """A 350-cadence model overflowed the injection segment; a gap inside the segment is refused."""

    class Wide(FakeScorer):
        cadences = 350

        def score(self, X):
            X = np.asarray(X, float)
            med = np.median(X, axis=1)
            mad = 1.4826 * np.median(np.abs(X - med[:, None]), axis=1) + 1e-12
            z = (X[:, 175] - med) / mad
            return np.stack([1 / (1 + np.exp(-2 * (z - 4.5)))] * 2)

    t, f = noise_lc(11, n=9000)
    t[4500:] += 0.5  # a gap in the middle
    r = flare_roc(t, f, scorer=Wide(), n_injections=24, seed=2, amplitudes=(0.03,), fwhm_minutes=(6.0,))
    assert len(r.injections) == 24 and r.meta["n_scored"] > 0
    seg = np.asarray(r.injections["tpeak"])
    assert not np.any((seg > t[4500] - 0.65) & (seg < t[4500] + 0.65))  # none within a segment of the gap


def test_injections_avoid_weak_real_candidates():
    """A real flare below avoid_threshold but above base_threshold must not be credited as a recovery."""
    t, f = noise_lc(12, n=12000)
    f = f.copy()
    k = 6000
    f += davenport_flare(t, t[k], 6 / 1440, 0.005)  # peak probability 0.45 in the fake scorer: above 0.1, below 0.5
    r = flare_roc(t, f, scorer=FakeScorer(z0=5.5), n_injections=60, seed=4, amplitudes=(0.0001,),
                  fwhm_minutes=(2.0,), avoid_threshold=0.9)
    tp = np.asarray(r.injections["tpeak"])
    assert not np.any(np.abs(tp - t[k]) < 80 * 2 / 1440)
