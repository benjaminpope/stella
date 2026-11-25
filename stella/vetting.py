"""Per-star ROC vetting of flare candidates.

A fixed acceptance cut (a probability, or a significance) has a very different
false-positive rate on a quiet M dwarf than on a fast rotator. ``flare_roc``
measures, for ONE star's light curve, how a candidate-acceptance statistic
trades false positives against recovered flares *on that star*, and
``FlareROC.threshold_for_fp_rate`` / ``threshold_for_fn_fraction`` return the
cut for a stated false-positive rate per day or a stated missed fraction.

How the curve is built
----------------------
* **False positives** come from the flux-inverted light curve of the same star:
  ``2 * baseline - flux``, with the baseline a running median that flares are
  masked out of (``stella.diagnostics.local_baseline``). Real flares become
  dips, so every candidate accepted on the inverted curve is false. They are
  reported as a rate per day of *scored* time (cadences for which the network
  produced a probability).
* **True positives** come from artificial flares (Davenport et al. 2014
  template, by default) injected into the real light curve on a grid of
  amplitudes and FWHMs, one at a time, at random places that are scored by the
  network and have no real candidate nearby. A flare is recovered when a
  candidate is accepted by ``FitFlares`` within ``match_days`` of its peak.
* The three selectable statistics are columns of the candidate table, so the
  cut can be changed later without rerunning the network:

  ``"prob"``
      the highest ensemble probability in the candidate's group
      (``prob_max``);
  ``"snr"``
      the peak significance in sigma above a 6 h running median (``snr_peak``);
  ``"combined"``
      the rule of the SRSC review: at least ``min_consecutive`` (2)
      consecutive cadences at or above 3 sigma, and then ``snr_peak`` at or
      above the threshold. Candidates failing the run condition have
      statistic -inf and are never accepted.

Limits (read before using a threshold)
--------------------------------------
* Inversion only measures false positives from sources that are symmetric
  about the baseline: white and red noise, and variability on timescales
  shorter than about half the baseline window (3 h for the default 6 h
  window), and many systematics. Slower variability, including most
  rotational modulation, is part of the running median, stays as it is, and is
  not inverted; neither are positive-only or asymmetric false positives:
  eclipse egress, asteroid crossings, scattered-light ramps, cosmic rays,
  contaminating flares from neighbouring stars. The false-positive rate is a
  lower bound. ``inversion_diagnostic`` plots flux, baseline and inverted flux
  together, to check them across rotational extrema and gaps.
* The baseline is built separately in each contiguous segment (a step of more
  than ``max_gap_cadences`` cadences is a gap). The flux is normalised by one
  median over the whole input: for several sectors, normalise each first.
* The baseline is a running median (default 6 h). Flares are masked out of it,
  but a flare lasting more than about half the window, or a star that flares
  almost continuously, still pulls the baseline up; the inverted curve is then
  not a clean null. Check ``roc.meta['frac_above_baseline']`` (the share of cadences more
  than 3 sigma above the baseline; a large share means frequent flaring).
* Statistics ``"snr"`` and ``"combined"`` are conditional on the network
  having produced a candidate at ``base_threshold``; ``base_threshold`` is the
  floor of every ROC.
* A star with little observing time gives a coarse curve: the false-positive
  rate is a Poisson count divided by the exposure. Intervals (default 68%) are
  exact Poisson (Garwood) for the rate and Clopper-Pearson for recoveries.
  With zero false candidates the upper limit is about 1.8 / exposure.
* Recovery depends on the injected template and the amplitude/FWHM grid; it is
  not the recovery of real flares, whose shapes are more varied.
"""
import functools
import warnings

import numpy as np
from astropy.table import Table
from scipy.ndimage import maximum_filter1d
from scipy.stats import beta, chi2

from .diagnostics import local_baseline
from .mark_flares import FitFlares

__all__ = [
    "davenport_flare",
    "EnsembleScorer",
    "FlareROC",
    "flare_roc",
    "inversion_diagnostic",
    "candidate_statistic",
    "threshold_for_fp_rate",
    "threshold_for_fn_fraction",
    "apply_threshold",
    "poisson_interval",
    "binomial_interval",
    "STATISTICS",
]

STATISTICS = ("prob", "snr", "combined")
DEFAULT_AMPLITUDES = (0.003, 0.01, 0.03, 0.1)  # relative to the baseline flux
DEFAULT_FWHM_MINUTES = (2.0, 6.0, 20.0)
_HALF = 100  # the network sees flux[i-100 : i+100]


# ---------------------------------------------------------------- templates
def davenport_flare(time, tpeak, fwhm_days, amplitude, exposure_days=2.0 / 1440, supersample=5):
    """Davenport et al. (2014) flare template (relative flux, 0 outside the flare).

    ``fwhm_days`` is the full width at half maximum, ``amplitude`` the
    fractional peak. The model is averaged over each exposure
    (``supersample`` points), which matters for flares as short as the
    cadence.
    """
    time = np.asarray(time, float)
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    x = ((time[:, None] + off[None, :] * exposure_days) - tpeak) / fwhm_days
    y = np.zeros_like(x)
    r = (x > -1) & (x <= 0)
    xr = x[r]
    y[r] = 1 + 1.941 * xr - 0.175 * xr**2 - 2.246 * xr**3 - 1.125 * xr**4
    d = x > 0
    xd = x[d]
    y[d] = 0.6890 * np.exp(-1.600 * xd) + 0.3030 * np.exp(-0.2783 * xd)
    return amplitude * y.mean(axis=1)


# ---------------------------------------------------------------- intervals
def poisson_interval(n, exposure, ci=0.68):
    """Exact (Garwood) interval of a Poisson rate: n counts in ``exposure``."""
    n = np.asarray(n, float)
    a = 1.0 - ci
    lo = np.where(n > 0, chi2.ppf(a / 2, 2 * np.maximum(n, 1)) / 2, 0.0)
    hi = chi2.ppf(1 - a / 2, 2 * (n + 1)) / 2
    return lo / exposure, hi / exposure


def binomial_interval(k, n, ci=0.68):
    """Clopper-Pearson interval of a fraction k / n (NaN when n == 0)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    a = 1.0 - ci
    with np.errstate(invalid="ignore"):
        lo = np.where(k > 0, beta.ppf(a / 2, k, np.maximum(n - k + 1, 1e-9)), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a / 2, k + 1, np.maximum(n - k, 1e-9)), 1.0)
    bad = n <= 0
    return np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)


# ---------------------------------------------------------------- statistics
def candidate_statistic(table, statistic, min_consecutive=2):
    """Per-candidate value of ``statistic`` (see module docstring); -inf where
    the candidate can never be accepted (undefined significance, or fewer than
    ``min_consecutive`` cadences at 3 sigma for ``"combined"``)."""
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    if len(table) == 0:
        return np.zeros(0)
    if statistic == "prob":
        v = np.asarray(table["prob_max"], float)
    else:
        v = np.asarray(table["snr_peak"], float).copy()
        if statistic == "combined":
            v[np.asarray(table["n_consec_3sigma"]) < min_consecutive] = -np.inf
    return np.where(np.isnan(v), -np.inf, v)


def apply_threshold(table, statistic, threshold, min_consecutive=2, column="vetted", drop=False):
    """Apply a threshold to a candidate table (``FitFlares.flare_table``).

    By default nothing is deleted: a boolean column ``column`` is added to a
    copy of the table. With ``drop=True`` only the accepted rows are returned.
    The table must have been produced at or below the ``base_threshold`` the
    threshold was derived with (``roc.meta['base_threshold']``).
    """
    out = table.copy()
    keep = candidate_statistic(out, statistic, min_consecutive) >= threshold
    out[column] = keep
    out.meta["vetting"] = dict(statistic=statistic, threshold=float(threshold), min_consecutive=min_consecutive)
    return out[keep] if drop else out


# ---------------------------------------------------------------- scoring
class EnsembleScorer(object):
    """Scores windows of 200 cadences with every model, loading each once.

    ``score(X)`` takes windows of shape (n, 200) (normalised flux) and returns
    the per-model probabilities, shape (n_models, n). Any object with this
    method and a ``cadences`` attribute can replace it (for tests).
    """

    def __init__(self, models=None, batch=512):
        from . import models as _models

        self.model_paths = list(_models.models if models is None else models)
        self.batch = int(batch)
        self._loaded = None
        self.cadences = 2 * _HALF

    def _load(self):
        if self._loaded is None:
            from .backends import require_backend

            require_backend()
            import keras

            self._loaded = [keras.models.load_model(p, compile=False) for p in self.model_paths]
            self.cadences = int(self._loaded[0].input_shape[1])
        return self._loaded

    def score(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(self._load()), len(X)))
        b = self.batch
        for m, model in enumerate(self._loaded):
            for i0 in range(0, len(X), b):
                chunk = X[i0:i0 + b]
                if len(chunk) < b:  # fixed shape: no recompilation
                    chunk = np.concatenate([chunk, np.zeros((b - len(chunk), X.shape[1]), np.float32)])
                p = np.asarray(model.predict_on_batch(chunk[..., None]))
                out[m, i0:i0 + b] = p.reshape(-1)[: min(b, len(X) - i0)]
        return out


def _aggregate(P, how):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "median":
            return np.nanmedian(P, axis=0)
        if how == "mean":
            return np.nanmean(P, axis=0)
    raise ValueError("aggregate must be 'median' or 'mean'")


def _window_matrix(flux, starts, cadences):
    return np.lib.stride_tricks.sliding_window_view(flux, cadences)[starts]


def _scored_mask(time, n, cadences, max_gap_cadences, cadence_days):
    from .neural_network import make_windows

    _, scored = make_windows(time, np.zeros(n), cadences, max_gap_cadences=max_gap_cadences,
                             cadence_days=cadence_days)
    return scored


def _score_series(scorer, time, flux, scored, cadences):
    """Per-model probabilities (n_models, N), NaN where not scored."""
    half = cadences // 2
    ks = np.where(scored)[0]
    X = _window_matrix(flux, ks - half, cadences)
    S = scorer.score(X)
    P = np.full((S.shape[0], len(flux)), np.nan)
    P[:, ks] = S
    return P


# ---------------------------------------------------------------- the ROC
class FlareROC(object):
    """Result of ``flare_roc``: tables of false and injected candidates and the
    methods that turn them into curves and thresholds.

    Attributes
    ----------
    false_table : astropy Table
        Candidates accepted by ``FitFlares`` on the inverted curve (all false),
        with the vetting columns.
    injections : astropy Table
        One row per injected flare: ``amp``, ``fwhm_min``, ``tpeak``,
        ``recovered`` and the best statistic of its matched candidates
        (``stat_prob``, ``stat_snr``, ``stat_combined``; -inf if none).
    exposure_days : float
        Scored time of the light curve.
    meta : dict
        Settings (``base_threshold``, ``aggregate``, ``seed``, ``ci``, ...).
    """

    def __init__(self, false_table, injections, exposure_days, amplitudes, meta):
        self.false_table = false_table
        self.injections = injections
        self.exposure_days = float(exposure_days)
        self.amplitudes = np.asarray(amplitudes, float)
        self.meta = meta

    # -- helpers
    def _false_stat(self, statistic):
        return candidate_statistic(self.false_table, statistic, self.meta["min_consecutive"])

    def _inj_stat(self, statistic, amplitude=None):
        v = np.asarray(self.injections[f"stat_{statistic}"], float)
        if amplitude is None:
            return v
        return v[self._amp_mask(amplitude)]

    def _amp_mask(self, amplitude):
        a = np.asarray(self.injections["amp"], float)
        i = int(np.argmin(np.abs(self.amplitudes - amplitude)))
        if not np.isclose(self.amplitudes[i], amplitude, rtol=1e-6):
            raise ValueError(f"amplitude {amplitude} was not injected; grid is {list(self.amplitudes)}")
        return np.isclose(a, self.amplitudes[i], rtol=1e-6)

    def floor(self, statistic):
        """Lowest meaningful threshold of a statistic."""
        return self.meta["base_threshold"] if statistic == "prob" else 0.0

    def fp_rate(self, statistic, threshold):
        """(rate, lo, hi) of false positives per day at a threshold, and the count."""
        n = int(np.sum(self._false_stat(statistic) >= threshold))
        lo, hi = poisson_interval(n, self.exposure_days, self.meta["ci"])
        return n / self.exposure_days, float(lo), float(hi), n

    def recovery(self, statistic, threshold, amplitude=None):
        """(fraction, lo, hi, n_injected) recovered at a threshold, overall or for one amplitude."""
        s = self._inj_stat(statistic, amplitude)
        k, n = int(np.sum(s >= threshold)), len(s)
        lo, hi = binomial_interval(k, n, self.meta["ci"])
        return (k / n if n else np.nan), float(lo), float(hi), n

    def operating_point(self, statistic, threshold):
        """dict with the false-positive rate per day and the recovery (overall and
        per amplitude), with intervals, at a threshold."""
        r, lo, hi, n = self.fp_rate(statistic, threshold)
        out = dict(statistic=statistic, threshold=float(threshold), n_false=n, fp_rate=r, fp_lo=lo, fp_hi=hi)
        out["recovery"], out["recovery_lo"], out["recovery_hi"], _ = self.recovery(statistic, threshold)
        out["recovery_by_amp"] = {float(a): self.recovery(statistic, threshold, a)[:3] for a in self.amplitudes}
        return out

    # -- curves
    def curve(self, statistic, thresholds=None):
        """The ROC as a Table over ``thresholds`` (default: every statistic value
        seen, plus one point above the largest, so that the curve is exact).

        Columns: threshold, n_false, fp_rate (per day), fp_lo, fp_hi, recovery,
        recovery_lo, recovery_hi, and ``recovery@<amp>`` with ``_lo`` / ``_hi``
        for each injected amplitude.
        """
        fs, js = self._false_stat(statistic), self._inj_stat(statistic)
        if thresholds is None:
            vals = np.concatenate([fs[np.isfinite(fs)], js[np.isfinite(js)], [self.floor(statistic)]])
            vals = np.unique(vals)
            vals = vals[vals >= self.floor(statistic)]
            thresholds = np.append(vals, vals[-1] + max(1e-6, 1e-6 * abs(vals[-1])))
        thr = np.asarray(thresholds, float)
        n_false = np.array([(fs >= t).sum() for t in thr])
        lo, hi = poisson_interval(n_false, self.exposure_days, self.meta["ci"])
        t = Table()
        t["threshold"] = thr
        t["n_false"] = n_false
        t["fp_rate"] = n_false / self.exposure_days
        t["fp_lo"], t["fp_hi"] = lo, hi

        def add(prefix, s):
            k = np.array([(s >= x).sum() for x in thr])
            rlo, rhi = binomial_interval(k, len(s), self.meta["ci"])
            t[prefix] = k / len(s) if len(s) else np.nan
            t[prefix + "_lo"], t[prefix + "_hi"] = rlo, rhi

        add("recovery", js)
        for a in self.amplitudes:
            add(f"recovery@{a:g}", self._inj_stat(statistic, a))
        return t

    # -- thresholds
    def threshold_for_fp_rate(self, rate, statistic="combined", bound="estimate"):
        """Lowest threshold whose false-positive rate is at most ``rate`` per day.

        ``bound="estimate"`` compares the observed count over the exposure with
        ``rate``; ``bound="upper"`` demands that the upper end of the interval
        is below ``rate`` (conservative; often impossible for a star with
        little observing time, in which case NaN is returned with a warning).
        """
        fs = np.sort(self._false_stat(statistic)[np.isfinite(self._false_stat(statistic))])[::-1]
        if bound == "estimate":
            n_max = int(np.floor(rate * self.exposure_days + 1e-9))
        elif bound == "upper":
            n_max = -1
            for n in range(len(fs) + 1):
                if poisson_interval(n, self.exposure_days, self.meta["ci"])[1] <= rate:
                    n_max = n
                else:
                    break
            if n_max < 0:
                warnings.warn(
                    f"rate {rate}/day is below the upper limit even for zero false candidates "
                    f"({poisson_interval(0, self.exposure_days, self.meta['ci'])[1]:.3g}/day over "
                    f"{self.exposure_days:.1f} d); no threshold qualifies"
                )
                return np.nan
        else:
            raise ValueError("bound must be 'estimate' or 'upper'")
        if len(fs) <= n_max:
            return self.floor(statistic)
        s = fs[n_max]  # largest false value that must be excluded
        above = fs[fs > s]
        step = max(1e-6, 1e-6 * abs(s))
        thr = s + step if len(above) == 0 else min(s + step, 0.5 * (s + above.min()))
        return float(max(thr, self.floor(statistic)))

    def threshold_for_fn_fraction(self, fn_fraction, amplitude, statistic="combined"):
        """Highest threshold that still recovers at least ``1 - fn_fraction`` of the
        flares injected at ``amplitude`` (one of the injected amplitudes).
        NaN with a warning when no threshold can: the statistic does not
        recover that many of them at all."""
        s = np.sort(self._inj_stat(statistic, amplitude))[::-1]
        need = int(np.ceil((1.0 - fn_fraction) * len(s) - 1e-9))
        if need <= 0:
            return float(s[0]) if np.isfinite(s[0]) else np.nan
        if need > len(s) or not np.isfinite(s[need - 1]):
            warnings.warn(f"fewer than {1 - fn_fraction:.0%} of flares injected at amplitude {amplitude:g} "
                          f"are ever recovered (statistic {statistic!r})")
            return np.nan
        return float(s[need - 1])

    def apply(self, table, statistic, threshold, **kw):
        """``apply_threshold`` with this ROC's ``min_consecutive``."""
        kw.setdefault("min_consecutive", self.meta["min_consecutive"])
        return apply_threshold(table, statistic, threshold, **kw)

    def plot(self, statistic="combined", ax=None, by_amplitude=True):
        """Recovery against false positives per day (matplotlib), with intervals."""
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        c = self.curve(statistic)
        floor = 0.5 / self.exposure_days  # where a zero-count rate is drawn
        # curve() runs from the lowest threshold (most false positives) up; plot with the
        # false-positive rate increasing so that a step holds the recovery already
        # achieved at a lower rate (an operating point you can actually choose)
        fp = np.asarray(c["fp_rate"])
        names = ["recovery"] + ([f"recovery@{a:g}" for a in self.amplitudes] if by_amplitude else [])
        for nm in names:
            # ties in the false-positive rate (zero counts): keep the best recovery last
            order = np.lexsort((np.asarray(c[nm]), fp))
            x, y = np.maximum(fp[order], floor), np.asarray(c[nm])[order]
            (line,) = ax.step(x, y, where="post", label="all" if nm == "recovery" else nm.split("@")[1],
                              lw=2 if nm == "recovery" else 1)
            if nm == "recovery":  # interval on the overall recovery
                ax.fill_between(x, np.asarray(c[nm + "_lo"])[order], np.asarray(c[nm + "_hi"])[order],
                                step="post", color=line.get_color(), alpha=0.15, lw=0)
        ax.axvline(floor, color="0.5", ls=":", lw=1)
        ax.annotate("0 false\ncandidates", (floor, 0.02), fontsize=7, color="0.4", ha="left", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("false positives per day (inverted light curve)")
        ax.set_ylabel("recovery of injected flares")
        ax.set_title(f"statistic: {statistic}")
        ax.legend(title="amplitude", fontsize=8)
        return ax


def threshold_for_fp_rate(roc, rate, statistic="combined", bound="estimate"):
    """See ``FlareROC.threshold_for_fp_rate``."""
    return roc.threshold_for_fp_rate(rate, statistic, bound)


def threshold_for_fn_fraction(roc, fn_fraction, amplitude, statistic="combined"):
    """See ``FlareROC.threshold_for_fn_fraction``."""
    return roc.threshold_for_fn_fraction(fn_fraction, amplitude, statistic)


# ---------------------------------------------------------------- the pipeline
def _clean(time, flux, err):
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    if err is None:
        d = np.diff(flux[np.isfinite(flux)])
        err = np.full(len(flux), 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    err = np.asarray(err, float)
    q = np.isfinite(time) & np.isfinite(flux) & np.isfinite(err)
    time, flux, err = time[q], flux[q], err[q]
    med = np.nanmedian(flux)
    if not np.isfinite(med) or med == 0:
        med = 1.0
    return time, flux / med, err / med


def inversion_diagnostic(time, flux, err=None, baseline_window=181, max_gap_cadences=5.5,
                         cadence_days=None, ax=None):
    """The flux, its baseline and the inverted flux ``2 * baseline - flux`` that
    ``flare_roc`` scores for false positives, so that they can be checked by eye.

    Returns ``(time, flux, base, flux_inv)`` (flux normalised by its median); with
    ``ax`` (or ``ax=True`` for the current axes) also plots them. Look at them
    across rotational extrema (the baseline cannot follow variability faster
    than about half the window, so it is left uninverted) and gaps.
    """
    time, flux, _ = _clean(time, flux, err)
    cad = cadence_days if cadence_days is not None else float(np.nanmedian(np.diff(time)))
    base = local_baseline(flux, window_cadences=baseline_window, time=time,
                          gap_factor=max_gap_cadences, cadence_days=cad)
    flux_inv = 2 * base - flux
    flux_inv = flux_inv / np.nanmedian(flux_inv)
    if ax is not None and ax is not False:
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is True else ax
        ax.plot(time, flux, ".", ms=2, color="0.5", label="flux")
        ax.plot(time, base, color="C1", lw=1.5, label="baseline")
        ax.plot(time, flux_inv, ".", ms=2, color="C0", alpha=0.6, label="inverted")
        ax.set_xlabel("time")
        ax.set_ylabel("relative flux")
        ax.legend(fontsize=8)
    return time, flux, base, flux_inv


def _candidates(fit_args, threshold, cadence_days):
    t, f, e, p = fit_args
    ff = FitFlares(id=np.asarray([0]), time=[t], flux=[f], flux_err=[e], predictions=[p],
                   cadence_days=cadence_days)
    ff.identify_flare_peaks(threshold=threshold)
    return ff.flare_table


def flare_roc(time, flux, err=None, scorer=None, aggregate="median", base_threshold=0.1,
              amplitudes=DEFAULT_AMPLITUDES, fwhm_minutes=DEFAULT_FWHM_MINUTES, n_injections=240,
              seed=0, template=davenport_flare, match_days=0.01, min_consecutive=2, ci=0.68,
              avoid_threshold=0.5, baseline_window=181, max_gap_cadences=5.5, cadence_days=None, chunk=60):
    """Build the per-star ROC (see the module docstring for the method and limits).

    Parameters
    ----------
    time, flux, err : arrays
        One star's light curve (one sector or several; gaps are handled). The
        flux is normalised by its median, as ``ConvNN.predict`` does.
    scorer : object, optional
        ``EnsembleScorer`` (default: the packaged ensemble); anything with
        ``score(X) -> (n_models, n)`` and ``cadences``.
    aggregate : {"median", "mean"} or sequence of them
        How the ensemble is combined. A sequence returns a dict of ``FlareROC``
        that share the network passes (one cost for both).
    base_threshold : float
        Probability at which candidates are generated (the floor of every
        statistic). Make the star's real candidate table at the same value.
    amplitudes, fwhm_minutes : sequences
        Injection grid: fractional peak amplitudes and FWHMs in minutes.
        Defaults 0.3, 1, 3, 10 percent and 2, 6, 20 min.
    n_injections : int
        Total injected flares, rounded up to a multiple of the number of grid
        cells. Each is injected and scored alone. Cost: each injection scores
        about 300 windows (the windows that contain the flare), so 240 cost
        about as much as 3 scored light curves of 18000 cadences, plus one
        scored light curve each for the real and inverted data.
    seed : int
        Seed of the injection sites and times; the result is deterministic.
    template : callable
        ``template(time, tpeak, fwhm_days, amplitude) -> relative flux``.
    match_days : float
        A candidate within this time of the injected peak recovers it.
    min_consecutive : int
        Run length at 3 sigma required by the ``"combined"`` statistic.
    ci : float
        Confidence level of the intervals (default 0.68).
    avoid_threshold : float
        Injections are placed at least 80 cadences from any cadence where the
        real light curve already has an ensemble probability at or above this
        value, so real flares are not credited as recoveries.
    chunk : int
        Injections scored per network call (bounds memory).

    Returns
    -------
    ``FlareROC``, or ``{aggregate: FlareROC}`` when ``aggregate`` is a sequence.
    """
    multi = not isinstance(aggregate, str)
    aggs = list(aggregate) if multi else [aggregate]
    scorer = scorer if scorer is not None else EnsembleScorer()
    cad = int(getattr(scorer, "cadences", 2 * _HALF))
    half = cad // 2

    time, flux, err = _clean(time, flux, err)
    n = len(time)
    cadence_days = cadence_days if cadence_days is not None else float(np.nanmedian(np.diff(time)))
    rng = np.random.default_rng(seed)

    scored = _scored_mask(time, n, cad, max_gap_cadences, cadence_days)
    if scored.sum() == 0:
        raise ValueError("no cadence has a complete window inside a contiguous segment")
    exposure = scored.sum() * cadence_days

    # real light curve: only needed to keep injections away from real candidates
    P_real = _score_series(scorer, time, flux, scored, cad)

    # inverted light curve about the baseline (flares masked out of the baseline)
    base = local_baseline(flux, window_cadences=baseline_window, time=time,
                          gap_factor=max_gap_cadences, cadence_days=cadence_days)
    flux_inv = 2 * base - flux
    flux_inv = flux_inv / np.nanmedian(flux_inv)
    P_inv = _score_series(scorer, time, flux_inv, scored, cad)

    # injection sites: all windows that contain the flare and its tail are scored,
    # and the real data has no candidate nearby. The segment around an injection
    # must hold the windows (half + tail_max either side), the +-0.5 d used for the
    # significance and the baseline, and lie in one contiguous stretch.
    tail_max = 250
    seg_half = max(int(np.ceil(0.5 / cadence_days)) + baseline_window // 2, half + tail_max, cad)
    seg_id = np.concatenate([[0], np.cumsum(np.diff(time) > max_gap_cadences * cadence_days)])
    ok = np.zeros(n, bool)
    lo_k, hi_k = max(seg_half, half), n - seg_half - 1
    cs = np.concatenate([[0], np.cumsum(scored)])
    for k in range(lo_k, hi_k + 1):
        a, b = k - half, k + half + tail_max + 1  # windows scored for this injection
        ok[k] = (b <= n and cs[b] - cs[a] == b - a
                 and seg_id[k - seg_half] == seg_id[k + seg_half])
    amp_cells = [(a, w) for a in amplitudes for w in fwhm_minutes]
    n_per = int(np.ceil(n_injections / len(amp_cells)))
    plan_cells = [c for c in amp_cells for _ in range(n_per)]

    # keep injections 80 cadences away from any real flare candidate, for every aggregate asked for
    # (cadences with probability >= avoid_threshold, and every FitFlares candidate made
    # at base_threshold, so a weak real candidate is never credited as a recovery)
    bad = np.zeros(n, bool)
    for how in aggs:
        pr = np.nan_to_num(_aggregate(P_real, how), nan=0.0)
        bad |= maximum_filter1d(pr, size=161) >= avoid_threshold
        real_tab = _candidates((time, flux, err, pr), base_threshold, cadence_days)
        if len(real_tab):
            pk = np.searchsorted(time, np.asarray(real_tab["tpeak"]))
            near = np.zeros(n, bool)
            near[np.clip(pk, 0, n - 1)] = True
            bad |= maximum_filter1d(near.astype(np.uint8), size=161).astype(bool)
    valid = np.where(ok & ~bad)[0]
    if len(valid) == 0:
        raise ValueError("no usable injection site: every scored stretch has a candidate nearby "
                         "or is too short; raise avoid_threshold or pass a longer light curve")
    sites_k = rng.choice(valid, size=len(plan_cells))
    jitter = rng.uniform(-0.5, 0.5, size=len(plan_cells)) * cadence_days
    sigma = 1.4826 * np.nanmedian(np.abs(np.diff(flux))) / np.sqrt(2)
    if template is davenport_flare:  # average over the actual exposure, not 2 min
        template = functools.partial(davenport_flare, exposure_days=cadence_days)

    # score the injections, a chunk of them per network call
    inj = []  # (cell, k, tpk, seg_flux, P_seg)
    for c0 in range(0, len(plan_cells), chunk):
        Xs, meta_c = [], []
        for j in range(c0, min(c0 + chunk, len(plan_cells))):
            a, w = plan_cells[j]
            k, tpk = int(sites_k[j]), time[sites_k[j]] + jitter[j]
            sl = slice(k - seg_half, k + seg_half + 1)
            seg_t = time[sl]
            add = template(seg_t, tpk, w / 1440.0, a) * base[sl]
            seg_f = flux[sl] + add
            # tail: windows past the peak that still contain flare flux
            above = np.where(np.abs(add) > 0.02 * sigma)[0]
            tail = int(min(max(above.max() - seg_half, 0) if len(above) else 0, tail_max))
            ks = np.arange(k - half, k + half + tail + 1)
            full = flux.copy()
            full[sl] = seg_f
            Xs.append(_window_matrix(full, ks - half, cad))
            meta_c.append((j, k, tpk, seg_f, ks))
        S = scorer.score(np.concatenate(Xs))
        pos = 0
        for (j, k, tpk, seg_f, ks), X in zip(meta_c, Xs):
            P_seg = P_real[:, k - seg_half:k + seg_half + 1].copy()
            P_seg[:, ks - (k - seg_half)] = S[:, pos:pos + len(X)]
            pos += len(X)
            inj.append((j, k, tpk, seg_f, P_seg))

    frac_high = float(np.mean(flux - base > 3 * sigma))  # share of cadences 3 sigma above the baseline
    out = {}
    for how in aggs:
        tab_inv = _candidates((time, flux_inv, err, _aggregate(P_inv, how)), base_threshold, cadence_days)
        rows = []
        for j, k, tpk, seg_f, P_seg in inj:
            sl = slice(k - seg_half, k + seg_half + 1)
            ctab = _candidates((time[sl], seg_f, err[sl], _aggregate(P_seg, how)), base_threshold, cadence_days)
            a, w = plan_cells[j]
            m = np.abs(np.asarray(ctab["tpeak"]) - tpk) <= match_days if len(ctab) else np.zeros(0, bool)
            row = dict(amp=a, fwhm_min=w, tpeak=tpk, recovered=bool(m.any()))
            for st in STATISTICS:
                v = candidate_statistic(ctab[m], st, min_consecutive) if m.any() else np.array([-np.inf])
                row[f"stat_{st}"] = float(np.max(v))
            rows.append(row)
        injections = Table(rows=rows) if rows else Table()
        meta = dict(aggregate=how, base_threshold=base_threshold, seed=seed, n_injections=len(rows),
                    ci=ci, min_consecutive=min_consecutive, match_days=match_days,
                    fwhm_minutes=tuple(fwhm_minutes), cadence_days=cadence_days,
                    n_scored=int(scored.sum()), frac_above_baseline=frac_high,
                    n_models=int(P_inv.shape[0]))
        out[how] = FlareROC(tab_inv, injections, exposure, amplitudes, meta)
    return out if multi else out[aggs[0]]
