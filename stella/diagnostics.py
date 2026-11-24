"""Light-curve diagnostics that flag, but never delete, flare candidates.

``dip_diagnostics`` finds short, deep dips (eclipses, transits, instrumental
drops). stella was trained without eclipsing binaries; on them the network
fires at eclipse ingress and egress, so candidates near a dip must be flagged.
"""
import numpy as np
from scipy.ndimage import median_filter

__all__ = ["dip_diagnostics", "near_times"]


def _runs(mask):
    """(start, stop) index pairs of runs of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def dip_diagnostics(time, flux, window_days=0.3, nsig=5.0, min_cadences=3,
                    cadence_days=None, min_dips=2):
    """Find dips: runs of at least ``min_cadences`` cadences more than ``nsig``
    robust standard deviations below a running median of width ``window_days``.

    The running median is taken over cadence index (gaps are ignored), so the
    window must be longer than about twice the longest eclipse but shorter
    than about a quarter of the rotation period; for fast rotators pass a
    window based on the published (TARS) period.

    Returns a dict: n_dips, dip_times (mid-times), dip_depths (fractional),
    sigma (robust scatter), eclipse_like (n_dips >= min_dips), dips_periodic
    (three or more dips that fit one period with at most two phases).
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    out = dict(n_dips=0, dip_times=np.array([]), dip_depths=np.array([]), sigma=np.nan,
               eclipse_like=False, dips_periodic=False)
    if len(time) < 10:
        return out
    cad = cadence_days if cadence_days is not None else np.nanmedian(np.diff(time))
    size = max(int(round(window_days / cad)) | 1, 3)
    med = np.nanmedian(flux)
    rel = flux / med
    trend = median_filter(rel, size=size, mode="nearest")
    resid = rel - trend
    sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    low = resid < -nsig * sigma
    runs = [(a, b) for a, b in _runs(low) if b - a >= min_cadences]
    out["sigma"] = float(sigma)
    out["n_dips"] = len(runs)
    out["dip_times"] = np.array([time[(a + b - 1) // 2] for a, b in runs])
    out["dip_depths"] = np.array([float(-resid[a:b].min()) for a, b in runs])
    out["eclipse_like"] = len(runs) >= min_dips
    out["dips_periodic"] = _dips_periodic(out["dip_times"])
    return out


def _dips_periodic(dip_times, tol_phase=0.03, min_period=0.1):
    """True when at least three dips fall in at most two phase bins of one
    trial period (primary and secondary eclipse, or repeated transits).

    Informational only: ``eclipse_like`` does not depend on it, because a
    missed eclipsing binary costs more than a spurious flag.
    """
    t = np.sort(np.asarray(dip_times, float))
    if len(t) < 3:
        return False
    trials = set()
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            for k in (1, 2, 3, 4):
                p = (t[b] - t[a]) / k
                if p >= min_period:
                    trials.add(round(p, 5))
    for p in trials:
        ph = ((t - t[0]) / p) % 1.0
        ph = np.where(ph > 0.5, ph - 1.0, ph)  # primary near 0
        prim = np.abs(ph) < tol_phase
        rest = ph[~prim]
        if len(rest) == 0:
            return True
        ph2 = ((rest - rest[0] + 0.5) % 1.0) - 0.5
        if np.all(np.abs(ph2) < tol_phase):
            return True
    return False


def near_times(t, times, tol_days=0.1):
    """Boolean array: is each t within tol_days of any of ``times``."""
    t = np.atleast_1d(np.asarray(t, float))
    times = np.asarray(times, float)
    if times.size == 0:
        return np.zeros(t.shape, bool)
    return np.min(np.abs(t[:, None] - times[None, :]), axis=1) < tol_days


def local_baseline(flux, window_cadences=181, nsig=3.0, grow_before=10, grow_after=60, n_iter=3):
    """Running-median baseline of a light curve that flares do not pull up.

    A first running median (``window_cadences``, default 181 = 6 h at 2 min) is
    subtracted. Cadences more than ``nsig`` robust sigma *above* it are masked,
    the mask is widened by ``grow_before`` cadences before and ``grow_after``
    after (flares rise fast and decay slowly), the masked cadences are replaced
    by a straight line between the unmasked ones either side, and the median is
    taken again, ``n_iter`` times in all (each pass re-detects against the
    cleaner baseline). A flare that fills more than about half of the window
    still lifts the baseline.
    """
    flux = np.asarray(flux, float)
    idx = np.arange(len(flux))
    trend = median_filter(flux, size=window_cadences, mode="nearest")
    for _ in range(max(n_iter - 1, 0)):
        r = flux - trend
        sig = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        if not np.isfinite(sig) or sig == 0:
            break
        hot = r > nsig * sig
        # widen: before by grow_before, after by grow_after
        mask = hot.copy()
        for s in range(1, int(grow_before) + 1):
            mask[:-s] |= hot[s:]
        for s in range(1, int(grow_after) + 1):
            mask[s:] |= hot[:-s]
        if mask.all() or not mask.any():
            break
        work = np.where(mask, np.interp(idx, idx[~mask], flux[~mask]), flux)
        trend = median_filter(work, size=window_cadences, mode="nearest")
    return trend


def candidate_significance(time, flux, tpeaks, window_cadences=181, half_width_days=0.5,
                           exclude_days=1.0 / 48, nsig_run=3.0, cadence_days=None):
    """Peak significance of each candidate against a running median.

    For each ``tpeak``: sigma is the robust (MAD) scatter of the residuals
    from a ``window_cadences`` running median in the cadences within
    ``half_width_days`` of the peak, excluding ``exclude_days`` (30 min)
    either side of it, so that the flare does not inflate its own sigma.

    Returns ``snr_peak`` (residual at the cadence nearest the peak / sigma) and
    ``n_consec`` (the number of consecutive cadences, starting at the peak and
    going forward without a gap, at or above ``nsig_run`` sigma). NaN / 0 where
    sigma cannot be estimated (fewer than 20 cadences in the window).

    The running median is taken over cadence index, as in
    ``dip_diagnostics``. It needs more than ``window_cadences`` cadences.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    tpeaks = np.atleast_1d(np.asarray(tpeaks, float))
    snr = np.full(len(tpeaks), np.nan)
    nrun = np.zeros(len(tpeaks), dtype=int)
    if len(time) < window_cadences or len(tpeaks) == 0:
        return snr, nrun
    cad = cadence_days if cadence_days is not None else np.nanmedian(np.diff(time))
    r = flux - median_filter(flux, size=window_cadences, mode="nearest")
    for k, tp in enumerate(tpeaks):
        i = int(np.argmin(np.abs(time - tp)))
        dt = np.abs(time - tp)
        w = (dt < half_width_days) & (dt > exclude_days)
        if w.sum() <= 20:
            continue
        sig = 1.4826 * np.median(np.abs(r[w] - np.median(r[w])))
        if not np.isfinite(sig) or sig <= 0:
            continue
        snr[k] = r[i] / sig
        j, n = i, 0
        while j < len(r) and r[j] >= nsig_run * sig and (j == i or time[j] - time[j - 1] < 1.5 * cad):
            n += 1
            j += 1
        nrun[k] = n
    return snr, nrun
