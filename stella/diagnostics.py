"""Light-curve diagnostics that flag, but never delete, flare candidates.

``dip_diagnostics`` finds short, deep dips (eclipses, transits, instrumental
drops). stella was trained without eclipsing binaries; on them the network
fires at eclipse ingress and egress, so candidates near a dip must be flagged.

Running medians are taken within contiguous segments: a gap (a step of more
than 5.5 cadences) starts a new segment, so that a rotational phase jump across
a downlink or sector gap is not read as a step in the residuals.
"""
import numpy as np
from scipy.ndimage import median_filter

__all__ = ["dip_diagnostics", "near_times", "local_baseline", "candidate_significance"]


def _runs(mask):
    """(start, stop) index pairs of runs of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _segments(time, gap_factor=5.5, cadence_days=None):
    """(start, stop) index pairs of the contiguous stretches of ``time``: a
    step of more than ``gap_factor`` cadences starts a new one. ``time=None``
    gives a single segment."""
    if time is None:
        return None
    time = np.asarray(time, float)
    n = len(time)
    if n < 2:
        return [(0, n)]
    dt = np.diff(time)
    cad = cadence_days if cadence_days is not None else np.nanmedian(dt)
    brk = np.where(dt > gap_factor * cad)[0] + 1
    edges = np.concatenate([[0], brk, [n]])
    return list(zip(edges[:-1], edges[1:]))


def _running_median(x, size, segments=None):
    """Running median over index, restricted to each segment (edges: nearest)."""
    x = np.asarray(x, float)
    if segments is None or len(segments) <= 1:
        return median_filter(x, size=size, mode="nearest")
    out = np.empty_like(x)
    for a, b in segments:
        out[a:b] = median_filter(x[a:b], size=size, mode="nearest")
    return out


def dip_diagnostics(time, flux, window_days=0.3, nsig=5.0, min_cadences=3,
                    cadence_days=None, min_dips=2, gap_factor=5.5):
    """Find dips: runs of at least ``min_cadences`` cadences more than ``nsig``
    robust standard deviations below a running median of width ``window_days``.

    The running median is taken over cadence index within contiguous segments
    (a step of more than ``gap_factor`` cadences is a gap), so the window must
    be longer than about twice the longest eclipse but shorter than about a
    quarter of the rotation period; for fast rotators pass a window based on
    the published (TARS) period.

    Returns a dict: n_dips, dip_times (mid-times), dip_depths (fractional),
    sigma (robust scatter), eclipse_like (n_dips >= min_dips), dips_periodic
    (the dip times fit a period; see ``_dips_periodic``, informational only).
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
    trend = _running_median(rel, size, _segments(time, gap_factor, cad))
    resid = rel - trend
    sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    low = resid < -nsig * sigma
    # a dip must not straddle a gap
    runs = [(a, b) for a, b in _runs(low) if b - a >= min_cadences
            and np.all(np.diff(time[a:b]) <= gap_factor * cad)]
    out["sigma"] = float(sigma)
    out["n_dips"] = len(runs)
    out["dip_times"] = np.array([time[(a + b - 1) // 2] for a, b in runs])
    out["dip_depths"] = np.array([float(-resid[a:b].min()) for a, b in runs])
    out["eclipse_like"] = len(runs) >= min_dips
    out["dips_periodic"] = _dips_periodic(out["dip_times"])
    return out


def _dips_periodic(dip_times, tol_phase=0.01, min_period=0.1, max_k=1, min_cycles=4):
    """True when the dip times are periodic with positive degrees of freedom.

    Either (a) at least ``min_cycles`` (4) dips at distinct cycles share one
    phase (repeated transits, or equal eclipses), or (b) at least two dips share
    a primary phase and at least two share a secondary phase, which together
    account for every dip (one stray dip is tolerated from six dips up). Every
    dip is tried as the phase anchor, so one isolated dropout cannot hide a
    real period; trial periods are the spacings of pairs of dips.

    Three dips can never pass: with an epoch and a period to fit they have one
    degree of freedom, and a free period always fits some of them. Random dip
    times pass about 0-6% of the time for 3 to 8 dips with the default
    ``tol_phase`` of 0.01 (the looser 0.03, or accepting three cycles, passed
    most random sets of five). Informational only: ``eclipse_like`` does not
    depend on it, because a missed eclipsing binary costs more than a spurious
    flag.
    """
    t = np.sort(np.asarray(dip_times, float))
    n = len(t)
    if n < 3:
        return False
    trials = set()
    for a in range(n):
        for b in range(a + 1, n):
            for k in range(1, max_k + 1):
                p = (t[b] - t[a]) / k
                if p >= min_period:
                    trials.add(round(p, 5))
    stray_ok = 1 if n >= 6 else 0
    for p in sorted(trials):
        for anchor in t:
            x = (t - anchor) / p
            cyc = np.round(x)
            ph = x - cyc  # signed phase in [-0.5, 0.5]
            prim = np.abs(ph) < tol_phase
            n_prim = len(np.unique(cyc[prim]))
            if n_prim >= min_cycles:
                return True
            rest = np.where(~prim)[0]
            if n_prim >= 2 and len(rest) >= 2:
                for r in rest:
                    d = ph[rest] - ph[r]
                    d -= np.round(d)
                    sec = np.abs(d) < tol_phase
                    if (len(np.unique(cyc[rest][sec])) >= 2
                            and len(rest) - sec.sum() <= stray_ok):
                        return True
    return False


def near_times(t, times, tol_days=0.1):
    """Boolean array: is each t within tol_days of any of ``times``."""
    t = np.atleast_1d(np.asarray(t, float))
    times = np.asarray(times, float)
    if times.size == 0:
        return np.zeros(t.shape, bool)
    return np.min(np.abs(t[:, None] - times[None, :]), axis=1) < tol_days


def local_baseline(flux, window_cadences=181, nsig=3.0, grow_before=10, grow_after=60, n_iter=3,
                   time=None, gap_factor=5.5, cadence_days=None):
    """Running-median baseline of a light curve that flares do not pull up.

    With ``time`` given, the baseline is built separately in each contiguous
    segment (a step of more than ``gap_factor`` cadences is a gap), so that the
    median window never straddles a gap. Without it the array is one segment.

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
    segs = _segments(time, gap_factor, cadence_days)
    if segs is not None and len(segs) > 1:
        out = np.empty_like(flux)
        for a, b in segs:
            out[a:b] = local_baseline(flux[a:b], window_cadences, nsig, grow_before,
                                      grow_after, n_iter)
        return out
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
    ``n_consec`` (the length of the run of consecutive cadences, without a gap,
    at or above ``nsig_run`` sigma that contains the peak cadence, counted both
    before and after it: a flare that peaks between two cadences splits its
    flux across both). NaN / 0 where sigma cannot be estimated (fewer than 20
    cadences in the window).

    The running median is taken over cadence index within contiguous
    segments, as in ``dip_diagnostics``. It needs more than
    ``window_cadences`` cadences.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    tpeaks = np.atleast_1d(np.asarray(tpeaks, float))
    snr = np.full(len(tpeaks), np.nan)
    nrun = np.zeros(len(tpeaks), dtype=int)
    if len(time) < window_cadences or len(tpeaks) == 0:
        return snr, nrun
    cad = cadence_days if cadence_days is not None else np.nanmedian(np.diff(time))
    r = flux - _running_median(flux, window_cadences, _segments(time, cadence_days=cad))
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
        if r[i] < nsig_run * sig:
            continue
        n = 1
        j = i + 1
        while j < len(r) and r[j] >= nsig_run * sig and time[j] - time[j - 1] < 1.5 * cad:
            n += 1
            j += 1
        j = i - 1
        while j >= 0 and r[j] >= nsig_run * sig and time[j + 1] - time[j] < 1.5 * cad:
            n += 1
            j -= 1
        nrun[k] = n
    return snr, nrun
