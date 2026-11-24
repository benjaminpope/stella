import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.signal import medfilt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .utils import flare_lightcurve
from .diagnostics import candidate_significance

__all__ = ["FitFlares"]

# np.trapz was removed in NumPy 2.x (deprecated in 2.0); np.trapezoid exists from 2.0.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class FitFlares(object):
    """
    Uses the predictions from the neural network and identifies flaring
    events based on consecutive points. Users define a given probability
    threshold for accepting a flare event as a candidate.

    The detection and fitting logic is that of upstream stella
    (Feinstein et al. 2020), with these changes (see docs/review_stella.md in
    benjaminpope/srsc):

    * one fitting routine instead of two copies of the loop;
    * ``np.trapezoid`` instead of ``np.trapz`` (removed in NumPy 2.x);
    * the fit window is clipped at the ends of the series and the peak index
      is the true peak, not the middle of a truncated window;
    * candidates whose fit window cannot be built are skipped and counted in
      ``self.n_skipped`` instead of raising;
    * duplicate peaks (two probability groups resolving to the same cadence)
      are reported once;
    * extra diagnostic columns: ``n_above`` (cadences >= threshold in the
      group), ``dist_gap`` (cadences to the nearest gap or end of series),
      ``fit_at_bound`` (rise or fall at a fit bound, i.e. the duration and
      equivalent duration are lower limits);
    * vetting statistics, always stored so that a cut can be chosen later
      without rerunning the network: ``prob_max`` (highest probability in the
      group), ``snr_peak`` (peak significance in sigma above a 6 h running
      median) and ``n_consec_3sigma`` (consecutive cadences from the peak at
      or above 3 sigma); see ``stella.vetting`` for how to choose cuts per star.

    Parameters
    ----------
    min_amp : float
        Final cut on the peak of the detrended flux (``amp`` column), which
        upstream applied without documenting it (1.002, i.e. 0.2% above the
        local median). Use ``None`` or 0 to keep every fitted candidate.
    """

    def __init__(self, id, time, flux, flux_err, predictions,
                 cadence_days=2.0 / 1440.0, gap_factor=5.5,
                 rise_bounds=(1e-4, 0.01), fall_bounds=(5e-4, 0.01), min_amp=1.002):
        self.IDs = id
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.predictions = predictions
        self.cadence_days = cadence_days
        self.gap_factor = gap_factor
        self.rise_bounds = rise_bounds
        self.fall_bounds = fall_bounds
        self.min_amp = min_amp

    # ------------------------------------------------------------------ grouping
    def group_inds(self, values):
        """Groups indices above threshold; indices within 3 of the group's
        largest index are one group. Returns a list of integer arrays."""
        results = []
        cur = []
        for v in values:
            if cur and np.abs(v - max(cur)) > 3:
                results.append(np.asarray(cur, dtype=int))
                cur = []
            cur.append(int(v))
        if cur:
            results.append(np.asarray(cur, dtype=int))
        return results

    def get_init_guesses(self, groupings, time, flux, err, prob, maskregion, region):
        """For each group, the time and flux of the highest-flux cadence with
        probability >= threshold within +-region cadences (as upstream)."""
        tpeaks, ampls = [], []
        n = len(time)
        for g in groupings:
            lo = max(int(g[0]) - region, 0)
            hi = min(int(g[-1]) + region, n)
            subp = prob[lo:hi]
            subf = flux[lo:hi]
            doubcheck = np.where(subp >= self.threshold)[0]
            if len(doubcheck) == 0:
                continue
            k = doubcheck[np.argmax(subf[doubcheck])]
            tpeaks.append(time[lo + k])
            ampls.append(flux[lo + k])
        return np.asarray(tpeaks, float), np.asarray(ampls, float)

    # ------------------------------------------------------------------ helpers
    def _gap_edges(self, time):
        brk = np.where(np.diff(time) > self.gap_factor * self.cadence_days)[0]
        return np.unique(np.concatenate([[0], brk, brk + 1, [len(time) - 1]]))

    def _fit_one(self, tid, time, flux, err, prob, tp, amp_guess, edges, group_size, prob_max):
        """Fit one candidate. Returns a row (list) or None."""
        kernel_size, kernel_size1 = 15, 21
        n = len(time)
        if amp_guess > 1.3:
            region, maskregion = 400, 150
        else:
            region, maskregion = 40, 10

        where = int(np.searchsorted(time, tp))
        lo, hi = max(where - region, 0), min(where + region, n)
        subt, subf, sube, subp = time[lo:hi], flux[lo:hi], err[lo:hi], prob[lo:hi]
        amp_ind = where - lo  # true peak index inside the window
        if amp_ind < 2 or amp_ind > len(subt) - 3 or len(subt) <= max(kernel_size1, 10):
            return "edge"

        mask = np.zeros(len(subt))
        mask[max(int(amp_ind - maskregion / 2.0), 0): int(amp_ind + maskregion)] = 1
        m = mask == 0
        if m.sum() < kernel_size:
            return "edge"

        func = interp1d(subt[m], medfilt(subf[m], kernel_size=kernel_size),
                        bounds_error=False, fill_value=np.nan)
        func1 = interp1d(subt, medfilt(subf, kernel_size=kernel_size1))
        detrended = subf / func(subt)
        if not np.all(np.isfinite(detrended[max(amp_ind - 1, 0): amp_ind + 2])):
            return "edge"
        std = np.nanstd(detrended[m])
        med = np.nanmedian(detrended[m])
        detrend_with_flare = subf / func1(subt)
        std1 = np.nanstd(detrend_with_flare)
        med1 = np.nanmedian(detrend_with_flare)

        amp = subf[amp_ind]
        amp1 = detrended[amp_ind]
        if amp > 1.5:
            decay_guess, rise_guess = 0.008, 0.003
        else:
            decay_guess, rise_guess = 0.001, 0.0001

        # Acceptance rule of upstream stella: peak >= 1.5 sigma above the
        # out-of-flare median, the two following and one preceding points not
        # higher than the peak, and the next point >= 1 sigma above the median.
        if not ((amp1 > (med + 1.5 * std)) and (subf[amp_ind + 1] <= amp)
                and (subf[amp_ind + 2] <= amp) and (subf[amp_ind - 1] <= amp)):
            return None
        if not detrended[amp_ind + 1] >= (med1 + std1):
            return None
        if not ((subf[amp_ind - 1] < amp) and ((amp - subf[-1]) < 2)):
            return None

        amp1 -= med
        f0, f1 = max(amp_ind - maskregion, 0), min(amp_ind + maskregion, len(subt))
        ft, fd, fe = subt[f0:f1], detrended[f0:f1], sube[f0:f1]
        ok = np.isfinite(fd) & np.isfinite(fe) & (fe > 0)
        t0_ind = amp_ind - f0

        def chi_square(var):
            a, rise, decay = var
            mdl, _ = flare_lightcurve(ft, t0_ind, a, rise, decay)
            return np.sum((fd[ok] - mdl[ok]) ** 2.0 / fe[ok] ** 2.0)

        x = minimize(chi_square, x0=[amp1, rise_guess, decay_guess],
                     bounds=((amp1 - 0.1, amp1 + 0.1), self.rise_bounds, self.fall_bounds),
                     method="L-BFGS-B")
        fm, params = flare_lightcurve(subt, amp_ind, np.nanmedian([amp1, x.x[0]]), x.x[1], x.x[2])
        dur = _trapezoid(fm - 1, subt) * u.day
        at_bound = bool(np.isclose(x.x[1], self.rise_bounds[1]) or np.isclose(x.x[2], self.fall_bounds[1]))
        dist_gap = int(np.min(np.abs(edges - (lo + amp_ind))))
        return [tid, params[0], detrended[amp_ind], dur.to(u.s).value, params[3], params[4],
                subp[amp_ind], int(group_size), dist_gap, at_bound, float(prob_max)]

    # ------------------------------------------------------------------ main
    def identify_flare_peaks(self, threshold=0.5, verbose=False):
        """
        Finds where the predicted value is above the threshold as a flare
        candidate, groups consecutive indices as one event and fits it.

        Attributes
        ----------
        threshold : float
        flare_table : astropy.table.Table
             Target_ID, tpeak, amp (peak of the detrended flux, 1 + A),
             ed_s (equivalent duration of the fitted model, s), rise, fall
             (days), prob, n_above, dist_gap, fit_at_bound, prob_max, snr_peak,
             n_consec_3sigma.
        n_skipped : int
             Candidates that could not be fitted because they sit at an end
             of the series or a gap.
        """
        self.threshold = threshold
        table = Table(names=["Target_ID", "tpeak", "amp", "ed_s", "rise", "fall", "prob",
                             "n_above", "dist_gap", "fit_at_bound", "prob_max"],
                      dtype=[object, float, float, float, float, float, float, int, int, bool, float])
        self.n_skipped = 0
        lc_index = []  # which light curve each row came from
        it = range(len(self.IDs))
        if verbose and tqdm is not None:
            it = tqdm(it, desc="Finding & Fitting Flares")
        for i in it:
            time = np.asarray(self.time[i], dtype=float)
            flux = np.asarray(self.flux[i], dtype=float)
            err = np.asarray(self.flux_err[i], dtype=float)
            prob = np.nan_to_num(np.asarray(self.predictions[i], dtype=float), nan=0.0)
            if len(time) < 50:
                continue
            edges = self._gap_edges(time)
            groupings = self.group_inds(np.where(prob >= threshold)[0])
            tpeaks, amps = self.get_init_guesses(groupings, time, flux, err, prob, 2, 50)
            sizes = [len(g) for g in groupings]
            gmax = [float(np.max(prob[g])) for g in groupings]
            seen = set()
            for tp, amp, gs, gm in zip(tpeaks, amps, sizes, gmax):
                key = round(float(tp), 7)
                if key in seen:  # two groups resolving to the same peak
                    continue
                seen.add(key)
                row = self._fit_one(self.IDs[i], time, flux, err, prob, tp, amp, edges, gs, gm)
                if row == "edge":
                    self.n_skipped += 1
                elif row is not None:
                    table.add_row(row)
                    lc_index.append(i)
        lc_index = np.asarray(lc_index, dtype=int)
        if self.min_amp:
            keep = np.asarray(table["amp"]) > self.min_amp
            table, lc_index = table[keep], lc_index[keep]
        self.flare_table = self._add_significance(table, lc_index)

    def _add_significance(self, table, lc_index):
        """Adds ``snr_peak`` and ``n_consec_3sigma`` (see ``candidate_significance``)."""
        snr = np.full(len(table), np.nan)
        nrun = np.zeros(len(table), dtype=int)
        for i in range(len(self.IDs)):
            rows = np.where(lc_index == i)[0]
            if len(rows) == 0:
                continue
            s, n = candidate_significance(self.time[i], self.flux[i], np.asarray(table["tpeak"])[rows],
                                          cadence_days=self.cadence_days)
            snr[rows], nrun[rows] = s, n
        table["snr_peak"] = snr
        table["n_consec_3sigma"] = nrun
        return table
