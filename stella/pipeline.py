import os
import warnings
import numpy as np
from typing import Iterable, List, Optional, Sequence, Tuple, Union
try:
    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TimeRemainingColumn,
            MofNCompleteColumn,
            TextColumn,
            track,
        )
        HAVE_RICH = True
    except Exception:  # pragma: no cover
        HAVE_RICH = False
    from tqdm.rich import tqdm  # prefer thin rich-style bars when falling back
    def _tqdm_args(**kwargs):
        mod = getattr(tqdm, "__module__", "")
        if mod.startswith("tqdm.rich"):
            kwargs.pop("position", None)
            kwargs.pop("dynamic_ncols", None)
        return kwargs
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras

from .neural_network import ConvNN
from .mark_flares import FitFlares

__all__ = [
    "predict",
    "predict_ensemble",
    "predict_and_mark",
    "mark_flares_from_preds",
    "remove_false_positives",
]


def _to_np(x):
    if hasattr(x, "value"):
        return np.asarray(x.value)
    return np.asarray(x)


# lightkurve's "default" TESS bitmask (17087) WITHOUT bit 512 ("impulsive outlier"):
# SPOC sets bit 512 on many flare peaks (29 of 442 Günther et al. 2020 peaks in the
# review sample), so dropping those cadences removes the flare peak.
FLARE_SAFE_BITMASK = 17087 & ~512


def _extract_series(
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
    quality_bitmask: int = FLARE_SAFE_BITMASK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accept either a lightkurve LightCurve-like object or (times, flux, flux_err).
    Ensures arrays are numpy and strips astropy units via .value when present.

    For a LightCurve, cadences with ``quality & quality_bitmask != 0`` and NaNs
    are removed. (Before the 2026 review every cadence with quality != 0 was
    removed, which drops flare peaks flagged as impulsive outliers.) Read the
    file with ``quality_bitmask="none"`` so that this is the only selection.
    """
    # LightCurve-like path (duck typing)
    if flux is None and hasattr(lc_or_times, "time") and hasattr(lc_or_times, "flux"):
        lc = lc_or_times
        if hasattr(lc, "quality"):
            q = np.asarray(getattr(lc.quality, "value", lc.quality)).astype(int)
            lc = lc[(q & int(quality_bitmask)) == 0]
        if hasattr(lc, "remove_nans"):
            lc = lc.remove_nans().normalize()

        t = _to_np(getattr(lc.time, "value", getattr(lc, "time", None)))
        f = _to_np(getattr(lc.flux, "value", getattr(lc, "flux", None)))
        if hasattr(lc, "flux_err") and lc.flux_err is not None:
            e = _to_np(getattr(lc.flux_err, "value", getattr(lc, "flux_err", None)))
        else:
            e = np.zeros_like(f)
        return t, f, e

    # Tuple/arrays path
    if flux is None or flux_err is None:
        raise ValueError(
            "Provide either a LightCurve-like object or (times, flux, flux_err) arrays."
        )
    t = _to_np(lc_or_times)
    f = _to_np(flux)
    e = _to_np(flux_err)
    return t, f, e


def predict(
    model_path: str,
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
    verbose: bool = True,
    progress: str = "auto",
    window_batch: Optional[int] = None,
    tqdm_position: Optional[int] = None,
    tqdm_desc: Optional[str] = None,
    rich_progress: Optional[object] = None,
    rich_desc: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single Keras (.keras) model to produce per-cadence predictions.
    Returns (times, flux, errs, preds) for convenience.
    """
    t, f, e = _extract_series(lc_or_times, flux, flux_err)

    cnn = ConvNN(output_dir=".")
    cnn.predict(
        modelname=model_path,
        times=t,
        fluxes=f,
        errs=e,
        verbose=verbose,
        progress=progress,
        window_batch=window_batch,
        tqdm_position=tqdm_position,
        tqdm_desc=tqdm_desc,
        rich_progress=rich_progress,
        rich_desc=rich_desc,
    )
    # predictions is shape (1, N)
    preds = np.asarray(cnn.predictions[0], dtype=float)
    return cnn.predict_time[0], cnn.predict_flux[0], cnn.predict_err[0], preds


def predict_ensemble(
    model_paths: Sequence[str],
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
    aggregate: str = "median",
    verbose: bool = True,
    progress: str = "auto",
    window_batch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run an ensemble of models and aggregate predictions.
    aggregate: 'median' (default; as in the upstream stella tutorial and the 2021 SRSC run) or 'mean'. NaN (unscored) cadences are ignored.
    Returns (times, flux, errs, agg_preds, per_model_preds[models, N]).
    """
    t, f, e = _extract_series(lc_or_times, flux, flux_err)
    per_model = []
    t_ref = f_ref = e_ref = None

    show_outer = verbose and (len(model_paths) > 1)
    if HAVE_RICH and show_outer:
        # Use Rich's track for reliable completion
        for idx, mp in enumerate(track(model_paths, description="Models")):
            tt, ff, ee, pr = predict(
                mp,
                t,
                f,
                e,
                verbose=verbose,
                progress=progress,
                window_batch=window_batch,
                tqdm_position=1,
                tqdm_desc=f"Model {idx+1}/{len(model_paths)}",
            )
            if t_ref is None:
                t_ref, f_ref, e_ref = tt, ff, ee
            per_model.append(pr)
    else:
        # Fallback to tqdm using a context manager (no finally)
        if show_outer:
            with tqdm(total=len(model_paths), desc="Models") as pbar:
                for idx, mp in enumerate(model_paths):
                    tt, ff, ee, pr = predict(
                        mp,
                        t,
                        f,
                        e,
                        verbose=verbose,
                        progress=progress,
                        window_batch=window_batch,
                        tqdm_position=1,
                        tqdm_desc=f"Model {idx+1}/{len(model_paths)}",
                    )
                    if t_ref is None:
                        t_ref, f_ref, e_ref = tt, ff, ee
                    per_model.append(pr)
                    pbar.update(1)
                # ensure visual completion
                if pbar.n < (pbar.total or 0):
                    pbar.update((pbar.total or 0) - pbar.n)
                pbar.refresh()
        else:
            for idx, mp in enumerate(model_paths):
                tt, ff, ee, pr = predict(
                    mp,
                    t,
                    f,
                    e,
                    verbose=verbose,
                    progress=progress,
                    window_batch=window_batch,
                    tqdm_position=1,
                    tqdm_desc=f"Model {idx+1}/{len(model_paths)}",
                )
                if t_ref is None:
                    t_ref, f_ref, e_ref = tt, ff, ee
                per_model.append(pr)

    per_model = np.asarray(per_model, dtype=float)  # object dtype made nanmedian fail
    if aggregate not in ("median", "mean"):
        raise ValueError("aggregate must be 'median' or 'mean'")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cadences no model scored are NaN
        agg = np.nanmedian(per_model, axis=0) if aggregate == "median" else np.nanmean(per_model, axis=0)
    return t_ref, f_ref, e_ref, agg, per_model


def mark_flares_from_preds(
    target_id: Union[str, int],
    times: np.ndarray,
    flux: np.ndarray,
    errs: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5,
):
    """
    Identify flares from precomputed predictions.
    Returns (fit, flare_table).
    """
    fit = FitFlares(
        id=np.asarray([target_id]),
        time=[np.asarray(times, dtype=float)],
        flux=[np.asarray(flux, dtype=float)],
        flux_err=[np.asarray(errs, dtype=float)],
        predictions=[np.asarray(preds, dtype=float)],
    )
    fit.identify_flare_peaks(threshold=threshold)
    return fit, fit.flare_table


def predict_and_mark(
    model_or_models: Union[str, Sequence[str]],
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
    target_id: Union[str, int] = "target",
    threshold: float = 0.5,
    aggregate: str = "median",
    verbose: bool = True,
    dip_window_days: float = 0.3,
):
    """
    Convenience wrapper: predict (single or ensemble) and mark flares.
    Returns (times, flux, errs, preds, flare_table).

    flare_table gains a boolean column ``near_dip`` (candidate within 0.1 d of
    a dip found by ``stella.diagnostics.dip_diagnostics``) and
    ``flare_table.meta['eclipse_like']`` (two or more dips in the light curve).
    """
    if isinstance(model_or_models, (list, tuple)):
        t, f, e, preds, _ = predict_ensemble(
            model_or_models,
            lc_or_times,
            flux,
            flux_err,
            aggregate=aggregate,
            verbose=verbose,
        )
    else:
        t, f, e, preds = predict(
            model_or_models, lc_or_times, flux, flux_err, verbose=verbose
        )

    _, table = mark_flares_from_preds(target_id, t, f, e, preds, threshold=threshold)

    # Flags, never deletions: candidates near short deep dips (eclipses, transits,
    # instrumental drops), where the network is known to fire.
    from .diagnostics import dip_diagnostics, near_times

    diag = dip_diagnostics(t, f, window_days=dip_window_days)
    table["near_dip"] = near_times(np.asarray(table["tpeak"], float), diag["dip_times"], 0.1)
    table.meta["eclipse_like"] = bool(diag["eclipse_like"])
    table.meta["n_dips"] = int(diag["n_dips"])
    return t, f, e, preds, table


def remove_false_positives(
    flare_table,
    min_duration_min: float = 4.0,
    drop_indices: Optional[Sequence[int]] = None,
):
    """
    Basic false-positive filtering for a flare table.

    - Removes flares with fitted duration shorter than `min_duration_min` minutes,
      where duration = (rise + fall) in days converted to minutes.
    - Optionally removes rows by 0-based indices via `drop_indices`.

    Returns a filtered copy of the table.
    """
    from astropy.table import Table
    import numpy as np

    if not isinstance(flare_table, Table):
        # Best-effort cast
        try:
            flare_table = Table(flare_table)
        except Exception:
            raise TypeError("flare_table must be an Astropy Table or table-like object")

    mask = np.ones(len(flare_table), dtype=bool)

    if all(c in flare_table.colnames for c in ("rise", "fall")):
        durations_min = (
            (np.array(flare_table["rise"]) + np.array(flare_table["fall"])) * 24 * 60
        )
        mask &= durations_min >= float(min_duration_min)

    if drop_indices:
        drop_indices = set(int(i) for i in drop_indices)
        keep = [i for i in range(len(flare_table)) if i not in drop_indices]
        mask2 = np.zeros(len(flare_table), dtype=bool)
        mask2[keep] = True
        mask &= mask2

    return flare_table[mask]
