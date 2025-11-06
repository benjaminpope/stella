import os
import numpy as np
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from tqdm.autonotebook import tqdm

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


def _extract_series(
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accept either a lightkurve LightCurve-like object or (times, flux, flux_err).
    Ensures arrays are numpy and strips astropy units via .value when present.
    """
    # LightCurve-like path (duck typing)
    if flux is None and hasattr(lc_or_times, "time") and hasattr(lc_or_times, "flux"):
        lc = lc_or_times
        # Apply filtering whenever a LightCurve-like object is passed
        try:
            if hasattr(lc, "remove_nans"):
                lc = lc.remove_nans().normalize()
            if hasattr(lc, "quality"):
                try:
                    lc = lc[lc.quality == 0]
                except Exception:
                    pass
        except Exception:
            # Best-effort: continue without mutation
            pass

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
    )
    # predictions is shape (1, N)
    preds = np.asarray(cnn.predictions[0])
    return cnn.predict_time[0], cnn.predict_flux[0], cnn.predict_err[0], preds


def predict_ensemble(
    model_paths: Sequence[str],
    lc_or_times: Union[object, Sequence[float], np.ndarray],
    flux: Optional[Union[Sequence[float], np.ndarray]] = None,
    flux_err: Optional[Union[Sequence[float], np.ndarray]] = None,
    aggregate: str = "mean",
    verbose: bool = True,
    progress: str = "auto",
    window_batch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run an ensemble of models and aggregate predictions.
    aggregate: 'mean' or 'median'.
    Returns (times, flux, errs, agg_preds, per_model_preds[models, N]).
    """
    t, f, e = _extract_series(lc_or_times, flux, flux_err)
    per_model = []
    t_ref = f_ref = e_ref = None

    show_outer = verbose and (len(model_paths) > 1)
    pbar = tqdm(total=len(model_paths), desc="Models", position=0, leave=True) if show_outer else None
    try:
        for mp in model_paths:
            tt, ff, ee, pr = predict(
                mp,
                t,
                f,
                e,
                verbose=verbose,
                progress=progress,
                window_batch=window_batch,
                tqdm_position=1,
            )
            if t_ref is None:
                t_ref, f_ref, e_ref = tt, ff, ee
            per_model.append(pr)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    per_model = np.asarray(per_model)
    if aggregate == "median":
        agg = np.nanmedian(per_model, axis=0)
    else:
        agg = np.nanmean(per_model, axis=0)
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
        time=np.asarray([times]),
        flux=np.asarray([flux]),
        flux_err=np.asarray([errs]),
        predictions=np.asarray([preds]),
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
    aggregate: str = "mean",
    verbose: bool = True,
):
    """
    Convenience wrapper: predict (single or ensemble) and mark flares.
    Returns (times, flux, errs, preds, flare_table).
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
