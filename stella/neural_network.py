import os, glob
import warnings
import numpy as np
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
try:
    from tqdm.rich import tqdm
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm
from .backends import require_backend as _require_backend
_require_backend()
import keras
from scipy.interpolate import interp1d
from astropy.table import Table, Column

__all__ = ["ConvNN", "make_windows", "find_gaps"]


class ConvNN(object):
    """
    Creates and trains the convolutional
    neural network.
    """

    def __init__(
        self,
        output_dir,
        ds=None,
        layers=None,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=None,
    ):
        """
        Creates and trains a Keras model (JAX backend)
        with either layers that have been passed in
        by the user or with default layers used in
        Feinstein et al. (2020), https://arxiv.org/abs/2005.07710.

        Parameters
        ----------
        ds : stella.DataSet object
        output_dir : str
             Path to a given output directory for files.
        training : float, optional
             Assigns the percentage of training set data for training.
             Default is 80%.
        validation : float, optional
             Assigns the percentage of training set data for validation.
             Default is 10%.
        layers : np.array, optional
             An array of keras.layers for the ConvNN.
        optimizer : str, optional
             Optimizer used to compile keras model. Default is 'adam'.
        loss : str, optional
             Loss function used to compile keras model. Default is
             'binary_crossentropy'.
        metrics: np.array, optional
             Metrics used to train the keras model on. If None, metrics are
             [accuracy, precision, recall].
        epochs : int, optional
             Number of epochs to train the keras model on. Default is 15.
        seed : int, optional
             Sets random seed for reproducable results. Default is 2.
        output_dir : path, optional
             The path to save models/histories/predictions to. Default is
             to create a hidden ~/.stella directory.

        Attributes
        ----------
        layers : np.array
        optimizer : str
        loss : str
        metrics : np.array
        training_matrix : stella.TrainingSet.training_matrix
        labels : stella.TrainingSet.labels
        image_fmt : stella.TrainingSet.cadences
        """
        self.ds = ds
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        if ds is not None:
            self.training_matrix = np.copy(ds.training_matrix)
            self.labels = np.copy(ds.labels)
            self.cadences = np.copy(ds.cadences)

            self.frac_balance = ds.frac_balance + 0.0

            self.tpeaks = ds.training_peaks
            self.training_ids = ds.training_ids

        else:
            # Inference-only usage: defer warnings to training-time methods
            pass

        self.prec_recall_curve = None
        self.history = None
        self.history_table = None

        self.output_dir = output_dir

    def create_model(self, seed):
        """
             Creates the Keras model with appropriate layers.

        Attributes
        ----------
             model : keras.models.Sequential
        """
        if getattr(self, "ds", None) is None:
            warnings.warn(
                "No stella.DataSet provided. Training requires ConvNN(ds=...). For inference, use predict()."
            )
            raise ValueError(
                "Training requires a stella.DataSet (ConvNN(ds=...)). For inference, use predict(modelname=..., ...)."
            )
        # SETS RANDOM SEED FOR REPRODUCABLE RESULTS
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)

        # INITIALIZE CLEAN MODEL
        keras.backend.clear_session()

        model = keras.models.Sequential()

        # DEFAULT NETWORK MODEL FROM FEINSTEIN ET AL. (in prep)
        if self.layers is None:
            filter1 = 16
            filter2 = 64
            dense = 32
            dropout = 0.1

            # CONVOLUTIONAL LAYERS
            model.add(
                keras.layers.Conv1D(
                    filters=filter1,
                    kernel_size=7,
                    activation="relu",
                    padding="same",
                    input_shape=(self.cadences, 1),
                )
            )
            model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Dropout(dropout))
            model.add(
                keras.layers.Conv1D(
                    filters=filter2, kernel_size=3, activation="relu", padding="same"
                )
            )
            model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Dropout(dropout))

            # DENSE LAYERS AND SOFTMAX OUTPUT
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(dense, activation="relu"))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(1, activation="sigmoid"))

        else:
            for l in self.layers:
                model.add(l)

        # COMPILE MODEL AND SET OPTIMIZER, LOSS, METRICS
        if self.metrics is None:
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
            )
        else:
            model.compile(
                optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
            )

        self.model = model

        # PRINTS MODEL SUMMARY
        model.summary()

    def load_model(self, modelname, mode="validation"):
        """
        Loads an already created model.

        Parameters
        ----------
        modelname : str
        mode : str, optional
        """
        model = keras.models.load_model(modelname)
        self.model = model

        if getattr(self, "ds", None) is None:
            # No dataset attached; just load model for inference and return
            return

        if mode == "test":
            pred = model.predict(self.ds.test_data)
        elif mode == "validation":
            pred = model.predict(self.ds.val_data)
        pred = np.reshape(pred, len(pred))

        # Placeholder for metrics calculation
        return

    def train_models(
        self,
        seeds=[2],
        epochs=350,
        batch_size=64,
        shuffle=False,
        pred_test=False,
        save=False,
    ):
        """
        Runs n number of models with given initial random seeds of
        length n. Also saves each model run to a hidden ~/.stella
        directory.

        Parameters
        ----------
        seeds : np.array
             Array of random seed starters of length n, where
             n is the number of models you want to run.
        epochs : int, optional
             Number of epochs to train for. Default is 350.
        batch_size : int, optional
             Setting the batch size for the training. Default
             is 64.
        shuffle : bool, optional
             Allows for shuffling of the training set when fitting
             the model. Default is False.
        pred_test : bool, optional
             Allows for predictions on the test set. DO NOT SET TO
             TRUE UNTIL YOU'VE DECIDED ON YOUR FINAL MODEL. Default
             is False.
        save : bool, optional
             Saves the predictions and histories of from each model
             in an ascii table to the specified output directory.
             Default is False.

        Attributes
        ----------
        history_table : Astropy.table.Table
             Saves the metric values for each model run.
        val_pred_table : Astropy.table.Table
             Predictions on the validation set from each run.
        test_pred_table : Astropy.table.Table
             Predictions on the test set from each run. Must set
             pred_test = True, or else it is an empty table.
        """

        if type(seeds) == int or type(seeds) == float or type(seeds) == np.int64:
            seeds = np.array([seeds])

        if getattr(self, "ds", None) is None:
            warnings.warn(
                "No stella.DataSet provided. Training requires ConvNN(ds=...). For inference, use predict()."
            )
            raise ValueError(
                "Training requires a stella.DataSet (ConvNN(ds=...)). For inference, use predict(modelname=..., ...)."
            )

        self.epochs = epochs

        # CREATES TABLES FOR SAVING DATA
        table = Table()
        val_table = Table(
            [self.ds.val_ids, self.ds.val_labels, self.ds.val_tpeaks],
            names=["tic", "gt", "tpeak"],
        )
        test_table = Table(
            [self.ds.test_ids, self.ds.test_labels, self.ds.test_tpeaks],
            names=["tic", "gt", "tpeak"],
        )

        for seed in seeds:

            fmt_tail = "_s{0:04d}_i{1:04d}_b{2}".format(
                int(seed), int(epochs), self.frac_balance
            )
            model_fmt = "ensemble" + fmt_tail + ".keras"

            keras.backend.clear_session()

            # CREATES MODEL BASED ON GIVEN RANDOM SEED
            self.create_model(seed)
            self.history = self.model.fit(
                self.ds.train_data,
                self.ds.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=(self.ds.val_data, self.ds.val_labels),
            )

            col_names = list(self.history.history.keys())
            for cn in col_names:
                col = Column(
                    self.history.history[cn], name=cn + "_s{0:04d}".format(int(seed))
                )
                table.add_column(col)

            # SAVES THE MODEL TO OUTPUT DIRECTORY
            self.model.save(os.path.join(self.output_dir, model_fmt))

            # GETS PREDICTIONS FOR EACH VALIDATION SET LIGHT CURVE
            val_preds = self.model.predict(self.ds.val_data)
            val_preds = np.reshape(val_preds, len(val_preds))
            val_table.add_column(
                Column(val_preds, name="pred_s{0:04d}".format(int(seed)))
            )

            # GETS PREDICTIONS FOR EACH TEST SET LIGHT CURVE IF PRED_TEST IS TRUE
            if pred_test is True:
                test_preds = self.model.predict(self.ds.test_data)
                test_preds = np.reshape(test_preds, len(test_preds))
                test_table.add_column(
                    Column(test_preds, name="pred_s{0:04d}".format(int(seed)))
                )

        # SETS TABLE ATTRIBUTES
        self.history_table = table
        self.val_pred_table = val_table
        self.test_pred_table = test_table

        # SAVES TABLE IS SAVE IS TRUE
        if save is True:
            fmt_table = "_i{0:04d}_b{1}.txt".format(int(epochs), self.frac_balance)
            hist_fmt = "ensemble_histories" + fmt_table
            pred_fmt = "ensemble_predval" + fmt_table

            table.write(os.path.join(self.output_dir, hist_fmt), format="ascii")
            val_table.write(
                os.path.join(self.output_dir, pred_fmt),
                format="ascii",
                fast_writer=False,
            )

            if pred_test is True:
                test_fmt = "ensemble_predtest" + fmt_table
                test_table.write(
                    os.path.join(self.output_dir, test_fmt),
                    format="ascii",
                    fast_writer=False,
                )

    def cross_validation(
        self,
        seed=2,
        epochs=350,
        batch_size=64,
        n_splits=5,
        shuffle=False,
        pred_test=False,
        save=False,
    ):
        """
        Performs cross validation for a given number of K-folds.
        Reassigns the training and validation sets for each fold.

        Parameters
        ----------
        seed : int, optional
             Sets random seed for creating CNN model. Default is 2.
        epochs : int, optional
             Number of epochs to run each folded model on. Default is 350.
        batch_size : int, optional
             The batch size for training. Default is 64.
        n_splits : int, optional
             Number of folds to perform. Default is 5.
        shuffle : bool, optional
             Allows for shuffling in scikitlearn.model_slection.KFold.
             Default is False.
        pred_test : bool, optional
             Allows for predicting on the test set. DO NOT SET TO TRUE UNTIL
             YOU ARE HAPPY WITH YOUR FINAL MODEL. Default is False.
        save : bool, optional
             Allows the user to save the kfolds table of predictions.
             Defaul it False.

        Attributes
        ----------
        crossval_predval : astropy.table.Table
             Table of predictions on the validation set from each fold.
        crossval_predtest : astropy.table.Table
             Table of predictions on the test set from each fold. ONLY
             EXISTS IF PRED_TEST IS TRUE.
        crossval_histories : astropy.table.Table
             Table of history values from the model run on each fold.
        """

        if getattr(self, "ds", None) is None:
            warnings.warn(
                "No stella.DataSet provided. Training requires ConvNN(ds=...). For inference, use predict()."
            )
            raise ValueError(
                "Training requires a stella.DataSet (ConvNN(ds=...)). For inference, use predict(modelname=..., ...)."
            )

        from sklearn.model_selection import KFold
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score

        num_flares = len(self.labels)
        trainval_cutoff = int(0.90 * num_flares)

        tab = Table()
        predtab = Table()

        x_trainval = self.training_matrix[0:trainval_cutoff]
        y_trainval = self.labels[0:trainval_cutoff]
        p_trainval = self.tpeaks[0:trainval_cutoff]
        t_trainval = self.training_ids[0:trainval_cutoff]

        kf = KFold(n_splits=n_splits, shuffle=shuffle)

        if pred_test is True:
            pred_test_table = Table()

        i = 0
        for ti, vi in kf.split(y_trainval):
            # CREATES TRAINING AND VALIDATION SETS
            x_train = x_trainval[ti]
            y_train = y_trainval[ti]
            x_val = x_trainval[vi]
            y_val = y_trainval[vi]

            p_val = p_trainval[vi]
            t_val = t_trainval[vi]

            # REFORMAT TO ADD ADDITIONAL CHANNEL TO DATA
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

            # CREATES MODEL AND RUNS ON REFOLDED TRAINING AND VALIDATION SETS
            self.create_model(seed)
            history = self.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=(x_val, y_val),
            )

            # SAVES THE MODEL BY DEFAULT
            self.model.save(
                os.path.join(
                    self.output_dir,
                    "crossval_s{0:04d}_i{1:04d}_b{2}_f{3:04d}.keras".format(
                        int(seed), int(epochs), self.frac_balance, i
                    ),
                )
            )

            # CALCULATE METRICS FOR VALIDATION SET
            pred_val = self.model.predict(x_val)
            pred_val = np.reshape(pred_val, len(pred_val))

            # SAVES PREDS FOR VALIDATION SET
            tab_names = ["id", "gt", "peak", "pred"]
            data = [t_val, y_val, p_val, pred_val]
            for j, tn in enumerate(tab_names):
                col = Column(data[j], name=tn + "_f{0:03d}".format(i))
                predtab.add_column(col)

            # PREDICTS ON TEST SET IF PRED_TEST IS TRUE
            if pred_test is True:
                preds = self.model.predict(self.ds.test_data)
                preds = np.reshape(preds, len(preds))
                data = [
                    self.ds.test_ids,
                    self.ds.test_labels,
                    self.ds.test_tpeaks,
                    np.reshape(preds, len(preds)),
                ]
                for j, tn in enumerate(tab_names):
                    col = Column(data[j], name=tn + "_f{0:03d}".format(i))
                    pred_test_table.add_column(col)
                self.crossval_predtest = pred_test_table

            precision, recall, _ = precision_recall_curve(y_val, pred_val)
            ap_final = average_precision_score(y_val, pred_val, average=None)

            # SAVES HISTORIES TO A TABLE
            col_names = list(history.history.keys())
            for cn in col_names:
                col = Column(history.history[cn], name=cn + "_f{0:03d}".format(i))
                tab.add_column(col)

            # KEEPS TRACK OF WHICH FOLD
            i += 1

        # SETS TABLES AS ATTRIBUTES
        self.crossval_predval = predtab
        self.crossval_histories = tab

        # IF SAVE IS TRUE, SAVES TABLES TO OUTPUT DIRECTORY
        if save is True:
            fmt = "crossval_{0}_s{1:04d}_i{2:04d}_b{3}.txt"
            predtab.write(
                os.path.join(
                    self.output_dir,
                    fmt.format("predval", int(seed), int(epochs), self.frac_balance),
                ),
                format="ascii",
                fast_writer=False,
            )
            tab.write(
                os.path.join(
                    self.output_dir,
                    fmt.format("histories", int(seed), int(epochs), self.frac_balance),
                ),
                format="ascii",
                fast_writer=False,
            )

            # SAVES TEST SET PREDICTIONS IF TRUE
            if pred_test is True:
                pred_test_table.write(
                    os.path.join(
                        self.output_dir,
                        fmt.format(
                            "predtest", int(seed), int(epochs), self.frac_balance
                        ),
                    ),
                    format="ascii",
                    fast_writer=False,
                )

    def calibration(self, df, metric_threshold):
        """
        Transform the rankings output by the CNN into probabilities.
        This can only be run for an ensemble of models.

        Parameters
        ----------
        df : astropy.table.Table
             Table of output predictions from the validation set.
        metric_threshold : float
             Defines ranking above which something is considered a flare.
        """
        # ADD COLUMN TO TABLE THAT CALCULATES THE FRACTION OF MODELS
        # THAT SAY SOMETHING IS A FLARE
        names = [i for i in df.colnames if "s" in i]
        flare_frac = np.zeros(len(df))

        for i in range(len(df)):
            preds = np.array(list(df[names][i]))
            flare_frac[i] = len(preds[preds >= metric_threshold]) / len(preds)

        df.add_column(Column(flare_frac, name="flare_frac"))

        # Placeholder for further calibration steps
        return df


    def predict(
        self,
        modelname,
        times,
        fluxes,
        errs,
        multi_models=False,
        injected=False,
        verbose=True,
        progress: str = "auto",
        window_batch: int = None,
        tqdm_position: int = None,
        tqdm_desc: str = None,
        rich_progress: object = None,
        rich_desc: str = None,
        max_gap_cadences: float = 5.5,
        cadence_days: float = None,
    ):
        """
        Takes in arrays of time and flux and predicts where the flares
        are based on the keras model created and trained.

        Parameters
        ----------
        modelname : str
             Path and filename of a model to load.
        times, fluxes, errs : np.ndarray or list of np.ndarray
             One light curve (1-D arrays) or a list of light curves, which
             may have different lengths.
        max_gap_cadences : float
             A step between consecutive cadences longer than this many
             nominal cadences is a gap. Default 5.5 (11 min at 2-min
             cadence): up to four missing cadences in a row are tolerated,
             since single dropped or flagged cadences are common. Upstream
             stella used median(dt) + 1.5 std(dt), which in a single sector
             counts almost every dropped cadence as a gap.
        cadence_days : float, optional
             Nominal cadence; default is the median time step.

        Notes
        -----
        The window for cadence i is flux[i-100 : i+100] (cadence i sits at
        position 100 of 0..199), as in training. A cadence whose window would
        run past an end of the series or across a gap is NOT scored: its
        prediction is NaN. (Upstream stella scored a window of zeros there,
        which returns about 0.01 to 0.11 depending on the model.)

        Attributes
        ----------
        predict_time, predict_flux, predict_err : list of np.ndarray
        predictions : list of np.ndarray (float, NaN where not scored)
        """
        model = keras.models.load_model(modelname)
        self.model = model

        cadences = int(model.input_shape[1])

        # One light curve (1-D sequence of numbers) or a list of them
        if len(times) == 0 or np.ndim(times[0]) == 0:
            times, fluxes, errs = [times], [fluxes], [errs]

        predictions, pred_t, pred_f, pred_e = [], [], [], []
        bs = window_batch if window_batch is not None else max(1024, cadences)
        for j in range(len(times)):
            time = np.asarray(times[j], dtype=float)
            lc = np.asarray(fluxes[j], dtype=float)
            err = np.asarray(errs[j], dtype=float) if errs is not None else np.zeros_like(time)

            q = np.isfinite(time) & np.isfinite(lc)
            if err.shape == time.shape:
                q &= np.isfinite(err)
            time, lc, err = time[q], lc[q], err[q]
            med = np.nanmedian(lc) if len(lc) else 1.0
            if not np.isfinite(med) or med == 0.0:
                med = 1.0
            lc = lc / med  # MUST BE NORMALIZED (as in training)

            pred_t.append(time)
            pred_f.append(lc)
            pred_e.append(err)

            X, scored = make_windows(time, lc, cadences,
                                     max_gap_cadences=max_gap_cadences,
                                     cadence_days=cadence_days)
            preds = np.full(len(lc), np.nan)
            idx = np.where(scored)[0]
            show = verbose and progress in ("auto", "windows") and len(idx) > 0
            task_id = None
            if show and rich_progress is not None and HAVE_RICH:
                task_id = rich_progress.add_task(rich_desc or tqdm_desc or "Model Predict", total=len(idx))
            wbar = None
            if show and task_id is None:
                kw = dict(total=len(idx), desc=(tqdm_desc or "Model Predict"), leave=False)
                if not getattr(tqdm, "__module__", "").startswith("tqdm.rich"):
                    kw.update(position=(tqdm_position or 1), dynamic_ncols=True)
                wbar = tqdm(**kw)
            for i0 in range(0, len(idx), bs):
                sl = idx[i0:i0 + bs]
                out = model.predict(X[sl][..., None], verbose=0)
                preds[sl] = np.reshape(out, (len(sl),))
                if task_id is not None:
                    rich_progress.update(task_id, advance=len(sl))
                if wbar is not None:
                    wbar.update(len(sl))
            if wbar is not None:
                wbar.close()
            predictions.append(preds)

        def _stack(arrs):
            # (n_lc, N) float array when lengths agree; otherwise a 1-D object array
            # of float arrays (np.asarray on ragged input raises under NumPy >= 1.24).
            if len({len(a) for a in arrs}) <= 1:
                return np.asarray(arrs, dtype=float)
            out = np.empty(len(arrs), dtype=object)
            out[:] = [np.asarray(a, dtype=float) for a in arrs]
            return out

        self.predict_time = _stack(pred_t)
        self.predict_flux = _stack(pred_f)
        self.predict_err = _stack(pred_e)
        self.predictions = _stack(predictions)

        if injected:
            return (
                self.predict_time,
                self.predict_flux,
                self.predict_err,
                self.predictions,
            )


def find_gaps(time, max_gap_cadences=5.5, cadence_days=None):
    """Indices b such that the step time[b] -> time[b+1] is a gap, i.e. longer
    than ``max_gap_cadences`` nominal cadences (the median step by default)."""
    diff = np.diff(time)
    if len(diff) == 0:
        return np.array([], dtype=int)
    cad = cadence_days if cadence_days is not None else np.nanmedian(diff)
    return np.where(diff > max_gap_cadences * cad)[0]


def make_windows(time, lc, cadences=200, max_gap_cadences=5.5, cadence_days=None):
    """Build one window per cadence, flux[i - cadences//2 : i + cadences//2].

    Returns (X, scored): X has shape (n, cadences) and holds data only where
    ``scored`` is True, i.e. the whole window lies inside the series and does
    not cross a gap.
    """
    n = len(lc)
    half = cadences // 2
    X = np.zeros((n, cadences), dtype=np.float32)
    scored = np.zeros(n, dtype=bool)
    if n < cadences:
        return X, scored
    seg = np.zeros(n, dtype=int)
    gaps = find_gaps(time, max_gap_cadences, cadence_days)
    seg[gaps + 1] = 1
    seg = np.cumsum(seg)  # segment label of each cadence
    i = np.arange(half, n - half + 1)
    ok = seg[i - half] == seg[i + half - 1]
    for k in i[ok]:
        X[k] = lc[k - half:k + half]
    scored[i[ok]] = True
    return X, scored
