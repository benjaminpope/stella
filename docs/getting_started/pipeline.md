# Pipeline

Typical workflow:

1. Prepare data with `stella.FlareDataSet`.
2. Initialize `stella.ConvNN` with desired backend.
3. Train with `cnn.train_models(...)` or predict with packaged models.

Example prediction using packaged model:

```python
from stella.neural_network import ConvNN
from stella.models import get_model_path

cnn = ConvNN(output_dir="./results")
cnn.predict(
    modelname=get_model_path(),
    times=lc.time.value,
    fluxes=lc.flux.value,
    errs=getattr(lc, "flux_err", None).value if getattr(lc, "flux_err", None) is not None else np.zeros_like(lc.time.value),
)
```

The packaged models are an ensemble. `stella.pipeline.predict_and_mark(stella.models.models, t, f, e)` predicts with all of
them (median by default), finds and fits the candidates and flags the ones near dips. A candidate needs vetting before
it is counted as a flare: see [Vetting and ROC](vetting.md), which also describes how to choose a cut for each star.
