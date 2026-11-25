<p align="center">
  <img width = "500" src="./figures/stella_logo.png"/>
</p>

<p align="center">
  <a href="https://github.com/afeinstein20/stella/actions?query=workflow%3Astella-tests"><img src="https://github.com/afeinstein20/stella/workflows/stella-tests/badge.svg"?color=D35968/></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2020AJ....160..219F/abstract"><img src="https://img.shields.io/badge/read-the_paper-3C1370.svg?style=flat"/></a>
  <a href="https://afeinstein20.github.io/stella/"><img src="https://img.shields.io/badge/read-the_docs-3C1370.svg?style=flat"/></a>
  <a href="https://pypi.org/project/stella"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/stella?color=D35968"></a>
  <a href="https://doi.org/10.21105/joss.02347">   <img src="https://joss.theoj.org/papers/10.21105/joss.02347/status.svg?color=D35968"></a>
  <!-- Fork-specific docs status -->
  <a href="https://github.com/benjaminpope/stella/actions/workflows/docs.yml"><img src="https://github.com/benjaminpope/stella/actions/workflows/docs.yml/badge.svg" alt="Docs Deploy"></a>
  <a href="https://benjaminpope.github.io/stella/"><img src="https://img.shields.io/badge/docs-fork_pages-3C1370.svg?style=flat" alt="GitHub Pages"></a>
</p>


</p>
stella is a Python package to create and train a neural network to identify stellar flares.
Within stella, users can simulate flares as a training set, run a neural network, and feed
in their own data to the neural network model. stella returns a probability at each data point
that that data point is part of a flare or not. stella can also characterize the flares identified.
</p>


To install stella with pip:

	pip install stella

Alternatively you can install the current development version of stella:

        git clone https://github.com/afeinstein20/stella
        cd stella
        python setup.py install

Backends and models
-------------------
- stella uses Keras Core and works with multiple backends: JAX or PyTorch. Choose one, install, and set it before importing `keras`:

```bash
# JAX
pip install -r requirements-jax.txt
export KERAS_BACKEND=jax

# or PyTorch
pip install -r requirements-torch.txt
export KERAS_BACKEND=torch
```

 Models are saved/loaded in the native Keras format (`.keras`). If you have older `.h5`
  or legacy SavedModel models, use the included converter:

```bash
# Convert .h5 or legacy SavedModel to .keras (TensorFlow backend only for conversion)
python scripts/convert_h5_to_keras.py /path/to/models -r --cadences 350 -o /path/to/out
```

Changes in 2026 (review of the JAX/Keras 3 port)
------------------------------------------------
A review of the port (see the pull request that added `stella.vetting`) changed several defaults.
The network weights are unchanged and bitwise equal to the published models.

- **NumPy 2**: `np.trapz` is no longer used. `predict_and_mark` accepts the default string `target_id`;
  ragged multi-sector input, the median ensemble and gaps near the end of a light curve no longer crash.
- **Ensemble aggregate**: `predict_ensemble` and `predict_and_mark` default to `aggregate="median"` (was `"mean"`),
  as in the published method.
- **Windows and gaps**: `ConvNN.predict(..., max_gap_cadences=5.5)`. A gap is a step longer than 5.5 cadences (11 min
  at 2-min cadence). A cadence whose 200-cadence window runs past an end or across a gap is not scored: its
  prediction is NaN (upstream zero-filled the window). Probabilities are float arrays, an object array when
  light curves differ in length.
- **Quality mask**: `stella.pipeline.FLARE_SAFE_BITMASK` (lightkurve's default bitmask without bit 512, "impulsive
  outlier", which SPOC sets on flare peaks) replaces `quality == 0` for lightkurve input. Read the file with
  `quality_bitmask="none"` so that this is the only selection.
- **`FitFlares`**: one fitting routine (the rich and tqdm copies of the loop are gone); duplicate peaks are reported
  once; fit windows are clipped at the ends instead of being skipped silently or mis-centred; new arguments
  `cadence_days`, `gap_factor` (5.5), `rise_bounds`, `fall_bounds` and `min_amp` (1.002, the amplitude cut that
  was hard-coded); new columns `prob_max`, `snr_peak`, `n_consec_3sigma`, `n_above`, `dist_gap`, `fit_at_bound`, `rise_unresolved`.
- **Flags, never deletions**: `stella.diagnostics.dip_diagnostics` and the `near_dip` column and
  `eclipse_like` metadata of `predict_and_mark` mark candidates near eclipses and transits. Masking eclipses is not done.
- **Per-star ROC vetting**: `stella.vetting.flare_roc` chooses a threshold for a stated false-positive rate per day on
  that star, using its flux-inverted light curve and injected flares; see
  [Vetting flare candidates](docs/getting_started/vetting.md) for a worked example and the limits.
- `MeasureProt` Lomb-Scargle periods are not reliable for eclipsing binaries or slow rotators: do not use them there.
  (`averaged_per_sector` had swapped `np.full` arguments; fixed.)

Pipeline quickstart
-------------------
- See the new pipeline guide for a friendly, end-to-end example (TIC 62124646):
  https://afeinstein20.github.io/stella/getting_started/pipeline.html

Swap backends quickly
---------------------
- Inspect availability and devices:

```python
import stella
stella.check_backend()
```

- Prepare a swap (restart your interpreter after):

```python
import stella
stella.swap_backend('torch', accelerator='mps')   # Apple Silicon
# or
stella.swap_backend('jax', accelerator='cpu')
```
```

<p>
If your work uses the stella software, please cite <a href="https://ui.adsabs.harvard.edu/abs/2020JOSS....5.2347F/abstract">Feinstein, Montet, & Ansdell (2020)</a>.
</p>
<p>
If your work discusses the flare rate of young stars in the TESS Southern Hemisphere or the details of the CNNs, please cite <a href="https://ui.adsabs.harvard.edu/abs/2020arXiv200507710F/abstract">Feinstein et al. (AJ, 2020)</a>.
</p>

<p>
<b><u>Bug Reports, Questions, & Contributions</u></b>
</p>
<p>
stella is an open source project under the MIT license. 
The source code is available on GitHub. In case of any questions or problems, please contact us via the Git Issues. 
Pull requests are also welcome through the GitHub page.
</p>