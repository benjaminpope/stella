import numpy as np
import pytest
from numpy.testing import assert_almost_equal

pytestmark = pytest.mark.integration

def _load_lc():
    from lightkurve.search import search_lightcurve
    lk = search_lightcurve(target='tic62124646', mission='TESS',
                           exptime=120, sector=13, author='SPOC')
    lk = lk.download(download_dir='.')
    lk = lk.remove_nans().normalize()
    return lk

def test_predictions():
    from stella.neural_network import ConvNN
    from stella import models as sm
    lk = _load_lc()
    modelname = sm.get_model_path()
    cnn = ConvNN(output_dir='.')
    cnn.predict(modelname=modelname,
                times=lk.time.value,
                fluxes=lk.flux.value,
                errs=lk.flux_err.value)
    high_flares = np.where(cnn.predictions[0]>0.99)[0]
    assert(len(high_flares) == 0)

def find_flares():
    from stella import FitFlares
    lk = _load_lc()
    flares = FitFlares(id=[lk.targetid],
                       time=[lk.time.value],
                       flux=[lk.flux.value],
                       flux_err=[lk.flux_err.value],
                       predictions=[np.zeros_like(lk.flux.value)])

    flares.identify_flare_peaks()
    assert(len(flares.flare_table)==0)
