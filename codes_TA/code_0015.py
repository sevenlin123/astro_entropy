import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -------------------- Spectral model --------------------
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-0.5 * ((x - cen) / wid)**2)

def synthetic_spectrum(wls, n_comp=3, seed=None):
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.5, 1.5, size=n_comp)
    cents = rng.uniform(wls.min(), wls.max(), size=n_comp)
    widths = rng.uniform(10, 30, size=n_comp)
    spec = np.zeros_like(wls)
    for a, c, w in zip(amps, cents, widths):
        spec += gaussian(wls, a, c, w)
    # add weak continuum
    spec += rng.normal(0, 0.05, size=len(wls))
    return spec

# -------------------- Filter responses --------------------
def gaussian_filter(wls, cen, wid):
    filt = gaussian(wls, 1.0, cen, wid)
    return filt / filt.sum()  # normalize

def build_filters(wls, n_filt=5):
    centers = np.linspace(wls.min()+50, wls.max()-50, n_filt)
    wid = 100.0
    return [gaussian_filter(wls, c, wid) for c in centers]

# -------------------- Photometry --------------------
def compute_photometry(specs, filters):
    # specs: (n_samp, n_wls)
    # filters: list of arrays (n_wls,)
    filt_arr = np.array(filters)  # (n_filt, n_wls)
    return specs @ filt_arr.T  # (n_samp, n_filt)

# -------------------- Reconstruction --------------------
def train_ridge(X_train, Y_train, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, Y_train)
    return model

def reconstruct_spectrum(model, X_new):
    return model.predict(X_new)

# -------------------- Main workflow --------------------
if __name__ == "__main__":
    np.random.seed(42)

    # wavelength grid
    wl_start, wl_end, n_wls = 4000, 8000, 200
    wls = np.linspace(wl_start, wl_end, n_wls)

    # filters
    filters = build_filters(wls, n_filt=5)

    # training data
    n_train = 200
    spectra_train = np.vstack([synthetic_spectrum(wls, seed=i) for i in range(n_train)])
    photometry_train = compute_photometry(spectra_train, filters)

    # test data
    n_test = 20
    spectra_test = np.vstack([synthetic_spectrum(wls, seed=100+i) for i in range(n_test)])
    photometry_test = compute_photometry(spectra_test, filters)

    # train model
    ridge = train_ridge(photometry_train, spectra_train, alpha=1.0)

    # predict
    spectra_pred = reconstruct_spectrum(ridge, photometry_test)

    # evaluate
    mse = mean_squared_error(spectra_test, spectra_pred)
    print(f"Mean Squared Error on test set: {mse:.4f}")