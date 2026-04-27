import numpy as np
from sklearn.linear_model import LinearRegression

def spectral_model(wavelength, params):
    """
    Sum of three Gaussian components.
    params: [a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3]
    """
    a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3 = params
    g1 = a1 * np.exp(-0.5 * ((wavelength - mu1) / sigma1)**2)
    g2 = a2 * np.exp(-0.5 * ((wavelength - mu2) / sigma2)**2)
    g3 = a3 * np.exp(-0.5 * ((wavelength - mu3) / sigma3)**2)
    return g1 + g2 + g3

def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Generate synthetic spectra and their parameters.
    """
    rng = np.random.default_rng(42)
    # Parameter bounds
    amps   = rng.uniform(0.2, 1.0, size=(n_samples, 3))
    mus    = rng.uniform(4500, 7500, size=(n_samples, 3))
    sigmas = rng.uniform(50, 200, size=(n_samples, 3))
    params = np.column_stack([amps[:,0], mus[:,0], sigmas[:,0],
                              amps[:,1], mus[:,1], sigmas[:,1],
                              amps[:,2], mus[:,2], sigmas[:,2]])
    spectra = np.zeros((n_samples, len(wavelengths)))
    for i in range(n_samples):
        spectra[i] = spectral_model(wavelengths, params[i])
    return spectra, params

def generate_filters(wavelengths):
    """
    Define three top‑hat filters.
    Returns a dictionary of filter names to transmission arrays.
    """
    filters = {}
    bins = [(5000, 5500), (6000, 6500), (7000, 7500)]
    for i, (lo, hi) in enumerate(bins, start=1):
        trans = np.where((wavelengths >= lo) & (wavelengths <= hi), 1.0, 0.0)
        filters[f'Filter{i}'] = trans
    return filters

def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmissions.
    """
    n_samples = spectra.shape[0]
    filter_names = sorted(filters.keys())
    n_filters = len(filter_names)
    photometry = np.empty((n_samples, n_filters))
    delta_lambda = np.diff(np.concatenate([np.array([0]), spectra[0].size * [1]]))  # dummy spacing
    for k, name in enumerate(filter_names):
        trans = filters[name]
        photometry[:,k] = spectra @ trans
    return photometry

def train_parameter_predictors(photometry, params):
    """
    Train a separate linear regression model for each parameter.
    """
    models = []
    for j in range(params.shape[1]):
        lr = LinearRegression()
        lr.fit(photometry, params[:,j])
        models.append(lr)
    return models

def predict_params(models, photometry):
    """
    Predict parameters from photometry using trained models.
    """
    preds = np.column_stack([m.predict(photometry) for m in models])
    return preds

def reconstruct_spectra_from_params(pred_params, wavelengths):
    """
    Reconstruct spectra from predicted parameters.
    """
    n_samples = pred_params.shape[0]
    spectra = np.zeros((n_samples, len(wavelengths)))
    for i in range(n_samples):
        spectra[i] = spectral_model(wavelengths, pred_params[i])
    return spectra

def main():
    # Wavelength grid
    wavelengths = np.linspace(4000, 8000, 400)

    # Generate synthetic data
    spectra, true_params = generate_synthetic_spectra(50, wavelengths)

    # Filters
    filters = generate_filters(wavelengths)

    # Photometric measurements
    photometry = compute_photometry(spectra, filters)

    # Train regressors to map photometry -> parameters
    models = train_parameter_predictors(photometry, true_params)

    # Predict parameters from photometry
    pred_params = predict_params(models, photometry)

    # Reconstruct spectra from predicted parameters
    recon_spectra = reconstruct_spectra_from_params(pred_params, wavelengths)

    # Evaluate reconstruction error
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.4f}")

if __name__ == "__main__":
    main()