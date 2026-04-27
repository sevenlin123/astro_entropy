import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

def gaussian_waveform(amplitude, center, sigma, wvl):
    return amplitude * np.exp(-(wvl - center)**2 / (2 * sigma**2))

def generate_synthetic_spectrum(wvl, params):
    """params: list of tuples (amplitude, center, sigma)"""
    flux = np.zeros_like(wvl)
    for amp, cen, sig in params:
        flux += gaussian_waveform(amp, cen, sig, wvl)
    return flux

def create_synthetic_dataset(n_samples, wvl):
    rng = np.random.default_rng(42)
    spectra = []
    all_params = []
    for _ in range(n_samples):
        n_lines = rng.integers(2, 5)
        params = []
        for _ in range(n_lines):
            amp = rng.uniform(0.5, 1.5)
            cen = rng.uniform(350, 950)
            sig = rng.uniform(5, 20)
            params.append((amp, cen, sig))
        spec = generate_synthetic_spectrum(wvl, params)
        spectra.append(spec)
        all_params.append(params)
    return np.array(spectra), all_params

def gaussian_filter(center, sigma, wvl):
    return np.exp(-(wvl - center)**2 / (2 * sigma**2))

def create_filters(wvl, centers, sigmas):
    filters = []
    for cen, sig in zip(centers, sigmas):
        filt = gaussian_filter(cen, sig, wvl)
        filt /= filt.max()  # normalize to max 1
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, len(wvl))

def compute_photometry(spectra, filters, wvl):
    """spectra: (n_samples, len(wvl)), filters: (n_filters, len(wvl))"""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        trans = filters[i]
        phot[:, i] = simps(spectra * trans[None, :], x=wvl, axis=1)
    return phot

def build_basis_functions(wvl, n_basis):
    """Return basis matrix of shape (len(wvl), n_basis)"""
    rng = np.random.default_rng(24)
    means = rng.uniform(wvl.min(), wvl.max(), size=n_basis)
    sigmas = rng.uniform(5, 30, size=n_basis)
    basis = np.column_stack([gaussian_waveform(1.0, m, s, wvl) for m, s in zip(means, sigmas)])
    return basis  # no normalization

def reconstruct_spectra_from_photometry(photometry, filters, wvl, n_basis=30, alpha=1.0):
    """
    photometry: (n_samples, n_filters)
    filters: (n_filters, len(wvl))
    returns: reconstructed spectra (n_samples, len(wvl))
    """
    n_filters, n_wvl = filters.shape
    basis = build_basis_functions(wvl, n_basis)   # (len(wvl), n_basis)
    # Compute design matrix X of shape (n_filters, n_basis)
    X = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for k in range(n_basis):
            X[j, k] = simps(basis[:, k] * filters[j], x=wvl)
    # Fit Ridge regression per sample
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, photometry.T)  # training: X (n_filters, n_basis) -> photometry.T (n_filters, n_samples)
    coeffs = ridge.coef_.T  # shape (n_samples, n_basis)
    # Reconstruct spectra
    recon = coeffs @ basis.T  # (n_samples, len(wvl))
    return recon

def main():
    # Wavelength grid
    wvl = np.linspace(300, 1000, 400)  # nm
    # Generate synthetic spectra
    spectra, params_list = create_synthetic_dataset(n_samples=10, wvl=wvl)
    # Define filters
    filter_centers = [400, 500, 600, 700, 800]
    filter_sigmas = [30, 30, 30, 30, 30]
    filters = create_filters(wvl, filter_centers, filter_sigmas)
    # Compute photometry
    photometry = compute_photometry(spectra, filters, wvl)
    # Reconstruct spectra
    recon_spectra = reconstruct_spectra_from_photometry(photometry, filters, wvl,
                                                        n_basis=50, alpha=0.1)
    # Evaluate reconstruction
    mse = np.mean((spectra - recon_spectra)**2, axis=1)
    print("Mean squared error per spectrum:", mse)
    print("Average MSE:", mse.mean())

if __name__ == "__main__":
    main()