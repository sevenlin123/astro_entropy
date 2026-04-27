import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Spectral model: weighted sum of Gaussian components
# ----------------------------------------------------------------------
def spectral_model(wavelengths, params):
    """
    wavelengths : array_like
        Array of wavelength points (nm).
    params : dict
        Dictionary containing lists:
            'amplitude': list of amplitudes for each component
            'center'   : list of centers for each component
            'sigma'    : list of standard deviations for each component
    Returns:
        spectrum: ndarray of flux values at the provided wavelengths.
    """
    amps = np.array(params['amplitude'])
    ctrs = np.array(params['center'])
    sigs = np.array(params['sigma'])
    spec = np.zeros_like(wavelengths, dtype=float)

    for a, c, s in zip(amps, ctrs, sigs):
        spec += a * np.exp(-0.5 * ((wavelengths - c) / s) ** 2)
    return spec

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra=200, rng=None):
    """
    Generate a set of synthetic spectra.
    Returns:
        wavelengths: 1D array of wavelengths (nm).
        spectra: 2D array of shape (n_spectra, len(wavelengths)).
    """
    if rng is None:
        rng = np.random.default_rng()
    wavelengths = np.linspace(400, 700, 301)  # 400–700 nm

    spectra = []
    for _ in range(n_spectra):
        n_gauss = rng.integers(2, 5)
        amps = rng.uniform(0.5, 1.5, size=n_gauss)
        ctrs = rng.uniform(420, 680, size=n_gauss)
        sigs = rng.uniform(10, 40, size=n_gauss)
        params = {'amplitude': amps, 'center': ctrs, 'sigma': sigs}
        spec = spectral_model(wavelengths, params)
        spectra.append(spec)
    spectra = np.vstack(spectra)
    return wavelengths, spectra

# ----------------------------------------------------------------------
# 3. Generate photometric band responses (top-hat filters)
# ----------------------------------------------------------------------
def band_response(wavelengths, center, width):
    """
    Top‑hat filter response.
    """
    return np.logical_and(wavelengths >= center - width / 2,
                          wavelengths <= center + width / 2).astype(float)

# ----------------------------------------------------------------------
# 4. Compute photometric fluxes from spectra
# ----------------------------------------------------------------------
def compute_photometry(spectra, wavelengths, bands):
    """
    bands: list of tuples (center_nm, width_nm)
    Returns:
        fluxes: 2D array (n_spectra, n_bands)
    """
    n_bands = len(bands)
    fluxes = np.empty((spectra.shape[0], n_bands))
    for i, (cen, wid) in enumerate(bands):
        resp = band_response(wavelengths, cen, wid)
        # integrate spectrum * response over wavelength
        fluxes[:, i] = simps(spectra * resp, wavelengths)
    return fluxes

# ----------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum(fluxes_train, spectra_train, fluxes_test, alpha=1.0):
    """
    Train a linear regressor (Ridge) to map photometric fluxes to full spectra.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(fluxes_train, spectra_train)
    recon = reg.predict(fluxes_test)
    return recon, reg

# ----------------------------------------------------------------------
# 6. Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    rng = np.random.default_rng(42)
    wavelengths, spectra = generate_synthetic_spectra(n_spectra=300, rng=rng)

    # Define photometric bands (Johnson U,B,V)
    bands = [(365, 80),   # U band
             (445, 90),   # B band
             (551, 90)]   # V band

    # Compute photometric fluxes
    fluxes = compute_photometry(spectra, wavelengths, bands)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        fluxes, spectra, test_size=0.2, random_state=0
    )

    # Reconstruct spectra from photometry
    recon_spectra, model = reconstruct_spectrum(X_train, y_train, X_test, alpha=0.5)

    # Evaluate reconstruction (RMSE per spectrum)
    rmse = np.sqrt(np.mean((recon_spectra - y_test)**2, axis=1))
    print(f"Mean RMSE over test set: {rmse.mean():.4f}")

    # Example: plot true vs reconstructed for a single spectrum
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, y_test[idx], label='True spectrum')
        plt.plot(wavelengths, recon_spectra[idx], '--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.title('Spectrum Reconstruction from Photometry')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        # Matplotlib may not be available; ignore plotting
        pass