import numpy as np
from sklearn.linear_model import Ridge

def create_wavelength_grid(start_nm=300, end_nm=1000, num_points=200):
    """Return linearly spaced wavelengths."""
    return np.linspace(start_nm, end_nm, num_points)

def create_filters(num_filters, wavelengths, rng=np.random.default_rng(42)):
    """Generate Gaussian filter transmittances."""
    filters = []
    for _ in range(num_filters):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform((wavelengths[-1]-wavelengths[0])/10,
                            (wavelengths[-1]-wavelengths[0])/4)
        filt = np.exp(-0.5*((wavelengths - center)/width)**2)
        filters.append(filt)
    return np.vstack(filters)

def generate_base_spectra(num_bases, wavelengths, rng=np.random.default_rng(0)):
    """Return a set of simple sinusoidal base spectra."""
    bases = []
    for i in range(num_bases):
        freq = rng.uniform(0.01, 0.05)
        phase = rng.uniform(0, 2*np.pi)
        amplitude = rng.uniform(0.5, 1.5)
        base = amplitude * np.sin(freq*wavelengths + phase) + 1
        bases.append(base)
    return np.vstack(bases)  # shape (num_bases, num_points)

def generate_random_spectra(n_samples, base_spectra, rng=np.random.default_rng(1)):
    """Produce random linear combinations of base spectra."""
    coeffs = rng.normal(size=(n_samples, base_spectra.shape[0]))
    spectra = coeffs @ base_spectra  # shape (n_samples, num_points)
    return spectra

def compute_photometry(spectra, filters):
    """Integrate spectra over filter transmittances."""
    # spectra shape (n_samples, num_points)
    # filters shape (n_filters, num_points)
    return spectra @ filters.T  # shape (n_samples, n_filters)

def fit_ridge_reconstruction(photometry, spectra, alpha=1.0):
    """Fit a ridge regression model mapping photometry to spectrum."""
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(photometry, spectra)
    return model

def main():
    # 1. Create wavelength grid
    wl = create_wavelength_grid()

    # 2. Create filter set
    filt = create_filters(num_filters=5, wavelengths=wl)

    # 3. Generate synthetic spectra
    bases = generate_base_spectra(num_bases=3, wavelengths=wl)
    true_spectra = generate_random_spectra(n_samples=150, base_spectra=bases)

    # 4. Generate photometric measurements
    phot = compute_photometry(true_spectra, filt)

    # 5. Reconstruct spectra from photometry
    recon_model = fit_ridge_reconstruction(phot, true_spectra, alpha=0.5)
    reconstructed = recon_model.predict(phot)

    # 6. Evaluate reconstruction accuracy
    mse = np.mean((true_spectra - reconstructed)**2)
    print(f"Mean squared reconstruction error: {mse:.4f}")

if __name__ == "__main__":
    main()