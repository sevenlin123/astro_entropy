import numpy as np
from sklearn.linear_model import Ridge

def gaussian(wavelengths, center, width):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def generate_basis(n_basis, wavelengths):
    """Generate a set of basis spectra as Gaussian profiles."""
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min() + 50, wavelengths.max() - 50, size=n_basis)
    widths = rng.uniform(20, 60, size=n_basis)
    basis = np.array([gaussian(wavelengths, c, w) for c, w in zip(centers, widths)])
    return basis

def generate_filter(center, width, wavelengths):
    """Generate a simple Gaussian filter transmission curve."""
    return gaussian(wavelengths, center, width)

def compute_photometry(spectrum, filters, wavelengths):
    """Integrate spectrum over each filter to obtain photometric fluxes."""
    phot = []
    for filt in filters:
        # weighted average flux through the filter
        numer = np.trapz(spectrum * filt, wavelengths)
        denom = np.trapz(filt, wavelengths)
        phot.append(numer / denom)
    return np.array(phot)

def train_reconstruction_model(
    n_samples,
    n_basis,
    n_filters,
    wavelengths,
    basis,
    filters,
):
    """Train a ridge regression model that maps photometry to full spectra."""
    rng = np.random.default_rng()

    # Generate synthetic training data
    coeffs = rng.uniform(0.0, 1.0, size=(n_samples, n_basis))
    spectra = coeffs @ basis   # shape (n_samples, n_wavelengths)
    photometry = np.array([compute_photometry(s, filters, wavelengths) for s in spectra])

    # Fit ridge regression (multi-output)
    model = Ridge(alpha=1.0, fit_intercept=False, solver='auto')
    model.fit(photometry, spectra)
    return model

def reconstruct_spectrum(model, photometry, wavelengths):
    """Predict full spectrum from photometric observations."""
    pred = model.predict(photometry.reshape(1, -1))
    return pred[0]

def main():
    # Settings
    n_wavelengths = 200
    wavelengths = np.linspace(400, 800, n_wavelengths)  # nm
    n_basis = 5
    n_filters = 3

    # Generate basis spectra and filter transmissions
    basis = generate_basis(n_basis, wavelengths)
    filter_centers = [450, 550, 650]   # nm
    filter_widths = [30, 40, 35]       # nm
    filters = [generate_filter(c, w, wavelengths) for c, w in zip(filter_centers, filter_widths)]

    # Train reconstruction model
    model = train_reconstruction_model(
        n_samples=500,
        n_basis=n_basis,
        n_filters=n_filters,
        wavelengths=wavelengths,
        basis=basis,
        filters=filters,
    )

    # Generate a new synthetic spectrum for testing
    rng = np.random.default_rng(seed=42)
    true_coeffs = rng.uniform(0.0, 1.0, size=n_basis)
    true_spectrum = true_coeffs @ basis
    true_photometry = compute_photometry(true_spectrum, filters, wavelengths)

    # Reconstruct spectrum from photometry
    recon_spectrum = reconstruct_spectrum(model, true_photometry, wavelengths)

    # Simple comparison
    error = np.linalg.norm(recon_spectrum - true_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative reconstruction error: {error:.4f}")
    print("True spectrum first 10 points:", true_spectrum[:10])
    print("Reconstructed spectrum first 10 points:", recon_spectrum[:10])

if __name__ == "__main__":
    main()