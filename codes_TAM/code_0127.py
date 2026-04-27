import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# ------------------------ Spectral model ------------------------

def gaussian_basis(wavelength, center, width):
    """Single Gaussian basis function."""
    return np.exp(-((wavelength - center) ** 2) / (2 * width ** 2))

def generate_basis_functions(n_basis, wavelength):
    """Generate a set of Gaussian basis functions with random parameters."""
    centers = np.random.uniform(450, 750, n_basis)
    widths  = np.random.uniform(20, 40, n_basis)
    return [gaussian_basis(wavelength, c, w) for c, w in zip(centers, widths)]

def synthesize_spectrum(weights, basis_funcs):
    """Combine basis functions with given weights."""
    spectrum = np.zeros_like(basis_funcs[0])
    for w, b in zip(weights, basis_funcs):
        spectrum += w * b
    return spectrum

# ------------------------ Photometry ------------------------

def filter_response(wavelength, center, width):
    """Simple Gaussian filter transmission curve."""
    return np.exp(-((wavelength - center) ** 2) / (2 * width ** 2))

def compute_photometry(spectrum, wavelength, filters):
    """Integrate spectrum through each filter."""
    phot = []
    for filt in filters:
        trans = filter_response(wavelength, filt['center'], filt['width'])
        flux = np.trapz(spectrum * trans, wavelength) / np.trapz(trans, wavelength)
        phot.append(flux)
    return np.array(phot)

# ------------------------ Data generation ------------------------

def generate_dataset(n_samples, n_basis, wavelength, filters):
    """Create synthetic spectra and corresponding photometric data."""
    basis_funcs = generate_basis_functions(n_basis, wavelength)
    spectra = []
    photometry = []
    weights_list = []

    for _ in range(n_samples):
        weights = np.random.uniform(0.5, 1.5, n_basis)
        spec = synthesize_spectrum(weights, basis_funcs)
        spectra.append(spec)
        photometry.append(compute_photometry(spec, wavelength, filters))
        weights_list.append(weights)

    return (
        np.vstack(spectra),
        np.vstack(photometry),
        np.vstack(weights_list)
    )

# ------------------------ Reconstruction ------------------------

def train_reconstruction_model(X, Y, alpha=1.0):
    """Train a Ridge regression model to map photometry to spectrum."""
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model

def reconstruct_spectrum(model, photometry_sample):
    """Predict spectrum from photometry using the trained model."""
    return model.predict(photometry_sample.reshape(1, -1))[0]

# ------------------------ Example usage ------------------------

if __name__ == "__main__":
    # Wavelength grid (400–800 nm)
    wavelength = np.linspace(400, 800, 100)

    # Define five broadband filters (U, B, V, R, I)
    filter_defs = [
        {'name': 'U', 'center': 365, 'width': 30},
        {'name': 'B', 'center': 445, 'width': 35},
        {'name': 'V', 'center': 551, 'width': 40},
        {'name': 'R', 'center': 658, 'width': 45},
        {'name': 'I', 'center': 806, 'width': 50},
    ]

    # Generate synthetic dataset
    n_samples = 200
    n_basis   = 5
    spectra, photometry, _ = generate_dataset(
        n_samples, n_basis, wavelength, filter_defs
    )

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=42
    )

    # Train reconstruction model
    recon_model = train_reconstruction_model(X_train, Y_train, alpha=10.0)

    # Evaluate on test set
    Y_pred = recon_model.predict(X_test)
    rmse = np.sqrt(((Y_test - Y_pred) ** 2).mean())
    print(f"Test RMSE per wavelength point: {rmse:.4f}")

    # Reconstruct a single spectrum from its photometry
    idx = 0  # first test sample
    true_spec = Y_test[idx]
    phot_sample = X_test[idx]
    recon_spec = reconstruct_spectrum(recon_model, phot_sample)

    # Simple numerical comparison
    error = np.linalg.norm(true_spec - recon_spec) / np.linalg.norm(true_spec)
    print(f"Relative reconstruction error for sample {idx}: {error:.4f}")