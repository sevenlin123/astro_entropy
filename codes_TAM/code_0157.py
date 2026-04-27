import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# Spectral model: sum of Gaussians
# -------------------------------------------------------------
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def generate_synthetic_spectrum(wav, n_components=3):
    """Generate one synthetic spectrum as sum of n Gaussian components."""
    amps = np.random.uniform(0.5, 1.5, size=n_components)
    cents = np.random.uniform(wav.min() + 20, wav.max() - 20, size=n_components)
    wids = np.random.uniform(5, 15, size=n_components)
    spec = np.zeros_like(wav)
    for a, c, w in zip(amps, cents, wids):
        spec += gaussian(wav, a, c, w)
    return spec

# -------------------------------------------------------------
# Filter generation
# -------------------------------------------------------------
def create_gaussian_filters(n_filters, wav):
    """Create a list of Gaussian filter transmissions."""
    centers = np.linspace(wav[0] + 50, wav[-1] - 50, n_filters)
    widths = np.full(n_filters, 30.0)
    filters = []
    for cen, wid in zip(centers, widths):
        filt = gaussian(wav, 1.0, cen, wid)
        filters.append(filt / filt.sum())  # normalize
    return np.array(filters)

# -------------------------------------------------------------
# Photometry simulation
# -------------------------------------------------------------
def compute_flux(spectrum, filt):
    """Integrate spectrum times filter transmission."""
    return np.trapz(spectrum * filt, axis=-1)

def generate_dataset(n_samples, wav, filters):
    spectra = []
    photometry = []
    for _ in range(n_samples):
        spec = generate_synthetic_spectrum(wav)
        spectra.append(spec)
        fluxes = [compute_flux(spec, f) for f in filters]
        photometry.append(fluxes)
    return np.array(spectra), np.array(photometry)

# -------------------------------------------------------------
# Reconstruction model
# -------------------------------------------------------------
def train_reconstruction_model(photometry, spectra):
    """Linear regression mapping photometry → full spectrum."""
    reg = LinearRegression()
    reg.fit(photometry, spectra)
    return reg

def reconstruct_spectrum(model, photometry_vector):
    return model.predict(photometry_vector.reshape(1, -1))[0]

# -------------------------------------------------------------
# Demo
# -------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(400, 800, 500)  # nm

    # Filters
    n_filters = 4
    filters = create_gaussian_filters(n_filters, wav)

    # Training data
    n_train = 200
    train_specs, train_phot = generate_dataset(n_train, wav, filters)

    # Test data
    n_test = 5
    test_specs, test_phot = generate_dataset(n_test, wav, filters)

    # Train model
    model = train_reconstruction_model(train_phot, train_specs)

    # Reconstruct test spectra
    recon_specs = []
    for p in test_phot:
        recon_specs.append(reconstruct_spectrum(model, p))
    recon_specs = np.array(recon_specs)

    # Simple evaluation: mean absolute error
    mae = np.mean(np.abs(recon_specs - test_specs))
    print(f"Mean absolute error on test set: {mae:.4f}")