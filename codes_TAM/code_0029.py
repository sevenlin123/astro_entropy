import numpy as np
from scipy.integrate import simps
from scipy.optimize import least_squares
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# --------------------
# Spectral model
# --------------------
def spectral_model(params, wavelengths):
    """Simple Gaussian spectral model."""
    A, mu, sigma = params
    return A * np.exp(-((wavelengths - mu) ** 2) / (2 * sigma ** 2))

# --------------------
# Generate synthetic spectra
# --------------------
def sample_params():
    """Randomly sample model parameters."""
    A = np.random.uniform(1.0, 5.0)
    mu = np.random.uniform(400.0, 700.0)
    sigma = np.random.uniform(20.0, 60.0)
    return np.array([A, mu, sigma])

def generate_synthetic_spectra(n_samples, wavelengths):
    """Generate synthetic spectra and true parameters."""
    spectra = []
    params_list = []
    for _ in range(n_samples):
        params = sample_params()
        spec = spectral_model(params, wavelengths)
        spectra.append(spec)
        params_list.append(params)
    return np.array(spectra), np.array(params_list)

# --------------------
# Generate photometric filters
# --------------------
def generate_filters(n_filters, wavelengths):
    """Create Gaussian filters."""
    centers = np.linspace(350, 750, n_filters)
    widths = np.full(n_filters, 50.0)  # fixed width
    filters = np.exp(-((wavelengths[:, None] - centers[None, :]) ** 2) /
                     (2 * widths[None, :] ** 2))
    return filters, centers, widths

# --------------------
# Compute photometry
# --------------------
def compute_photometry(spectra, filters):
    """Integrate spectra through filters."""
    photometry = []
    for spec in spectra:
        flux = []
        for filt in filters.T:
            num = simps(spec * filt, x=wavelengths)
            den = simps(filt, x=wavelengths)
            flux.append(num / den)
        photometry.append(flux)
    return np.array(photometry)

# --------------------
# Reconstruction
# --------------------
def reconstruct_params(observed_flux, filters, wavelengths):
    """Recover model parameters by least-squares fitting."""
    def residuals(params):
        model_flux = []
        spec = spectral_model(params, wavelengths)
        for filt in filters.T:
            num = simps(spec * filt, x=wavelengths)
            den = simps(filt, x=wavelengths)
            model_flux.append(num / den)
        return np.array(model_flux) - observed_flux

    x0 = np.array([3.0, 550.0, 40.0])  # initial guess
    result = least_squares(residuals, x0, bounds=([0, 300, 10], [10, 800, 100]))
    return result.x

# --------------------
# Main routine
# --------------------
np.random.seed(42)
wavelengths = np.linspace(300.0, 800.0, 1000)

# Generate data
n_samples = 10
spectra, true_params = generate_synthetic_spectra(n_samples, wavelengths)

# Filters
filters, centers, widths = generate_filters(5, wavelengths)

# Photometry
photometry = compute_photometry(spectra, filters)

# Reconstruction
reconstructed_params = []
reconstructed_spectra = []

for obs_flux in photometry:
    params_rec = reconstruct_params(obs_flux, filters, wavelengths)
    reconstructed_params.append(params_rec)
    rec_spec = spectral_model(params_rec, wavelengths)
    reconstructed_spectra.append(rec_spec)

reconstructed_params = np.array(reconstructed_params)
reconstructed_spectra = np.array(reconstructed_spectra)

# Example output
idx = 0
print("True params:", true_params[idx])
print("Reconstructed params:", reconstructed_params[idx])
print("First 5 wavelengths of true spectrum:", wavelengths[:5])
print("First 5 wavelengths of reconstructed spectrum:", reconstructed_spectra[idx][:5])

# Optional: use sklearn for demonstration (not critical for reconstruction)
pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
X = true_params
y = spectr