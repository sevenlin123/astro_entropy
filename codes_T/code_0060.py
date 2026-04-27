import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def generate_spectrum(wavelengths, params):
    """
    Simple Gaussian spectral model.
    :param wavelengths: 1D array of wavelengths.
    :param params: list of tuples [(amplitude, center, width), ...]
    :return: flux array corresponding to wavelengths.
    """
    flux = np.zeros_like(wavelengths)
    for amp, cen, wid in params:
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2)
    return flux

# ---------- Synthetic data ----------
np.random.seed(0)

# wavelength grid for generating synthetic spectra
lam_grid = np.linspace(4000, 8000, 1000)  # Angstroms

# Define true parameters for two spectra
true_params_1 = [(1.0, 5000, 200), (0.5, 6000, 150)]
true_params_2 = [(0.8, 5500, 180), (0.6, 6500, 130)]

spectra_true = np.vstack([
    generate_spectrum(lam_grid, true_params_1),
    generate_spectrum(lam_grid, true_params_2)
])

# Add noise
noise_level = 0.05
spectra_noise = spectra_true + noise_level * np.random.randn(*spectra_true.shape)

# ---------- Photometry ----------
# Define simple bandpasses (center, width)
band_centers = np.array([4500, 5200, 5800, 6400, 7000])  # Å
band_widths = np.array([400, 400, 400, 400, 400])       # Å

def get_band_fluxes(spectrum, lam, centers, widths):
    """Integrate spectrum over top-hat bandpasses."""
    fluxes = []
    for c, w in zip(centers, widths):
        mask = (lam >= c - w/2) & (lam <= c + w/2)
        flux = np.trapz(spectrum[mask], lam[mask]) / (w)
        fluxes = np.append(flux????).????
Wait we must output final code. Let's finish properly.assistantfinal codeassistantfinal code