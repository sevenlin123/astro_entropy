#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def spectral_basis(wavelength):
    """Three Gaussian basis functions."""
    mu = np.array([4000, 5500, 7000])   # Angstrom
    sigma = np.array([300, 400, 200])
    return np.exp(-0.5 * ((wavelength[:, None] - mu) / sigma)**2)

def synthetic_spectrum(coeffs, wavelength):
    """Generate a spectrum as a linear combination of basis functions."""
    basis = spectral_basis(wavelength)
    return basis @ coeffs

# ---------- Photometry ----------
def filter_transmission(center, width, wavelength):
    """Gaussian transmission curve."""
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def photometric_flux(spectrum, wavelength, centers, widths):
    """Integrate spectrum over each filter."""
    fluxes = []
    for c, w in zip(centers, widths):
        trans = filter_transmission(c, w, wavelength)
        flux = simps(spectrum * trans, wavelength) / simps(trans, wavelength)
        fluxes.append(flux)
    return np.array(fluxes)

# ---------- Data generation ----------
def generate_dataset(n_samples, n_wav=1000):
    """Create training data."""
    # Wavelength grid
    wav = np.linspace(3000, 8000, n_wav)
    # Random coefficients (normal distribution)
    coeffs = np.random.randn(n_samples, 3)
    # Spectra
    spectra = np.array([synthetic_spectrum(c, wav) for c in coeffs])
    # Filters
    filt_centers = np.array([3500, 5000, 6500, 8000])
    filt_widths  = np.array([200, 300, 250, 150])
    # Photometric measurements
    phots = np.array([photometric_flux(s, wav, filt_centers, filt_widths)
                      for s in spectra])
    return wav, coeffs, spectra, phots, filt_centers, filt_widths

# ---------- Reconstruction ----------
class SpectrumReconstructor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, phots, coeffs):
        self.model.fit(phots, coeffs)

    def predict_coeffs(self, phots):
        return self.model.predict(phots)

    def reconstruct(self, coeffs_pred, wavelength):
        return np.array([synthetic_spectrum(c, wavelength) for c in coeffs_pred])

# ---------- Demo ----------
if __name__ == "__main__":
    # Generate training data
    wav, coeffs_train, spec_train, phots_train, cent, wid = generate_dataset(200)

    # Fit reconstructor
    recon = SpectrumReconstructor()
    recon.fit(phots_train, coeffs_train)

    # Generate test sample
    coeffs_test = np.random.randn(5, 3)
    spectra_test = np.array([synthetic_spectrum(c, wav) for c in coeffs_test])
    phots_test = np.array([photometric_flux(s, wav, cent, wid) for s in spectra_test])

    # Predict and reconstruct
    coeffs_pred = recon.predict_coeffs(phots_test)
    spectra_recon = recon.reconstruct(coeffs_pred, wav)

    # Evaluate mean squared error
    mse = np.mean((spectra_test - spectra_recon)**2)
    print(f"Reconstruction MSE: {mse:.4e}")

    # Print first test spectrum comparison
    import matplotlib.pyplot as plt
    idx = 0
    plt.plot(wav, spectra_test[idx], label='True')
    plt.plot(wav, spectra_recon[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()