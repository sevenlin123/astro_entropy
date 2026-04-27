import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wave, coeffs):
    """Simple linear combination of basis spectra."""
    return coeffs @ wave.T

def generate_basis(num_basis, wave):
    """Generate random basis spectra."""
    return np.random.randn(num_basis, len(wave))

def synth_spectrum(coeffs, basis):
    """Create synthetic spectrum from given coefficients."""
    return coeffs @ basis

def photometry_from_spectrum(spectrum, filt_wave, filt_trans):
    """
    Compute synthetic photometric magnitudes in a set of filters.
    The magnitude is defined as -2.5*log10(∫ S(λ) T(λ) dλ) + ZP.
    """
    flux = spectrum * filt_trans
    integral = np.trapz(flux, filt_wave)
    zp = 0.0
    return -2.5 * np.log10(integral) + zp

def generate_synthetic_data():
    # wavelength grid
    wave = np.linspace(500, 2500, 2000)
    num_basis = 5
    basis = generate_basis(num_basis, wave)
    # random true coefficients
    true_coeffs = np.random.randn(num_basis)
    # synthetic spectrum
    spec = synth_spectrum(true_coeffs, basis)
    # define filters
    filt_wav = np.array([600, 900, 1200, 1500, 1800])   # central wavelengths
    filt_width = 100
    fns = []
    for cen in filt_wav:
        mask = (wave >= cen-filt_width/2) & (wave <= cen+filt_width/2)
        trans = np.zeros_like(wave)
        trans[mask] = 1.0
        return (wave, filt_wavelengths, int=???)#...