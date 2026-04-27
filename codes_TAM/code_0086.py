import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# -----------------------------
# Helper functions
# -----------------------------

def wavelength_grid(start=400, stop=700, num=1000):
    """Generate a wavelength array in nm."""
    return np.linspace(start, stop, num)

def gaussian(x, mu, sigma):
    """Simple 1‑D Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def basis_functions(wave):
    """
    Return an array of basis functions evaluated at wave.
    Here we use 5 simple analytical functions.
    """
    N = len(wave)
    B = np.zeros((N, 5))
    B[:, 0] = np.ones(N)                     # constant
    B[:, 1] = np.sin(2 * np.pi * wave / 100)  # sinusoid
    B[:, 2] = np.cos(2 * np.pi * wave / 150)  # cosine
    B[:, 3] = gaussian(wave, 500, 30)        # Gaussian peak
    B[:, 4] = gaussian(wave, 600, 40)        # Second Gaussian
    return B

def filter_responses(wave):
    """
    Create a dictionary of filter transmission curves.
    Returns a dict of filter name -> transmission array.
    """
    filt = {}
    filt['U'] = gaussian(wave, 365, 35)
    filt['B'] = gaussian(wave, 445, 35)
    filt['V'] = gaussian(wave, 550, 30)
    filt['R'] = gaussian(wave, 640, 35)
    return filt

# -----------------------------
# Synthetic data generation
# -----------------------------

def synthetic_spectrum(B, coeffs):
    """Generate a spectrum as a linear combination of basis functions."""
    return B @ coeffs

def photometric_flux(spectrum, wave, filters):
    """Compute synthetic photometry by integrating spectrum over each filter."""
    flux = {}
    for name, trans in filters.items():
        # Simple weighted integral; normalize by filter width
        numerator = simps(spectrum * trans, wave)
        denominator = simps(trans, wave)
        flux[name] = numerator / denominator
    return flux

# -----------------------------
# Reconstruction
# -----------------------------

def build_design_matrix(filters, B):
    """
    Build the design matrix A where each row corresponds to a filter,
    and columns correspond to the integral of each basis function through that filter.
    """
    wave = B[:, 0]  # first column contains wavelength values
    A = []
    for trans in filters.values():
        row = []
        for col in range(B.shape[1]):
            integ = simps(B[:, col] * trans, wave)
            norm = simps(trans, wave)
            row.append(integ / norm)
        A.append(row)
    return np.array(A)

def reconstruct_spectrum(flux_dict, filters, B, alpha=1.0):
    """
    Reconstruct the spectrum from photometric measurements using Ridge regression.
    """
    # Build design matrix
    A = build_design_matrix(filters, B)
    # Prepare target vector
    y = np.array([flux_dict[name] for name in filters.keys()])
    # Fit ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, y)
    coeffs = ridge.coef_
    # Return reconstructed spectrum
    return B @ coeffs, coeffs

# -----------------------------
# Main demonstration
# -----------------------------

if __name__ == "__main__":
    # Wavelength grid
    wave = wavelength_grid()

    # Basis functions
    B = basis_functions(wave)

    # True coefficients (random but fixed)
    np.random.seed(42)
    true_coeffs = np.array([10, 3, 2, 5, 4])

    # Generate synthetic spectrum
    spec_true = synthetic_spectrum(B, true_coeffs)

    # Filter responses
    filt = filter_responses(wave)

    # Generate synthetic photometry
    phot = photometric_flux(spec_true, wave, filt)

    # Reconstruct spectrum
    spec_rec, rec_coeffs = reconstruct_spectrum(phot, filt, B, alpha=0.5)

    # Simple comparison prints
    print("True coefficients :", true_coeffs)
    print("Reconstructed coefficients :", rec_coeffs)

    # Compare spectra
    rms_error = np.sqrt(np.mean((spec_true - spec_rec)**2))
    print(f"Reconstruction RMS error: {rms_error:.3f}")