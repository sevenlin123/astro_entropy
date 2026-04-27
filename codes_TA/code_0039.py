import numpy as np
from numpy.linalg import lstsq

# ------------------------------
# Spectral model and utilities
# ------------------------------

def create_wavelength_grid(start=300.0, stop=800.0, num=1000):
    """Create an evenly spaced wavelength grid in nm."""
    return np.linspace(start, stop, num)

def spectral_basis_functions(wavelengths):
    """Return a list of basis functions evaluated on the wavelength grid.
    Here we use three simple basis functions."""
    b1 = np.ones_like(wavelengths)                        # constant
    b2 = wavelengths / wavelengths[-1]                   # linear rise
    b3 = np.sin(2 * np.pi * wavelengths / wavelengths[-1])  # sinusoidal
    return [b1, b2, b3]

def generate_random_coefficients(num_samples, num_basis):
    """Generate random coefficients for synthetic spectra."""
    return np.random.randn(num_samples, num_basis)

def synthesize_spectra(coeffs, basis_funcs, wavelengths):
    """Construct spectra as linear combinations of basis functions."""
    spectra = np.zeros((coeffs.shape[0], len(wavelengths)))
    for i, bf in enumerate(basis_funcs):
        spectra += coeffs[:, i][:, None] * bf
    return spectra

# ------------------------------
# Photometry generation
# ------------------------------

def gaussian_filter(wavelengths, center, width):
    """Normalized Gaussian filter response."""
    resp = np.exp(-0.5 * ((wavelengths - center) / width)**2)
    return resp / np.trapz(resp, wavelengths)  # normalize area to 1

def compute_photometry(spectrum, filters, wavelengths):
    """Integrate spectrum with each filter to obtain photometric fluxes."""
    fluxes = []
    for filt in filters:
        flux = np.trapz(spectrum * filt, wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)

# ------------------------------
# Spectrum reconstruction
# ------------------------------

def build_filter_matrix(filters, basis_funcs, wavelengths):
    """Matrix of integrals of each basis function through each filter."""
    num_filters = len(filters)
    num_basis = len(basis_funcs)
    A = np.zeros((num_filters, num_basis))
    for i, filt in enumerate(filters):
        for j, bf in enumerate(basis_funcs):
            A[i, j] = np.trapz(bf * filt, wavelengths)
    return A

def reconstruct_spectrum(photometry, filter_matrix, basis_funcs, wavelengths):
    """Estimate coefficients that best reproduce the photometry."""
    coeffs, *_ = lstsq(filter_matrix, photometry, rcond=None)
    reconstructed = synthesize_spectra(coeffs[np.newaxis, :], basis_funcs, wavelengths)[0]
    return reconstructed, coeffs

# ------------------------------
# Example workflow
# ------------------------------

def main():
    # Wavelength grid
    wav = create_wavelength_grid()

    # Basis functions
    basis = spectral_basis_functions(wav)

    # Synthetic spectra
    num_samples = 5
    coeffs_true = generate_random_coefficients(num_samples, len(basis))
    spectra_true = synthesize_spectra(coeffs_true, basis, wav)

    # Define 3 filters
    centers = [350, 500, 650]   # nm
    widths  = [30, 40, 30]      # nm
    filters = [gaussian_filter(wav, c, w) for c, w in zip(centers, widths)]

    # Generate photometry for each synthetic spectrum
    photometries = np.array([compute_photometry(spec, filters, wav) for spec in spectra_true])

    # Build filter matrix once (same for all samples)
    A = build_filter_matrix(filters, basis, wav)

    # Reconstruct each spectrum and compare
    for idx, (phot, true_spec) in enumerate(zip(photometries, spectra_true)):
        recon_spec, coeff_est = reconstruct_spectrum(phot, A, basis, wav)
        print(f"\nSample {idx+1}")
        print(f"True coefficients:     {coeffs_true[idx]}")
        print(f"Estimated coefficients:{coeff_est}")
        mse = np.mean((true_spec - recon_spec)**2)
        print(f"Reconstruction MSE:    {mse:.4e}")

if __name__ == "__main__":
    main()