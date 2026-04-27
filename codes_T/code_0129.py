import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------
# Spectral model (basis functions)
# ----------------------------------
def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def make_basis_functions(wavelength):
    """Return an array of basis functions evaluated on the grid."""
    # constant, linear, quadratic, two Gaussians
    basis = []
    basis.append(np.ones_like(wavelength))                    # constant
    basis.append(wavelength - wavelength.mean())             # linear
    basis.append((wavelength - wavelength.mean())**2)        # quadratic
    basis.append(gaussian(wavelength, 460, 30))              # gaussian 1
    basis.append(gaussian(wavelength, 580, 40))              # gaussian 2
    return np.vstack(basis)                                  # shape (5, N)

# ----------------------------------
# Photometric filters
# ----------------------------------
def make_filters(wavelength):
    """Return a list of simple top‑hat filter responses."""
    filters = []
    # filter 1: 400–500 nm
    f1 = np.logical_and(wavelength >= 400, wavelength <= 500).astype(float)
    # filter 2: 500–600 nm
    f2 = np.logical_and(wavelength >= 500, wavelength <= 600).astype(float)
    # filter 3: 600–700 nm
    f3 = np.logical_and(wavelength >= 600, wavelength <= 700).astype(float)
    filters.extend([f1, f2, f3])
    return np.array(filters)                                # shape (3, N)

# ----------------------------------
# Synthetic data generation
# ----------------------------------
def generate_spectra(num_samples, basis):
    """Generate synthetic spectra as random linear combinations of basis."""
    coeffs = np.random.randn(num_samples, basis.shape[0])   # (S, B)
    spectra = coeffs @ basis.T                              # (S, N)
    return spectra, coeffs

def compute_photometry(spectra, filters, wavelength):
    """Integrate each spectrum through each filter."""
    dw = np.diff(wavelength).mean()
    phot = spectra @ (filters * dw).T                       # (S, F)
    return phot

# ----------------------------------
# Reconstruction
# ----------------------------------
def reconstruct_spectra(photometry, filters, basis, wavelength):
    """Recover spectra from photometric fluxes."""
    # Build design matrix: integral of each basis over each filter
    dw = np.diff(wavelength).mean()
    M = np.zeros((filters.shape[0], basis.shape[0]))        # (F, B)
    for i in range(filters.shape[0]):
        for j in range(basis.shape[0]):
            M[i, j] = np.sum(basis[j] * filters[i]) * dw   # scalar
    # Solve least squares for each sample
    coeffs_rec = np.linalg.lstsq(M, photometry.T, rcond=None)[0].T   # (S, B)
    # Reconstruct spectra
    spectra_rec = coeffs_rec @ basis.T                         # (S, N)
    return spectra_rec, coeffs_rec

# ----------------------------------
# Main routine
# ----------------------------------
def main():
    # Wavelength grid
    wl = np.linspace(350, 750, 400)          # 350–750 nm

    # Basis functions and filters
    basis = make_basis_functions(wl)         # (B, N)
    filters = make_filters(wl)               # (F, N)

    # Generate synthetic data
    n_samples = 20
    true_spectra, true_coeffs = generate_spectra(n_samples, basis)

    # Photometry
    phot = compute_photometry(true_spectra, filters, wl)

    # Reconstruction
    rec_spectra, rec_coeffs = reconstruct_spectra(phot, filters, basis, wl)

    # Evaluation
    mae = np.abs(rec_spectra - true_spectra).mean()
    print(f"Mean absolute error of reconstructed spectra: {mae:.4e}")

    # Show first spectrum comparison
    idx = 0
    print("\nFirst spectrum:")
    print("True coefficients:", true_coeffs[idx])
    print("Recovered coeffs :", rec_coeffs[idx])
    print("\nWavelength | True | Reconstructed")
    for w, t, r in zip(wl[:10], true_spectra[idx][:10], rec_spectra[idx][:10]):
        print(f"{w:6.1f} | {t: .3f} | {r: .3f}")

if __name__ == "__main__":
    main()