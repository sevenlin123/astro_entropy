import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model – a set of basis spectra
# ----------------------------------------------------------------------
def generate_basis_spectra(num_basis, wavelength):
    """Generate num_basis random basis spectra as smooth functions."""
    rng = np.random.default_rng(42)
    basis = []
    for _ in range(num_basis):
        # create a smooth random curve using Fourier series
        coeffs = rng.normal(size=10)
        spectrum = np.sum(
            [c * np.sin(2 * np.pi * n * wavelength / wavelength.max())
             for n, c in enumerate(coeffs, start=1)],
            axis=0,
        )
        # shift into positive range and normalize
        spectrum = (spectrum - spectrum.min()) / (spectrum.ptp() + 1e-12)
        basis.append(spectrum)
    return np.vstack(basis)  # shape (num_basis, len(wavelength))

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def synthesize_spectrum(basis, coeffs=None):
    """Linear combination of basis spectra."""
    rng = np.random.default_rng()
    if coeffs is None:
        coeffs = rng.uniform(-1, 1, size=basis.shape[0])
    spectrum = np.dot(coeffs, basis)
    return spectrum, coeffs

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------------
def create_filters():
    """Create simple top‑hat filter transmissions."""
    centers = np.array([410, 455, 520, 590, 680])   # nm
    widths  = np.array([40, 40, 60, 60, 70])        # nm
    filters = []
    for c, w in zip(centers, widths):
        lower, upper = c - w/2, c + w/2
        filt = np.where((wavelength >= lower) & (wavelength <= upper), 1.0, 0.0)
        filters.append(filt)
    return filters

def compute_photometry(spectrum, filters, wavelength):
    """Integrate spectrum over each filter band."""
    phots = []
    for filt in filters:
        # weighted average over the band
        num = simps(spectrum * filt, wavelength)
        den = simps(filt, wavelength)
        phots.append(num / (den + 1e-12))
    return np.array(phots)

# ----------------------------------------------------------------------
# 4. Reconstruct a synthetic spectrum from photometry
# ----------------------------------------------------------------------
def build_design_matrix(filters, basis, wavelength):
    """Each row = integral of basis spectrum * filter transmission."""
    mat = np.zeros((len(filters), basis.shape[0]))
    for i, filt in enumerate(filters):
        for j, spec in enumerate(basis):
            mat[i, j] = simps(spec * filt, wavelength) / (simps(filt, wavelength)+1e-12)
    return mat

def reconstruct_spectrum(phots, design_mat, basis):
    """Least‑squares fit of coefficients."""
    lr = LinearRegression(fit_intercept=False).fit(design_mat, phots)
    coeffs = lr.coef_
    recon = np.dot(coeffs, basis)
    return recon, coeffs

# ----------------------------------------------------------------------
# Main routine – demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wavelength = np.linspace(400, 700, 301)  # nm

    # Create basis spectra
    basis = generate_basis_spectra(num_basis=5, wavelength=wavelength)

    # Generate a synthetic spectrum
    true_spectrum, true_coeffs = synthesize_spectrum(basis)

    # Build filters and compute photometry
    filters = create_filters()
    phots = compute_photometry(true_spectrum, filters, wavelength)

    # Construct design matrix and reconstruct spectrum
    design_mat = build_design_matrix(filters, basis, wavelength)
    recon_spectrum, recon_coeffs = reconstruct_spectrum(phots, design_mat, basis)

    # Evaluate reconstruction error
    rmse = np.sqrt(np.mean((true_spectrum - recon_spectrum)**2))
    print("True coefficients :", true_coeffs)
    print("Reconstructed coeffs:", recon_coeffs)
    print("RMSE of spectrum reconstruction:", rmse)