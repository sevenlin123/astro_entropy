import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----- Spectral model -----
def make_wavelength_grid(start=300, stop=800, n=1000):
    """Create a uniform wavelength grid in nanometers."""
    return np.linspace(start, stop, n)

def gaussian_spectrum(center, width, amplitude, wav):
    """Generate a Gaussian spectral feature."""
    return amplitude * np.exp(-0.5 * ((wav - center) / width) ** 2)

def create_basis_spectra(n_basis, wav):
    """Generate a set of Gaussian basis spectra."""
    rng = np.random.default_rng(seed=42)
    centers = np.linspace(350, 750, n_basis)
    widths  = rng.uniform(20, 40, size=n_basis)
    amplitudes = rng.uniform(0.5, 1.5, size=n_basis)
    basis = np.vstack([gaussian_spectrum(c, w, a, wav)
                       for c, w, a in zip(centers, widths, amplitudes)])
    return basis  # shape (n_basis, len(wav))

# ----- Synthetic spectra generation -----
def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return coeffs @ basis  # shape (len(wav),)

def generate_synthetic_data(n_samples, n_basis, wav):
    """Generate random coefficients and corresponding spectra."""
    rng = np.random.default_rng(seed=123)
    coeffs = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_basis))
    basis = create_basis_spectra(n_basis, wav)
    spectra = np.array([synthesize_spectrum(c, basis) for c in coeffs])
    return coeffs, basis, spectra

# ----- Photometric simulation -----
def make_filter_transmission(wav, center, width):
    """Top‑hat filter centered at `center` nm with full width `width`."""
    return np.where((wav >= center - width/2) & (wav <= center + width/2), 1.0, 0.0)

def create_filters(wav, centers=[400, 500, 600], width=100):
    """Create a set of simple broadband filters."""
    return [make_filter_transmission(wav, c, width) for c in centers]

def compute_photometry(spectrum, filters, wav):
    """Integrate spectrum through each filter."""
    phots = []
    for filt in filters:
        # numerical integration with Simpson's rule
        flux = simps(spectrum * filt, wav)
        phots.append(flux)
    return np.array(phots)

def generate_photometric_dataset(spectra, filters, wav):
    """Compute photometry for each spectrum in the dataset."""
    return np.array([compute_photometry(s, filters, wav) for s in spectra])

# ----- Reconstruction framework -----
def build_design_matrix(basis, filters, wav):
    """Matrix that maps basis coefficients to photometry."""
    n_basis = basis.shape[0]
    design = np.zeros((len(filters), n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            design[i, j] = simps(basis[j] * filt, wav)
    return design.T  # shape (n_basis, n_phot)

def reconstruct_coefficients(photometry, design, alpha=1e-4):
    """Estimate coefficients from photometry via ridge regression."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(design.T, photometry.T)
    return reg.coef_.T  # shape (n_samples, n_basis)

def reconstruct_spectra(coeffs, basis):
    """Produce spectra from estimated coefficients."""
    return np.array([synthesize_spectrum(c, basis) for c in coeffs])

# ----- Main script -----
if __name__ == "__main__":
    # Set up wavelength grid
    wav = make_wavelength_grid()

    # Generate synthetic spectra
    n_samples = 50
    n_basis   = 10
    true_coeffs, basis, spectra = generate_synthetic_data(n_samples, n_basis, wav)

    # Create filters and simulate photometry
    filters = create_filters(wav)
    photometry = generate_photometric_dataset(spectra, filters, wav)

    # Build design matrix for reconstruction
    design = build_design_matrix(basis, filters, wav)

    # Reconstruct spectra from photometry
    est_coeffs = reconstruct_coefficients(photometry, design)
    recon_spectra = reconstruct_spectra(est_coeffs, basis)

    # Evaluate reconstruction quality (RMS error per spectrum)
    rms_errors = np.sqrt(np.mean((spectra - recon_spectra)**2, axis=1))
    print("Mean RMS reconstruction error:", np.mean(rms_errors))
    print("First 5 RMS errors:", rms_errors[:5])