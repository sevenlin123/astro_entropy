import numpy as np
from scipy.interpolate import interp1d

def create_wavelength_grid(start=0.4, end=2.5, n=500):
    """Generate a wavelength grid in microns."""
    return np.linspace(start, end, n)

def create_basis_spectra(wave, n_basis=5, rng=np.random.default_rng(0)):
    """Create a set of Gaussian basis spectra."""
    bases = []
    centers = rng.uniform(0.5, 2.0, size=n_basis)
    widths  = rng.uniform(0.05, 0.15, size=n_basis)
    amps    = rng.uniform(0.8, 1.2, size=n_basis)
    for c,w,a in zip(centers,widths,amps):
        spec = a*np.exp(-0.5*((wave-c)/w)**2)
        bases.append(spec)
    return np.array(bases)  # shape (n_basis, n_wave)

def generate_true_spectrum(basis, rng=np.random.default_rng(1)):
    """Generate a synthetic spectrum as a linear combo of basis spectra."""
    coeffs = rng.uniform(0.5, 1.5, size=basis.shape[0])
    true   = coeffs @ basis
    return true, coeffs

def create_bandpasses(wave, n_bands=4, rng=np.random.default_rng(2)):
    """Create Gaussian bandpasses."""
    bandpasses = []
    centers = rng.uniform(0.5, 2.0, size=n_bands)
    widths  = rng.uniform(0.05, 0.15, size=n_bands)
    for c,w in zip(centers,widths):
        bp = np.exp(-0.5*((wave-c)/w)**2)
        bp /= np.trapz(bp, wave)  # normalize to unit area
        bandpasses.append(bp)
    return np.array(bandpasses)  # shape (n_bands, n_wave)

def compute_photometry(spectrum, bandpasses, wave):
    """Integrate spectrum over each bandpass."""
    return np.array([np.trapz(spectrum*bp, wave) for bp in bandpasses])

def reconstruct_spectrum(photometry, bandpasses, basis, wave):
    """Reconstruct spectrum coefficients via least‑squares."""
    # Build matrix P_ij = integral(B_i * BP_j)
    P = np.array([[np.trapz(b*bp, wave) for bp in bandpasses] for b in basis])
    # Solve P.T @ c = photometry  ->  c = (P^T P)^(-1) P^T photometry
    coeffs, *_ = np.linalg.lstsq(P.T, photometry, rcond=None)
    recon_spec = coeffs @ basis
    return recon_spec, coeffs

def main():
    wave = create_wavelength_grid()
    basis = create_basis_spectra(wave)
    true_spec, true_coeffs = generate_true_spectrum(basis)
    bandpasses = create_bandpasses(wave)
    phot = compute_photometry(true_spec, bandpasses, wave)
    recon_spec, recon_coeffs = reconstruct_spectrum(phot, bandpasses, basis, wave)
    print("True coefficients:", true_coeffs)
    print("Recovered coefficients:", recon_coeffs)
    print("Mean absolute error in spectrum:", np.mean(np.abs(recon_spec-true_spec)))

if __name__ == "__main__":
    main()