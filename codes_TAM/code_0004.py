import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- spectral model ----------
def gaussian(x, amp, cen, wid):
    """Simple Gaussian profile."""
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

def synth_spectrum(wl, amps, cents, wids):
    """Generate a synthetic spectrum as a sum of Gaussians."""
    spec = np.zeros_like(wl)
    for amp, cen, wid in zip(amps, cents, wids):
        spec += gaussian(wl, amp, cen, wid)
    return spec

# ---------- filter curves ----------
def filter_curve(wl, low, high):
    """Box filter from low to high nm."""
    return np.where((wl >= low) & (wl <= high), 1.0, 0.0)

# ---------- photometry ----------
def photometric_flux(spec, wl, filt):
    """Integrate spectrum over a filter curve."""
    return simps(spec * filt, wl) / simps(filt, wl)

# ---------- synthetic data generation ----------
def generate_synthetic_set(n_samples=5, seed=42):
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 700, 301)  # wavelength grid (nm)
    
    # Fixed line parameters
    cents = np.array([450.0, 600.0])   # centers
    wids  = np.array([8.0, 10.0])     # widths
    
    # Filters: U (400-500), B (500-600), V (600-700)
    filt_names = ['U', 'B', 'V']
    filt_ranges = [(400, 500), (500, 600), (600, 700)]
    filters = {name: filter_curve(wl, lo, hi) for name, (lo, hi) in zip(filt_names, filt_ranges)}
    
    spectra = []
    phot = []
    true_coeffs = []
    for _ in range(n_samples):
        amps = rng.uniform(0.5, 1.5, size=2)  # random amplitudes
        spec = synth_spectrum(wl, amps, cents, wids)
        spectra.append(spec)
        true_coeffs.append(amps)
        phots = {name: photometric_flux(spec, wl, filt) for name, filt in filters.items()}
        phot.append(phots)
    return wl, cents, wids, spectra, phot, true_coeffs, filt_names, filters

# ---------- reconstruction ----------
def compute_design_matrix(wl, cents, wids, filt_names, filters):
    """
    Pre‑compute the integrals of each Gaussian line through each filter.
    Returns matrix A of shape (n_filters, n_lines).
    """
    A = np.zeros((len(filt_names), len(cents)))
    for i, (cen, wid) in enumerate(zip(cents, wids)):
        gauss = gaussian(wl, 1.0, cen, wid)  # unit amplitude
        for j, name in enumerate(filt_names):
            filt = filters[name]
            A[j, i] = simps(gauss * filt, wl) / simps(filt, wl)
    return A

def reconstruct_spectra(wl, cents, wids, phot_list, filt_names, filters, A):
    """Reconstruct spectra from photometry."""
    recon_specs = []
    coeffs_list = []
    for phots in phot_list:
        # vector of observed fluxes
        y = np.array([phots[name] for name in filt_names])
        # Least‑squares fit for amplitudes
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        coeffs_list.append(coeffs)
        recon_spec = synth_spectrum(wl, coeffs, cents, wids)
        recon_specs.append(recon_spec)
    return recon_specs, coeffs_list

# ---------- demo ----------
if __name__ == "__main__":
    # Generate data
    wl, cents, wids, true_specs, phot_list, true_coeffs, filt_names, filters = generate_synthetic_set()
    
    # Design matrix for reconstruction
    A = compute_design_matrix(wl, cents, wids, filt_names, filters)
    
    # Reconstruct
    recon_specs, recon_coeffs = reconstruct_spectra(
        wl, cents, wids, phot_list, filt_names, filters, A
    )
    
    # Evaluate
    for i, (true, recon) in enumerate(zip(true_specs, recon_specs)):
        error = np.mean((true - recon)**2)
        print(f"Sample {i+1} MSE: {error:.4f}")