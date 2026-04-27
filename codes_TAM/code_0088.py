import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Spectral model – sum of Gaussian components
# ------------------------------------------------------------------
def gaussian(wave, amp, cen, wid):
    """Single Gaussian component."""
    return amp * np.exp(-0.5 * ((wave - cen) / wid)**2)

def synthetic_spectrum(wave, params):
    """Generate spectrum as sum of Gaussian components.
    
    Parameters
    ----------
    wave : ndarray
        Wavelength array.
    params : list of tuples
        Each tuple is (amplitude, center, width) of a Gaussian.
    """
    spec = np.zeros_like(wave)
    for amp, cen, wid in params:
        spec += gaussian(wave, amp, cen, wid)
    return spec


# ------------------------------------------------------------------
# 2. Photometric filter definition
# ------------------------------------------------------------------
def filter_gaussian(wave, cen, wid):
    """Gaussian bandpass filter transmission."""
    return np.exp(-0.5 * ((wave - cen) / wid)**2)


def generate_filters():
    """Create a set of simple Gaussian filters."""
    # Filter definitions: (name, center [Å], width [Å])
    filt_defs = [
        ("U", 3500., 250.),
        ("B", 4400., 200.),
        ("V", 5500., 150.),
        ("R", 6500., 200.),
    ]
    filters = {}
    for name, cen, wid in filt_defs:
        filters[name] = filter_gaussian(wave_grid, cen, wid)
    return filters


# ------------------------------------------------------------------
# 3. Generate synthetic data
# ------------------------------------------------------------------
wave_grid = np.linspace(3000., 10000., 500)          # Å

# Parameters for synthetic spectrum: (amp, center, width)
true_params = [
    (1.0, 3500., 100.),
    (0.8, 4800., 120.),
    (0.6, 6000., 80.),
    (0.4, 7500., 90.),
    (0.3, 9000., 70.)
]

# True spectrum
true_spec = synthetic_spectrum(wave_grid, true_params)

# Filters
filters = generate_filters()

# Compute synthetic photometry (integrated flux through each filter)
photometric_fluxes = {}
for name, filt in filters.items():
    num   = simps(true_spec * filt, wave_grid)      # numerator
    denom = simps(filt, wave_grid)                  # normalisation
    photometric_fluxes[name] = num / denom

photometric_values = np.array(list(photometric_fluxes.values()))


# ------------------------------------------------------------------
# 4. Reconstruction framework
# ------------------------------------------------------------------
def build_basis_functions(wave, params):
    """Build matrix of basis functions evaluated on the wavelength grid."""
    basis = []
    for amp, cen, wid in params:
        basis.append(gaussian(wave, 1.0, cen, wid))  # amplitude to be fitted
    return np.column_stack(basis)  # shape (N_wave, N_basis)


def build_projection_matrix(filters, basis_funcs):
    """Project basis functions onto filters to form matrix A."""
    A = []
    for filt in filters.values():                 # iterate over filter transmission
        col = []
        for bf in basis_funcs.T:                 # each basis function
            proj = simps(bf * filt, wave_grid)   # integrate product
            col.append(proj)
        A.append(col)
    return np.array(A)                            # shape (N_filters, N_basis)


def reconstruct_spectrum(phot_vals, A, basis_funcs):
    """Reconstruct spectrum from photometric values."""
    # Solve for basis amplitudes with Ridge regression (regularisation λ=1e-3)
    reg = Ridge(alpha=1e-3, fit_intercept=False, normalize=False)
    reg.fit(A, phot_vals)
    coeffs = reg.coef_
    recon_spec = basis_funcs @ coeffs
    return recon_spec, coeffs


# Build basis functions using the true component parameters
basis_funcs = build_basis_functions(wave_grid, true_params)  # shape (N_wave, N_basis)

# Build projection matrix
A = build_projection_matrix(filters, basis_funcs)          # shape (N_filters, N_basis)

# Reconstruct spectrum
recon_spec, fitted_coeffs = reconstruct_spectrum(photometric_values, A, basis_funcs)


# ------------------------------------------------------------------
# 5. Results
# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    plt.plot(wave_grid, true_spec, label='True Spectrum')
    plt.plot(wave_grid, recon_spec, '--', label='Reconstructed Spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (arbitrary units)')
    plt.title('Spectrum Reconstruction from Photometry')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("True parameters:")
    for p in true_params:
        print(f"  {p}")
    print("\nFitted coefficients (amplitudes):")
    for i, amp in enumerate(fitted_coeffs):
        print(f"  Component {i+1}: {amp:.3f}")