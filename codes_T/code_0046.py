import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----- Spectral model ---------------------------------------------------------
def generate_spectrum(wl, amps, centers, sigma):
    """Generate a synthetic spectrum as a sum of Gaussians."""
    spectrum = np.zeros_like(wl)
    for a, mu in zip(amps, centers):
        spectrum += a * np.exp(-0.5 * ((wl - mu) / sigma)**2)
    return spectrum

# ----- Filter generation -----------------------------------------------------
def generate_filters(wl, centers, widths):
    """Return a list of Gaussian filter transmission curves."""
    filters = []
    for mu, w in zip(centers, widths):
        filt = np.exp(-0.5 * ((wl - mu) / w)**2)
        filters.append(filt)
    return filters

# ----- Photometric calculation -----------------------------------------------
def compute_photometry(spectrum, wl, filters):
    """Integrate spectrum through each filter to obtain photometric fluxes."""
    fluxes = []
    for filt in filters:
        numerator = simps(spectrum * filt, wl)
        denominator = simps(filt, wl)
        fluxes.append(numerator / denominator)
    return np.array(fluxes)

# ----- Reconstruction --------------------------------------------------------
def reconstruct_amps(fluxes, wl, basis_centers, sigma, filter_centers, filter_widths):
    """
    Reconstruct Gaussian amplitudes from photometric fluxes.
    Solve linear system A * amps = fluxes where A_ij = <basis_j | filter_i>.
    """
    # Build design matrix
    n_filters = len(filter_centers)
    n_basis = len(basis_centers)
    A = np.zeros((n_filters, n_basis))
    for i, (mu_f, w_f) in enumerate(zip(filter_centers, filter_widths)):
        filt = np.exp(-0.5 * ((wl - mu_f) / w_f)**2)
        denom = simps(filt, wl)
        for j, mu_b in enumerate(basis_centers):
            basis = np.exp(-0.5 * ((wl - mu_b) / sigma)**2)
            numer = simps(basis * filt, wl)
            A[i, j] = numer / denom
    # Least-squares solution
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, fluxes)
    return lr.coef_

# ----- Main routine ----------------------------------------------------------
def main():
    np.random.seed(0)
    # Wavelength grid (nm)
    wl = np.linspace(400, 800, 400)
    # Basis Gaussian parameters
    n_basis = 5
    centers = np.linspace(450, 750, n_basis)
    sigma = 10.0
    # Generate true amplitudes
    true_amps = np.random.uniform(0.5, 1.5, size=n_basis)
    # Synthetic spectrum
    spec_true = generate_spectrum(wl, true_amps, centers, sigma)
    # Filters
    n_filters = 4
    filter_centers = np.linspace(470, 730, n_filters)
    filter_widths = np.full(n_filters, 30.0)
    filters = generate_filters(wl, filter_centers, filter_widths)
    # Photometry
    fluxes = compute_photometry(spec_true, wl, filters)
    # Reconstruction
    recovered_amps = reconstruct_amps(fluxes, wl, centers, sigma,
                                      filter_centers, filter_widths)
    # Compare
    print("True amplitudes :", true_amps)
    print("Recovered amps :", recovered_amps)
    err = np.linalg.norm(true_amps - recovered_amps) / np.linalg.norm(true_amps)
    print(f"Relative L2 error: {err:.4f}")

if __name__ == "__main__":
    main()