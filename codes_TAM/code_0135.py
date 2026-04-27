import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# ---------- Model definition ----------
def gaussian_basis(wavelengths, center, width):
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def build_basis(wavelengths, n_basis=5):
    rng = np.random.default_rng(42)
    centers = rng.uniform(400, 800, size=n_basis)
    widths = rng.uniform(20, 50, size=n_basis)
    basis = np.vstack([gaussian_basis(wavelengths, c, w) for c, w in zip(centers, widths)]).T
    return basis, centers, widths

# ---------- Synthetic spectrum ----------
def synthesize_spectrum(basis, coeffs):
    return basis @ coeffs

# ---------- Filters ----------
def top_hat_filter(wavelengths, center, width):
    return np.where(np.abs(wavelengths - center) <= width / 2, 1.0, 0.0)

def build_filters(wavelengths, n_filters=4):
    rng = np.random.default_rng(24)
    centers = rng.uniform(400, 800, size=n_filters)
    widths = rng.uniform(40, 80, size=n_filters)
    filters = np.vstack([top_hat_filter(wavelengths, c, w) for c, w in zip(centers, widths)]).T
    return filters, centers, widths

# ---------- Photometry ----------
def compute_photometry(spectrum, filters, wavelengths):
    # simple Riemann sum over wavelengths
    delta = np.gradient(wavelengths)
    return (filters.T @ (spectrum * delta)).ravel()

# ---------- Reconstruction ----------
def reconstruct_spectrum(photometry, filters, basis, wavelengths, alpha=1e-3):
    # Precompute matrix A where A[i,j] = integral T_i * B_j over λ
    delta = np.gradient(wavelengths)
    A = filters @ (basis * delta[:, None])  # shape (n_filters, n_basis)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(A, photometry)
    coeffs = model.coef_
    reconstructed = basis @ coeffs
    return reconstructed, coeffs

# ---------- Main ----------
if __name__ == "__main__":
    wavelengths = np.linspace(400, 800, 400)  # nm

    # Build basis and filters
    basis, b_centers, b_widths = build_basis(wavelengths, n_basis=5)
    filters, f_centers, f_widths = build_filters(wavelengths, n_filters=4)

    # Generate true spectrum
    rng = np.random.default_rng(99)
    true_coeffs = rng.normal(scale=1.0, size=basis.shape[1])
    true_spectrum = synthesize_spectrum(basis, true_coeffs)

    # Generate photometric measurements
    photometry = compute_photometry(true_spectrum, filters, wavelengths)

    # Reconstruct spectrum
    recon_spectrum, recon_coeffs = reconstruct_spectrum(photometry, filters, basis, wavelengths, alpha=1e-2)

    # ---- Output ----
    print("True coefficients:", true_coeffs)
    print("Reconstructed coefficients:", recon_coeffs)
    err = np.linalg.norm(recon_spectrum - true_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative L2 error of reconstructed spectrum: {err:.3f}")