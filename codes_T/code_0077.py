import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge


# ---------- Spectral model ----------
def gaussian_template(wl, amp, cen, wid):
    """Single Gaussian template."""
    return amp * np.exp(-0.5 * ((wl - cen) / wid) ** 2)


def build_basis_templates(wl, n_tmpl=10, seed=0):
    """Generate a set of Gaussian basis templates."""
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.5, 1.5, n_tmpl)
    cents = rng.uniform(wl.min(), wl.max(), n_tmpl)
    wids = rng.uniform((wl.max() - wl.min())/20,
                       (wl.max() - wl.min())/10,
                       n_tmpl)
    basis = [gaussian_template(wl, a, c, w) for a, c, w in zip(amps, cents, wids)]
    return np.vstack(basis)  # shape (n_tmpl, len(wl))


# ---------- Synthetic spectra ----------
def synthesize_spectra(basis, n_spec=5, coeff_range=(0.1, 2.0), seed=1):
    """Generate synthetic spectra as linear combos of basis templates."""
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(coeff_range[0], coeff_range[1], size=(n_spec, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_spec, len(wl))
    return spectra, coeffs


# ---------- Photometric filters ----------
def gaussian_filter(wl, cen, wid):
    """Gaussian photometric filter curve."""
    return np.exp(-0.5 * ((wl - cen) / wid) ** 2)


def build_filters(wl, n_filt=4, seed=2):
    """Generate simple Gaussian filters."""
    rng = np.random.default_rng(seed)
    cents = rng.uniform(wl.min(), wl.max(), n_filt)
    wids = rng.uniform((wl.max() - wl.min())/15,
                       (wl.max() - wl.min())/8,
                       n_filt)
    filt_curves = [gaussian_filter(wl, c, w) for c, w in zip(cents, wids)]
    return np.vstack(filt_curves)  # shape (n_filt, len(wl))


# ---------- Photometry ----------
def compute_photometry(spectra, filters, wl):
    """Integrate spectra over filters to get broadband fluxes."""
    # Assume unit spectral units; flux ∝ integral S(λ)*T(λ)dλ
    fluxes = []
    for spec in spectra:
        f = [simps(spec * filt, wl) for filt in filters]
        fluxes.append(f)
    return np.array(fluxes)  # shape (n_spec, n_filt)


# ---------- Reconstruction ----------
def reconstruct_spectra(photon_fluxes, filters, basis, wl, alpha=1.0):
    """
    Reconstruct spectra from broadband fluxes via ridge regression.
    Returns estimated spectra and fitted coefficients.
    """
    # Build design matrix: filter integrals over basis templates
    G = np.zeros((filters.shape[0], basis.shape[0]))
    for i, filt in enumerate(filters):
        G[i, :] = [simps(tmpl * filt, wl) for tmpl in basis]
    # Fit coefficients: solve (G^T G + alpha I) c = G^T fluxes.T
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(G.T, photon_fluxes.T)
    coeffs_est = reg.coef_.T  # shape (n_spec, n_tmpl)
    spectra_rec = coeffs_est @ basis  # shape (n_spec, len(wl))
    return spectra_rec, coeffs_est


# ---------- Example usage ----------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(3000, 10000, 2000)  # Ångstroms

    # Build basis and filters
    basis = build_basis_templates(wl, n_tmpl=10, seed=42)
    filters = build_filters(wl, n_filt=4, seed=24)

    # Generate synthetic spectra and photometry
    spectra_true, coeff_true = synthesize_spectra(basis, n_spec=5, seed=84)
    photometry = compute_photometry(spectra_true, filters, wl)

    # Reconstruct spectra from photometry
    spectra_rec, coeff_est = reconstruct_spectra(photometry, filters, basis, wl, alpha=0.01)

    # Simple diagnostics
    for i in range(5):
        err = np.linalg.norm(spectra_true[i] - spectra_rec[i]) / np.linalg.norm(spectra_true[i])
        print(f"Spectrum {i}: relative L2 error = {err:.3f}")