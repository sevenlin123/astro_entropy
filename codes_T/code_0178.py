#!/usr/bin/env python3
import numpy as np
from scipy.signal import gaussian
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model utilities
# ----------------------------------------------------------------------
def wavelength_grid(n=1000, lam_min=400.0, lam_max=800.0):
    """Return an evenly spaced wavelength array (nm)."""
    return np.linspace(lam_min, lam_max, n)


def gaussian_template(wavelength, amp, cen, wid):
    """Gaussian spectral template."""
    return amp * np.exp(-0.5 * ((wavelength - cen) / wid) ** 2)


def construct_templates(wavelength):
    """Build a set of spectral templates."""
    templates = [
        gaussian_template(wavelength, 1.0, 450.0, 20.0),
        gaussian_template(wavelength, 0.8, 550.0, 30.0),
        gaussian_template(wavelength, 0.6, 650.0, 25.0),
    ]
    return np.vstack(templates)  # shape (n_templates, n_wavelengths)


def spectral_model(coeffs, templates):
    """Linear combination of templates."""
    return coeffs @ templates


# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, templates, noise_level=0.02):
    """Generate synthetic spectra by random mixing of templates."""
    n_templates, n_wave = templates.shape
    coeffs = np.random.uniform(0.5, 1.5, size=(n_spectra, n_templates))
    spectra = coeffs @ templates.T
    spectra += noise_level * np.std(spectra, axis=1, keepdims=True) * np.random.randn(*spectra.shape)
    return spectra, coeffs


# ----------------------------------------------------------------------
# Photometric system
# ----------------------------------------------------------------------
def make_filter(wavelength, center, width, shape='gaussian'):
    """Create a simple filter transmission curve."""
    if shape == 'gaussian':
        return gaussian(len(wavelength), std=width, msize=len(wavelength)) * \
               np.exp(-0.5 * ((wavelength - center) / width) ** 2)
    else:
        raise ValueError('Unsupported filter shape')


def generate_filters(wavelength):
    """Define a few synthetic broadband filters."""
    filt_defs = [
        ('u', 350.0, 50.0),
        ('g', 475.0, 70.0),
        ('r', 620.0, 80.0),
        ('i', 760.0, 90.0),
    ]
    filters = {}
    for name, cen, wid in filt_defs:
        filters[name] = make_filter(wavelength, cen, wid)
    return filters


def photometry_from_spectrum(spectrum, wavelength, filters):
    """Integrate spectrum through each filter to obtain fluxes."""
    fluxes = {}
    for name, trans in filters.items():
        flux = np.trapz(spectrum * trans, wavelength) / np.trapz(trans, wavelength)
        fluxes[name] = flux
    return fluxes


def spectra_to_photometry(spectra, wavelength, filters):
    """Vectorised photometry extraction."""
    n_spectra = spectra.shape[0]
    flux_matrix = np.zeros((n_spectra, len(filters)))
    for i, (name, trans) in enumerate(filters.items()):
        numer = np.trapz(spectra.T * trans, wavelength, axis=0)
        denom = np.trapz(trans, wavelength)
        flux_matrix[:, i] = numer / denom
    return flux_matrix


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def build_pca_basis(spectra, n_components=5):
    """Fit PCA to synthetic spectra and return the fitted model."""
    pca = PCA(n_components=n_components, svd_solver='randomized')
    pca.fit(spectra)
    return pca


def filter_projection_matrix(pca, wavelength, filters):
    """Project PCA components onto filter space."""
    comps = pca.components_  # shape (n_components, n_wavelengths)
    n_filters = len(filters)
    proj = np.zeros((n_filters, pca.n_components))
    for i, (name, trans) in enumerate(filters.items()):
        denom = np.trapz(trans, wavelength)
        proj[i] = np.trapz(comps * trans, wavelength, axis=1) / denom
    return proj


def reconstruct_from_photometry(flux_obs, proj_matrix, pca, mean_spectrum):
    """Solve for PCA coefficients given observed photometry."""
    # Least-squares fit: flux_obs ≈ proj_matrix @ coeffs
    lr = LinearRegression(fit_intercept=False)
    lr.fit(proj_matrix.T, flux_obs)
    coeffs = lr.predict(proj_matrix.T)
    # Reconstruct spectrum
    reconstructed = coeffs @ pca.components_ + mean_spectrum
    return reconstructed


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    np.random.seed(42)

    # 1. Define wavelength grid and templates
    wave = wavelength_grid()
    templates = construct_templates(wave)

    # 2. Generate synthetic spectra
    n_spectra = 200
    spectra, true_coeffs = generate_synthetic_spectra(n_spectra, templates)

    # 3. Define filters and extract photometry
    filters = generate_filters(wave)
    fluxes = spectra_to_photometry(spectra, wave, filters)

    # 4. Fit PCA basis to spectra
    pca = build_pca_basis(spectra, n_components=5)

    # 5. Compute projection matrix of PCA components onto filters
    proj_matrix = filter_projection_matrix(pca, wave, filters)

    # 6. Reconstruct each spectrum from its photometry
    reconstructions = []
    for flux in fluxes:
        recon = reconstruct_from_photometry(
            flux, proj_matrix, pca, pca.mean_
        )
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)

    # 7. Evaluate reconstruction accuracy
    rms_error = np.sqrt(np.mean((spectra - reconstructions) ** 2))
    print(f"Reconstruction RMS error over {n_spectra} spectra: {rms_error:.4f}")

    # 8. Demo: plot one example (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(wave, spectra[idx], label='True spectrum')
        plt.plot(wave, reconstructions[idx], '--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (arb. units)')
        plt.title('Spectrum reconstruction example')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()