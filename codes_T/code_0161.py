import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def build_basis(wavelengths, n_basis=5):
    """Create a set of Gaussian basis spectra."""
    centers = np.linspace(400, 700, n_basis)
    widths = 30.0
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c)/widths)**2)
        basis.append(g)
    return np.array(basis)          # shape (n_basis, N_wave)

def generate_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return coeffs @ basis            # shape (N_wave,)

# ---------- Photometric model ----------
def gaussian_filter(wave, center, width):
    return np.exp(-0.5 * ((wave - center)/width)**2)

def make_filters(wavelengths):
    """Create four simple Gaussian filters."""
    filt_centers = [450, 550, 650, 700]
    filt_width = 20.0
    filters = []
    for c in filt_centers:
        filters.append(gaussian_filter(wavelengths, c, filt_width))
    return np.array(filters)         # shape (n_filters, N_wave)

def compute_photometry(spectrum, wavelengths, filters):
    """Integrate spectrum over each filter."""
    fluxes = []
    for f in filters:
        flux = trapz(spectrum * f, wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)          # shape (n_filters,)

# ---------- Reconstruction ----------
def train_regressor(X_train, y_train):
    """Fit a linear model from photometry to basis coefficients."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)
    return reg

def reconstruct_spectrum(reg, photometry, basis, wavelengths):
    """Predict basis coefficients then rebuild spectrum."""
    coeffs = reg.predict(photometry.reshape(1, -1))[0]
    return generate_spectrum(basis, coeffs)

# ---------- Example usage ----------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(400, 700, 300)   # nm

    # Build basis
    basis = build_basis(wav, n_basis=5)  # shape (5, 300)

    # Generate synthetic training data
    n_train = 50
    X_train = []   # photometry
    y_train = []   # coefficients
    for _ in range(n_train):
        coeffs = np.random.rand(5)       # random coefficients
        spec = generate_spectrum(basis, coeffs)
        filt = make_filters(wav)
        phot = compute_photometry(spec, wav, filt)
        X_train.append(phot)
        y_train.append(coeffs)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train regressor
    reg = train_regressor(X_train, y_train)

    # Generate a new synthetic star
    true_coeffs = np.random.rand(5)
    true_spec = generate_spectrum(basis, true_coeffs)
    filt = make_filters(wav)
    phot = compute_photometry(true_spec, wav, filt)

    # Reconstruct spectrum
    recon_spec = reconstruct_spectrum(reg, phot, basis, wav)

    # Simple check: compare true and reconstructed spectra
    err = np.linalg.norm(true_spec - recon_spec) / np.linalg.norm(true_spec)
    print(f"Relative reconstruction error: {err:.3f}")