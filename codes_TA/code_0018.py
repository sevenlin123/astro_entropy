import numpy as np
from scipy.constants import h, c, k
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Spectral model – black‑body spectrum
# ----------------------------------------------------------------------
def blackbody_flux(wl, T):
    """
    Planck’s law for a black–body.
    wl : array of wavelengths in meters
    T  : temperature in Kelvin
    Returns spectral radiance in W·m⁻²·sr⁻¹·m⁻¹
    """
    exponent = h * c / (wl * k * T)
    return (2.0 * h * c**2) / (wl**5 * (np.exp(exponent) - 1.0))

# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samp, wl):
    """
    Generates n_samp synthetic spectra with random temperatures.
    Returns:
        spectra : shape (n_samp, len(wl))
        params  : dict of parameters (here just temperature)
    """
    temps = np.random.uniform(3500, 10000, size=n_samp)   # K
    spectra = np.array([blackbody_flux(wl, t) for t in temps])
    return spectra, {"T": temps}

# ----------------------------------------------------------------------
# 3. Filter definitions (Gaussian filters)
# ----------------------------------------------------------------------
def gaussian_filter(wl, center, width):
    """Return normalized Gaussian filter transmission."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

def define_filters():
    """Return a dictionary of filter transmissions."""
    # Wavelengths in microns for convenience
    wl_micron = np.arange(0.35, 0.9, 0.001)  # 350–900 nm
    wl = wl_micron * 1e-6                    # convert to meters
    filters = {
        "U": gaussian_filter(wl, 365e-9, 30e-9),
        "B": gaussian_filter(wl, 445e-9, 40e-9),
        "V": gaussian_filter(wl, 551e-9, 50e-9),
        "R": gaussian_filter(wl, 658e-9, 60e-9),
        "I": gaussian_filter(wl, 806e-9, 70e-9)
    }
    return wl, filters

# ----------------------------------------------------------------------
# 4. Compute photometry
# ----------------------------------------------------------------------
def compute_photometry(spectra, wl, filters):
    """
    Integrate each spectrum over each filter.
    Returns an array of shape (n_samp, n_filters)
    """
    fluxes = []
    for name, trans in filters.items():
        # Truncate to match wavelength grid
        integ = spectra * trans
        flux = np.trapz(integ, wl, axis=1)
        fluxes.append(flux)
    return np.column_stack(fluxes)

# ----------------------------------------------------------------------
# 5. Spectrum reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra(X_train, y_train, X_new):
    """
    Train a multi‑output linear regression to predict spectrum from photometry.
    X_train : (n_train, n_filters)
    y_train : (n_train, n_wl)
    X_new   : (n_test, n_filters)
    Returns predicted spectra: (n_test, n_wl)
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)
    return reg.predict(X_new)

# ----------------------------------------------------------------------
# 6. Main routine
# ----------------------------------------------------------------------
def main():
    # Define wavelength grid (400–700 nm)
    wl = np.linspace(400e-9, 700e-9, 301)          # 301 points

    # Generate synthetic spectra
    n_samples = 500
    spectra, params = generate_synthetic_spectra(n_samples, wl)

    # Define filters and compute photometry
    wl_filt, filters = define_filters()
    photometry = compute_photometry(spectra, wl_filt, filters)

    # Split into training / test
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=42
    )

    # Reconstruct spectra from photometry
    recon_test = reconstruct_spectra(X_train, y_train, X_test)

    # Evaluate reconstruction quality
    mae = np.mean(np.abs(recon_test - y_test))
    print(f"Mean absolute error of reconstruction: {mae:.3e} W·m⁻²·sr⁻¹·m⁻¹")

if __name__ == "__main__":
    main()