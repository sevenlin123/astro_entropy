import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------- 1. Spectral model ----------
def spectral_model(params, wavelengths):
    """
    Simple synthetic spectral model: a Gaussian line plus a continuum offset.
    params : [center, width, offset]
    wavelengths : array of wavelength values
    """
    center, width, offset = params
    gaussian = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return gaussian + offset


# ---------- 2. Generate synthetic spectra ----------
def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Generate n_samples synthetic spectra with random parameters.
    Returns:
        spectra : shape (n_samples, len(wavelengths))
        true_params : shape (n_samples, 3)
    """
    rng = np.random.default_rng(seed=42)
    centers = rng.uniform(500, 600, size=n_samples)      # nm
    widths = rng.uniform(10, 20, size=n_samples)         # nm
    offsets = rng.uniform(0.1, 0.3, size=n_samples)      # arbitrary units
    true_params = np.vstack([centers, widths, offsets]).T
    spectra = np.array([spectral_model(p, wavelengths) for p in true_params])
    return spectra, true_params


# ---------- 3. Generate photometric data ----------
def filter_response(wavelengths, center, width):
    """Top‑hat filter centred at `center` with half‑width `width`."""
    return ((wavelengths >= center - width) & (wavelengths <= center + width)).astype(float)


def photometric_flux(spectrum, wavelengths, filter_centers, filter_width):
    """
    Compute synthetic photometric fluxes by integrating the spectrum over filters.
    """
    fluxes = []
    for c in filter_centers:
        filt = filter_response(wavelengths, c, filter_width)
        flux = np.trapz(spectrum * filt, wavelengths) / np.sum(filt)
        fluxes.append(flux)
    return np.array(fluxes)


def generate_photometric_data(spectra, wavelengths, filter_centers, filter_width):
    """Apply all filters to all spectra."""
    return np.array([
        photometric_flux(spec, wavelengths, filter_centers, filter_width)
        for spec in spectra
    ])


# ---------- 4. Reconstruct spectra from photometry ----------
def reconstruct_spectra(photon_data, spectra_shape, wavelengths, filter_centers, filter_width):
    """
    Train a linear regression model to map photometric fluxes to full spectra.
    Returns predicted spectra for the provided photometric data.
    """
    # Fit on synthetic training data
    reg = LinearRegression()
    reg.fit(photon_data, spectra_shape)

    # Predict
    return reg.predict(photon_data)


# ---------- Main routine ----------
def main():
    # Define wavelength grid
    wavelengths = np.linspace(400, 700, 300)  # nm

    # Generate synthetic data
    n_samples = 200
    spectra, _ = generate_synthetic_spectra(n_samples, wavelengths)

    # Photometric filter definition
    filter_centers = np.arange(450, 650, 40)  # nm
    filter_width = 30  # nm

    photometry = generate_photometric_data(
        spectra,
        wavelengths,
        filter_centers,
        filter_width
    )

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        photometry,
        spectra,
        test_size=0.2,
        random_state=0
    )

    # Train regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predict spectra for test set
    spectra_pred = reg.predict(X_test)

    # Evaluate reconstruction error (root mean square)
    rmse = np.sqrt(np.mean((y_test - spectra_pred) ** 2))
    print(f"Reconstruction RMSE per wavelength: {rmse:.4f}")

if __name__ == "__main__":
    main()