import numpy as np
from scipy.constants import h, c, k
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model: simple black‑body with optional metallicity scaling
# ----------------------------------------------------------------------
def blackbody_flux(wave, temp, scale=1.0):
    """
    Planck function in W sr^-1 m^-2 μm^-1.
    wave : array of wavelengths in microns
    temp : temperature in K
    scale: multiplicative scaling factor
    """
    wave_m = wave * 1e-6  # convert to meters
    exponent = h * c / (wave_m * k * temp)
    intensity = (2.0 * h * c**2) / (wave_m**5) / (np.exp(exponent) - 1.0)
    return scale * intensity

def spectral_model(wave, temp, logg, feh):
    """
    Simple composite model: blackbody * metallicity factor
    """
    base = blackbody_flux(wave, temp)
    # metallicity scaling: approximate as 10^(0.4*feh)
    metal_factor = 10 ** (0.4 * feh)
    return base * metal_factor

# ----------------------------------------------------------------------
# Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, wave_grid):
    temps = np.random.uniform(4000, 8000, n_spectra)
    loggs = np.random.uniform(3.5, 5.0, n_spectra)
    feas = np.random.uniform(-1.0, 0.5, n_spectra)

    spectra = np.zeros((n_spectra, len(wave_grid)))
    for i in range(n_spectra):
        spectra[i] = spectral_model(wave_grid, temps[i], loggs[i], feas[i])
    params = np.column_stack([temps, loggs, feas])
    return spectra, params

# ----------------------------------------------------------------------
# Bandpass definitions (UBVRI approximated by Gaussians)
# ----------------------------------------------------------------------
def gaussian_bandpass(center, width, wave_grid):
    return np.exp(-0.5 * ((wave_grid - center)/width)**2)

bandpasses = {
    'U': gaussian_bandpass(0.36, 0.05, None),
    'B': gaussian_bandpass(0.44, 0.05, None),
    'V': gaussian_bandpass(0.55, 0.05, None),
    'R': gaussian_bandpass(0.64, 0.05, None),
    'I': gaussian_bandpass(0.79, 0.05, None),
}

# ----------------------------------------------------------------------
# Photometric synthesis
# ----------------------------------------------------------------------
def generate_photometry(spectra, wave_grid, bandpasses):
    """
    Compute synthetic broadband fluxes for each spectrum.
    """
    phot = []
    for bp_name, resp in bandpasses.items():
        # ensure response defined over wave_grid
        if resp is None:
            raise ValueError("Bandpass response must be defined")
        # normalize bandpass
        norm = np.trapz(resp, wave_grid)
        flux = np.trapz(spectra * resp[np.newaxis, :], wave_grid, axis=1) / norm
        phot.append(flux)
    return np.column_stack(phot)

# ----------------------------------------------------------------------
# Reconstruction pipeline
# ----------------------------------------------------------------------
def train_reconstruction(spectra_train, phot_train, n_components=5):
    """
    Train PCA on spectra and linear regression to predict PCA
    coefficients from photometry.
    """
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    coeffs_train = pca.fit_transform(spectra_train)
    reg = LinearRegression()
    reg.fit(phot_train, coeffs_train)
    return pca, reg

def reconstruct_spectrum(pca, reg, phot_obs):
    """
    Predict PCA coefficients from photometry and reconstruct spectrum.
    """
    coeffs_pred = reg.predict(phot_obs.reshape(1, -1))
    spec_rec = pca.inverse_transform(coeffs_pred)
    return spec_rec[0]

# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
def main():
    # wavelength grid: 0.3 – 1.0 micron, 1000 points
    wave_grid = np.linspace(0.3, 1.0, 1000)

    # generate synthetic dataset
    n_samples = 200
    spectra, params = generate_synthetic_spectra(n_samples, wave_grid)
    phot = generate_photometry(spectra, wave_grid, bandpasses)

    # split into training and test sets
    n_train = int(0.8 * n_samples)
    spectra_train = spectra[:n_train]
    phot_train = phot[:n_train]
    spectra_test  = spectra[n_train:]
    phot_test     = phot[n_train:]

    # train reconstruction model
    pca, reg = train_reconstruction(spectra_train, phot_train, n_components=20)

    # reconstruct spectra for test set
    spectra_rec = []
    for phot_obs in phot_test:
        spec_rec = reconstruct_spectrum(pca, reg, phot_obs)
        spectra_rec.append(spec_rec)
    spectra_rec = np.array(spectra_rec)

    # evaluate reconstruction quality (mean squared error)
    mse = np.mean((spectra_test - spectra_rec)**2)
    print(f"Reconstruction MSE: {mse:.3e}")

if __name__ == "__main__":
    main()