import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def gaussian(wl, amp, cen, wid):
    return amp * np.exp(-0.5 * ((wl - cen) / wid) ** 2)

def synthetic_spectrum(wl, params):
    """params: [(amp, cen, wid), ...]"""
    spec = np.random.uniform(0.8, 1.2, size=wl.size)  # flat continuum with noise
    for amp, cen, wid in params:
        spec += gaussian(wl, amp, cen, wid)
    return spec

# ---------- Generate synthetic data ----------
def generate_spectra(n_spectra=50, wl_start=4000, wl_end=7000, wl_step=1):
    wl = np.arange(wl_start, wl_end + wl_step, wl_step)
    spectra = []
    for _ in range(n_spectra):
        n_lines = np.random.randint(1, 5)
        params = []
        for _ in range(n_lines):
            amp = np.random.uniform(0.1, 0.5)
            cen = np.random.uniform(wl_start, wl_end)
            wid = np.random.uniform(5, 30)
            params.append((amp, cen, wid))
        spectra.append(synthetic_spectrum(wl, params))
    return wl, np.array(spectra)

# ---------- Filter definitions ----------
def top_hat_filter(wl, center, width):
    """Simple top-hat filter."""
    trans = np.where(np.abs(wl - center) <= width / 2, 1.0, 0.0)
    return trans

def define_filters():
    """Define a few simple filters."""
    wl = np.arange(4000, 7001, 1)
    filters = {
        'B': top_hat_filter(wl, 4400, 100),
        'V': top_hat_filter(wl, 5500, 80),
        'R': top_hat_filter(wl, 6500, 120),
    }
    return wl, filters

# ---------- Compute photometric fluxes ----------
def compute_photometry(spectra, wl, filters):
    phot = []
    for spec in spectra:
        mags = []
        for filt in filters.values():
            # simple integral of spec * filter (unit area)
            flux = np.sum(spec * filt) / np.sum(filt)
            mags.append(flux)
        phot.append(mags)
    return np.array(phot)

# ---------- Reconstruction ----------
def train_reconstruction(phot, spectra):
    """Linear regression from photometry to spectrum."""
    lr = LinearRegression()
    lr.fit(phot, spectra)
    return lr

def reconstruct_spectrum(lr, phot_vec):
    """Predict full spectrum from photometric vector."""
    return lr.predict(phot_vec.reshape(1, -1))[0]

# ---------- Main ----------
if __name__ == "__main__":
    np.random.seed(42)

    # Generate training spectra
    wl, spectra = generate_spectra(n_spectra=100)

    # Define filters and compute photometry
    wl_flt, filters = define_filters()
    phot = compute_photometry(spectra, wl_flt, filters)

    # Train reconstruction model
    lr = train_reconstruction(phot, spectra)

    # Create a test spectrum and its photometry
    test_params = [(0.3, 4500, 20), (0.2, 5900, 15)]
    test_spec = synthetic_spectrum(wl, test_params)
    test_phot = np.array([np.sum(test_spec * filt) / np.sum(filt) for filt in filters.values()])

    # Reconstruct
    recon_spec = reconstruct_spectrum(lr, test_phot)

    # Simple sanity check
    print("True vs reconstructed flux difference (rms):",
          np.sqrt(np.mean((test_spec - recon_spec)**2)))