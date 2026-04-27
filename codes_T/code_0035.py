import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Define a spectral model (planck function with parameters T, F)
# ------------------------------------------------------------------
def planck_lambda(wl_nm, T, F):
    """
    Planck's law in wavelength units (nm).  
    wl_nm: wavelength array (nm)
    T   : temperature (K)
    F   : scaling factor (arbitrary)
    """
    h = 6.62607015e-34      # J*s
    c = 2.99792458e8        # m/s
    k_B = 1.380649e-23      # J/K

    wl_m = wl_nm * 1e-9     # nm -> m
    B = (2*h*c**2) / (wl_m**5) / (np.exp(h*c/(wl_m*k_B*T)) - 1.0)
    return F * B

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, wl_grid, T_range=(2000,4000), F_range=(0.5,2.0)):
    """
    Produce n_spectra synthetic spectra using random T and F.
    """
    rng = np.random.default_rng()
    spectra = []
    params = []
    for _ in range(n_synth):
        T = rng.uniform(*T_range)
        F = rng.uniform(*F_range)
        spec = planck_lambda(wl_grid, T, *[F])
        spectra.append(spec)
        params.append((T,F))
    return np.array(spectra), np.array(params)

# -----------------------------------------------------------------
# 3. create  *reconstruct*  photo data from &t=1? 
? (??... )

...