import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# Wavelength grid
lam = np.linspace(3000, 10000, 2000)          # Å

# Basis functions – Gaussian lines
def gaussian(lam, cen, wid):
    return np.exp(-0.5 * ((lam - cen) / wid) ** 2)

def make_basis(n=10):
    centers = np.linspace(3500, 9500, n)
    widths  = np.full(n, 50.)
    return np.array([gaussian(lam, c, w) for c, w in zip(centers, widths)])   # shape (n, len(lam))

basis = make_basis()

# Synthetic spectrum – linear combo of basis
def synth_spec(coeffs):
    return coeffs @ basis

# Filter responses – simple top‑hat
filters = {
    'U': lambda l: ((l >= 3200) & (l <= 3800)).astype(float),
    'B': lambda l: ((l >= 3800) & (l <= 4400)).astype(float),
    'V': lambda l: ((l >= 4400) & (l <= 5200)).astype(float),
    'R': lambda l: ((l >= 6200) & (l <= 7500)).astype(float),
    'I': lambda l: ((l >= 7500) & (l <= 9000)).astype(float),
}

# Convert spectrum to photometry
def phot_from_spec(spec, filt_dict):
    out = []
    for f in filt_dict.values():
        resp = f(lam)
        out.append(simps(spec * resp, lam))
    return np.array(out)          # shape (n_filters,)

# Reconstruct spectrum from photometry
def recon_from_phot(phot, filt_dict, basis):
    n_filt, n_basis = len(filt_dict), basis.shape[0]
    A = np.empty((n_filt, n_basis))
    for i, f in enumerate(filt_dict.values()):
        resp = f(lam)
        A[i] = [simps(b * resp, lam) for b in basis]
    coeff, *_ = np.linalg.lstsq(A, phot, rcond=None)
    return coeff @ basis

# Demo
if __name__ == "__main__":
    np.random.seed(42)
    true_coeffs = np.random.rand(basis.shape[0])
    true_spec   = synth_spec(true_coeffs)
    phot_meas   = phot_from_spec(true_spec, filters)

    recon_spec  = recon_from_phot(phot_meas, filters, basis)

    rmse = np.sqrt(np.mean((true_spec - recon_spec) ** 2))
    print(f"RMSE between true and reconstructed spectrum: {rmse:.4f}")