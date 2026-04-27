import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

def wavelength_grid(start=300, stop=800, num=1000):
    return np.linspace(start, stop, num)

def gaussian_line(wl, amp, cen, width):
    return amp * np.exp(-(wl - cen)**2 / (2 * width**2))

def synthetic_spectrum(wl, amps, censes, widths):
    spec = np.zeros_like(wl)
    for a, c, w in zip(amps, censes, widths):
        spec += gaussian_line(wl, a, c, w)
    return spec

def top_hat_filter(wl, center, width):
    return np.logical_and(wl >= center - width/2,
                          wl <= center + width/2).astype(float)

def filter_responses(wl):
    centers = [350, 450, 550, 650]
    widths  = [50, 50, 50, 50]
    return [top_hat_filter(wl, c, w) for c,w in zip(centers,widths)]

def measure_fluxes(spec, wl, filters):
    fluxes = []
    for f in filters:
        # integrate spec * filter response over wavelength
        flux = simps(spec * f, wl)
        fluxes.append(flux)
    return np.array(fluxes)

def construct_basis(wl, n_basis=50):
    # Chebyshev polynomials of first kind
    x = (2*(wl - wl.min())/(wl.max()-wl.min())) - 1
    basis = np.vstack([np.cos(i*np.arccos(x)) for i in range(n_basis)]).T
    return basis

def reconstruct_spectrum(fluxes, filters, wl, n_basis=50):
    # Build design matrix: for each filter integrate basis over filter
    basis = construct_basis(wl, n_basis)
    X = np.array([simps(basis * f[:,None], wl) for f in filters]).T
    # Solve linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, fluxes)
    coeffs = reg.coef_
    recon_spec = basis @ coeffs
    return recon_spec, coeffs

def main():
    wl = wavelength_grid()
    n_samp = 5
    recon_error = []

    for _ in range(n_samp):
        amps   = np.random.uniform(0.5, 1.5, size=3)
        censes = np.random.uniform(350, 650, size=3)
        widths = np.random.uniform(10, 30, size=3)

        true_spec = synthetic_spectrum(wl, amps, censes, widths)

        filters = filter_responses(wl)
        fluxes  = measure_fluxes(true_spec, wl, filters)

        recon_spec, coeffs = reconstruct_spectrum(fluxes, filters, wl)

        err = np.linalg.norm(recon_spec - true_spec) / np.linalg.norm(true_spec)
        recon_error.append(err)

    print("Reconstruction errors:", recon_error)

if __name__ == "__main__":
    main()