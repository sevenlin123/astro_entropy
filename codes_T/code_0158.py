import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of basis spectra.
    :param wavelengths: 1D array of wavelengths.
    :param coeffs: 1D array of coefficients for each basis.
    :return: model spectrum at given wavelengths.
    """
    # Define three arbitrary basis spectra
    basis1 = np.exp(-(wavelengths - 400)**2 / (2 * 30**2))
    basis2 = np.sin(0.02 * wavelengths) * np.exp(-((wavelengths - 550)**2) / (2 * 50**2))
    basis3 = 0.5 + 0.5 * np.cos(100 / wavelengths)
    return coeffs[0] * basis1 + coeffs[1] * basis2 + coeffs[2] * basis3

def generate_synthetic_spectra(num_spectra, wavelengths):
    """Generate a set of synthetic spectra using random coefficients."""
    np.random.seed(42)
    coeffs_list = np.random.rand(num_spectra, 3)  # random coefficients
    spectra = np.array([spectral_model(wavelengths, coeffs) for coeffs in 
                       coeffs_list])
    return spectra, coeffs_list

def photometry_from_spectrum(spectrum, wavelengths, filter_wavels):
    """
    calculate photometric fluxes from a single spectrum.
    :param i..??!..??..???????...???????????......????
We need to produce a full code that does: define spectral model, generate synthetic spectra, generate photometric data from them, reconstruct synthetic spectrum from photometry. Use numpy, scipy, sklearn. Provide clear functions.

We need to implement photometry_from_spectrum: given a spectrum and wavelengths, and filter wavelengths or bandpasses. We can create simple filters: Gaussian filters centered at certain wavelengths with some widths. Then compute integrated flux in each band by weighting the spectrum by filter transmission and integrating over wavelength. Then reconstruct the spectrum: we could use linear regression on basis spectra: given photometric fluxes, we want to estimate coefficients that best fit those fluxes using known filter responses applied to basis spectra. That is: we know for each filter j, the flux from a basis k is integral of basis_k * filter_j. So we can build design matrix A of shape (num_filters, num_bases). Given measured flux vector f, solve for coefficients via ridge regression: coeffs_est = Ridge().fit(A, f). Then reconstruct spectrum: spectral_model(wavelengths, coeffs_est). That will give reconstructed spectrum.

Implementation steps:

- define wavelengths grid: e.g., 300-800 nm with step 1 nm.
- define basis spectra as above.
- define filter functions: maybe 4 Gaussian filters at wavelengths 350, 450, 550, 650 nm with width 20 nm.
- photometry function: compute integrated flux: sum(spectrum * filter * dw) maybe normalized.
- generate synthetic spectra: choose random coeffs per spectrum; generate spectra.
- generate photometry: for each synthetic spectrum, compute fluxes through filters.
- For reconstruction: given photometry for a spectrum, estimate coefficients via ridge regression.
- Evaluate reconstruction: compare reconstructed vs true spectrum.

Also we can test by reconstructing all synthetic spectra and printing relative error.

Implement functions:

- filter_response(wavelengths, center, width): returns Gaussian.

- photometry_from_spectrum(spectrum,wavelengths,filters): returns flux vector.

- reconstruct_spectrum_from_photometry(fluxes, wavelengths, filters): perform ridge regression using precomputed basis filtered integrals.

Implement main block to run test.

Let's write code. Use numpy, scipy, sklearn. Provide reproducible results.

We need to compute A: matrix of shape (n_filters, n_bases), where each entry A[j,k] = integral(basis_k * filter_j). We'll compute this once.

Let's implement. Use np.trapz for integration. Use small smoothing.

Now produce code. Let's craft.

Check dependencies: numpy, scipy (interpolate if needed), sklearn.linear_model Ridge.

We also need to import sklearn; ensure it's installed.

We'll provide clear functions.

Let's implement.

Also we may want to show errors but not required.

Ok let's produce final code.assistantfinal code