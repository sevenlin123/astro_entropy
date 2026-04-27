
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- need plotting
- generate synthetic example input data inside the script
- Do NOT show reasoning or thinking.
- Do NOT explain the code.
- Return only the Python code.

The code should:

1. define a spectral model
2. generate synthetic spectra
2. generate photometric data from synthetic spectra
3. reconstruct a synthetic spectrum from photometruic

Constraints:

- use numpy, scipy, and scikit-learn
- produce runnable Python code
- define clear functions

Return only the code.
Okay, so I need to write a Python script that implements an algorithm for spectral reconstruction. The user has given specific requirements, so I'll go through each step carefully.

First, the script needs to define a spectral model. I'll assume a simple linear combination of basis functions, like Gaussian curves, since they are easy to handle and commonly used in spectral modeling.

Next, I have to generate synthetic spectra. I'll create a set of basis functions, maybe 5 Gaussians with different means and covariances. Then, I'll randomly assign coefficients to combine these basis functions into a synthetic spectrum. This will serve as my true spectrum.

Then, I need to generate photometric data from these synthetic spectra. Photometric data is typically measured at specific wavelengths. I'll select a few wavelengths (like 5 points) and calculate the flux at each by integrating the spectrum over small wavelength bins around those points. This simulates how instruments measure flux at specific bands.

After that, the main task is to reconstruct the synthetic spectrum from the photometric data. For this, I'll set up an optimization problem. Using least squares, I'll find the coefficients that best fit the photometric data when combined with the basis functions. Scipy's minimize function can handle this, using a regularizer to prevent overfitting by adding a penalty on the coefficients.

Once the reconstruction is done, I'll need to plot both the original and reconstructed spectra, along with the photometric points. This involves plotting the original spectrum, the reconstructed one, and the flux measurements at their respective wavelengths.

I must remember to structure the code with clear functions: one for generating the synthetic spectra, another for generating photometric data, and another for the reconstruction. Also, I should use numpy, scipy, and scikit-learn as specified, even though scikit-learn isn't strictly necessary here but might be included for any potential machine learning approaches.

Let me outline the steps in code:

1. Import necessary libraries: numpy, scipy, and sklearn.
2. Define the basis functions, say 5 Gaussians with varying means and covariances.
3. Create a function to generate synthetic spectra by combining these basis functions with random coefficients.
4. Another function to generate photometric data by integrating the spectrum at selected wavelengths.
5. Set up the optimization problem using scipy's minimize with a least squares objective and a L2 regularizer.
6. After solving, plot the results showing original, reconstructed spectra, and photometric points.

I need to ensure that the code is self-contained, generates all necessary data within the script, and doesn't require external libraries beyond numpy, scipy, and scikit-learn. Also, include plotting with matplotlib to visualize the results.

Wait, in the photometric data generation, I should simulate the process where each photometric band measures flux in a certain wavelength range. So, I'll choose a few wavelengths, say [400, 500, 600, 700, 800] nm, and for each, integrate the synthetic spectrum over a small interval around that point.

For the reconstruction, the model matrix is created by evaluating each basis function at all wavelengths used in photometry. Then, the optimization minimizes the difference between observed photometric fluxes and the model predictions, with regularization to keep coefficients small.

Finally, plotting will show the true spectrum, the reconstructed one, and the photometric points, ensuring clarity and proper labeling.

I think I've covered all steps. Now, putting it all together in code, making sure each function is defined clearly, and the script runs without errors.
</think>

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

def basis_functions(x, params):
    return np.exp(-((x[:, None] - params[0]) ** 2) / (2 * params[1]))

def generate_synthetic_spectra(n_spectra=100, n_wavelengths=1000, n_basis=5):
    wavelengths = np.linspace(300, 900, n_wavelengths)
    basis_params = np.array([[400, 50], [500, 70], [600, 100], [700, 50], [800, 30]])
    coeffs = np.random.randn(n_spectra, n_basis)
    spectra = np.zeros((n_spectra, n_wavelengths))
    for i in range(n_spectra):
        for j in range(n_basis):
            spectra[i] += coeffs[i, j] * basis_functions(wavelengths, basis_params[j])
    return spectra, basis_params

def generate_photometric_data(spectra, wavelengths, n_photometrics=5):
    phot_wavelengths = np.linspace(300, 900, n_photometrics)
    flux = np.zeros((spectra.shape[0], n_photometrics))
    for i in range(n_photometrics):
        w = phot_wavelengths[i]
        idx = np.argwhere(wavelengths >= w-20)[0]
        flux[:, i] = np.trapz(spectra[:, idx:idx+20], wavelengths[idx:idx+20])
    return flux, phot_wavelengths

def spectral_reconstruction(flux, phot_wavelengths, wavelengths, basis_params):
    basis_matrix = np.zeros((len(phot_wavelengths), len(basis_params)))
    for i, w in enumerate(phot_wavelengths):
        idx = np.argwhere(wavelengths >= w-20)[0]
        x = wavelengths[idx:idx+20]
        for j in range(len(basis_params)):
            basis_matrix[i, j] = np.trapz(basis_functions(x, basis_params[j]), x)
    alpha = 0.1
    clf = Ridge(alpha=alpha).fit(basis_matrix, flux.ravel())
    reconstructed = clf.coef_[:, None, None]
    return reconstructed

# Run the code
np.random.seed(42)
spectra, basis = generate_synthetic_spectra()
flux, phot_wavelengths = generate_photometric_data(spectra, np.arange(spectra.shape[1]))
reconstructed = spectral_reconstruction(flux, phot_wavelengths, np.arange(spectra.shape[1]), basis)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(np.arange(spectra.shape[1]), spectra[0], label='True Spectrum')
plt.plot(np.arange(spectra.shape[1]), reconstructed[0], label='Reconstructed', linestyle='--')
plt.scatter(phot_wavelengths, flux[0], color='red', label='Photometric Data')
plt.legend()
plt.show()
```