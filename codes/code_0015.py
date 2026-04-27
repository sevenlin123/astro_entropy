
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
Okay, I need to write a minimal Python implementation for the given algorithm. Let's break down the requirements.

First, the code must define a spectral model. So I'll create a function that generates a synthetic spectrum. Maybe using a Gaussian-like shape for simplicity.

Next, generate synthetic spectra. I'll create a few example spectra with varying parameters to show the model's flexibility.

Then, I need to generate photometric data from these spectra. Photometry typically involves integrating the spectrum over certain bands. I'll define a set of filters (like 'g', 'r', 'i') and compute the flux in each band for the spectra.

Now, the main part: reconstructing a synthetic spectrum from photometric data. I'll set up an optimization problem where I minimize the difference between the generated photometric data and the true values. Using scipy's minimize with least squares as the method makes sense here.

I'll structure this into functions: one for generating the initial guess, another for the loss function, and then perform the minimization. After that, I'll plot the original and reconstructed spectra for visual comparison.

Finally, I'll include plotting using matplotlib to visualize both the spectra and the photometric data points.

I should ensure that all necessary libraries are imported (numpy, scipy, sklearn) and that the code is well-structured with clear functions. I'll avoid any validation methods like LOOCV as per the constraints.

Putting it all together, the code will generate synthetic data, run the reconstruction, and display the results.
</think>

```python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def spectral_model(x, amplitude, wavelength, width):
    return amplitude * np.exp(-((x - wavelength)/width)**2)

def generate_spectra(wavelengths, amplitudes, wavelengths0, widths):
    spectra = []
    for amp, wv, wid in zip(amplitudes, wavelengths0, widths):
        x = np.linspace(300, 700, 400)
        spec = spectral_model(x, amp, wv, wid)
        spectra.append(spec + np.random.normal(0, 0.05, len(spec)))
    return x, spectra

def photometric_data(spectra, filter_transmissions):
    photometric = []
    for spec in spectra:
        flux = np.sum(spec * filter_transmissions[:, np.newaxis], axis=1)
        photometric.append(flux + np.random.normal(0, 0.02, len(flux)))
    return photometric

def initialize_reconstruction(filt_trans, n_params=3):
    return np.array([1.0]*n_params)

def loss_function(params, x_obs, y_obs, filt_trans):
    y_pred = np.sum(filt_trans[:, np.newaxis] * params, axis=1)
    return np.sum((y_obs - y_pred)**2)

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic spectra
    n_spectra = 5
    wavelengths = np.random.uniform(300, 700, n_spectra)
    amplitudes = np.random.uniform(0.3, 1.2, n_spectra)
    wavelengths0 = np.random.uniform(400, 600, n_spectra)
    widths = np.random.uniform(10, 30, n_spectra)
    
    x, true_spectra = generate_spectra(wavelengths, amplitudes, wavelengths0, widths)
    
    # Define filter transmissions (simple example)
    n_filters = 3
    filter_wavelengths = np.array([450, 550, 650])
    filter_transmissions = np.array([0.4, 0.7, 0.9])[:, np.newaxis]
    
    # Generate photometric data
    photometric = photometric_data(true_spectra, filter_transmissions)
    
    # Reconstruct spectrum from photometric data
    params_init = initialize_reconstructed_spectrum(filter_transmissions.shape[0])
    
    result = optimize.leastsq(loss_function, params_init, args=(filter_wavelengths, photometric, filter_transmissions))
    reconstructed = np.dot(filter_transmissions, result.x)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('Synthetic Spectra')
    for spec in true_spectra:
        plt.plot(x, spec)
    
    plt.subplot(2, 1, 2)
    plt.title('Reconstructed Spectrum')
    plt.plot(x, reconstructed)
    plt.show()
```