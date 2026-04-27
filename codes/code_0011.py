
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
Okay, so I need to write a Python code that implements an algorithm with specific requirements. Let me break down what's needed.

First, the code must define a spectral model. That probably means creating a function that represents the spectral shape, maybe something like a blackbody or a polynomial.

Next, I have to generate synthetic spectra. I'll need to create some wavelength points and compute the flux based on the spectral model. Adding noise might be necessary to make it realistic.

Then, from these synthetic spectra, I need to generate photometric data. Photometric data typically consists of magnitudes at certain bandpasses. I'll have to integrate the spectrum over each bandpass to get the flux in each filter, then convert that to magnitudes.

The main task is to reconstruct the synthetic spectrum from the photometric data. This sounds like an inverse problem. I can set this up as a linear regression problem where the coefficients are the spectrum values, and the data are the photometric measurements.

I remember that scipy has a function called optimize.curve_fit which can fit a function to data. Alternatively, using numpy's lstsq for linear algebra solutions might be more straightforward since it's a linear problem if the model is linear.

Wait, but the relationship between the spectrum and the photometry is linear if the photometric bands are fixed. So each pixel in the spectrum contributes to each band's flux. So yes, we can set up a matrix where each row corresponds to a filter and each column to a wavelength bin. Then, solving this system will give the reconstructed spectrum.

So I'll create a function to generate the photometric response for each wavelength. Each filter has a response curve, and the observed flux is the integral of the spectrum multiplied by the filter response.

But to simplify, perhaps I can represent each filter as a vector in the wavelength space and then set up the linear system.

Let me outline the steps:

1. Define a spectral model, say a Gaussian or a simple polynomial.
2. Generate synthetic spectra with some parameters, add noise.
3. For each wavelength, compute the response for each filter (e.g., u, g, r bands) and integrate the spectrum over each filter to get the photometric flux.
4. Convert fluxes to magnitudes.
5. Use photometric data (magnitudes) to reconstruct the original spectrum. This involves solving a linear system where the matrix is built from the filter responses, and the vector is the photometric data.

I need to make sure to use numpy for arrays, scipy for any optimizations, and scikit-learn for machine learning parts if needed.

Now, writing the code step by step:

- Import necessary libraries: numpy, scipy.optimize, sklearn.

- Define the spectral model. Maybe a simple Gaussian or a linear function.

- Generate wavelengths, then compute the synthetic flux using the model, add noise.

- Create filter response curves for each band. For example, u, g, r bands from some standard set.

- For each wavelength, calculate the flux through each filter by convolving or integrating. Since I'm using discrete points, perhaps I can approximate it by dot product.

- Compute magnitudes from fluxes.

- Set up the linear system Ax = b, where A is the filter response matrix, x is the spectrum, and b is the photometric data.

- Solve for x using numpy.linalg.lstsq.

- Plot the original and reconstructed spectra for comparison.

I should also include a function to plot the results, showing both the true and reconstructed spectra.

Potential issues: The matrix might be singular or ill-conditioned, so regularization could help, but the problem says to ignore validation, so maybe it's okay without it.

I think that's the plan. Now, putting it all into code.
</think>

```python
import numpy as np
from scipy import optimize
from sklearn.linear_model import LinearRegression

def spectral_model(wavelength, amplitude, width):
    return np.exp(-((wavelength - amplitude)/width)**2)

def generate_photometric_data(spectrum, filter_curves):
    photometry = []
    for filter_response in filter_curves:
        flux = np.sum(spectrum * filter_response)
        photometry.append(flux)
    return photometry

def main():
    np.random.seed(42)

    # Generate synthetic spectra
    wavelengths = np.linspace(400, 700, 100)  # nm
    true_spectrum = spectral_model(wavelengths, 550, 50) + np.random.normal(0, 0.1, len(wavelengths))

    # Define filter response curves (approximated)
    filters = {
        'u': np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]),
        'g': np.array([0.1, 0.25, 0.4, 0.45, 0.4, 0.25, 0.1]),
        'r': np.array([0.05, 0.15, 0.3, 0.4, 0.35, 0.2, 0.05])
    }

    filter_curves = [filters['u'], filters['g'], filters['r']]
    
    # Generate photometric data
    photometry = generate_photometric_data(true_spectrum, filter_curves)
    
    # Set up the linear system for reconstruction
    A = np.zeros((len(filter_curves), len(wavelengths)))
    for i, filter_resp in enumerate(filter_curves):
        A[i] = filter_resp

    # Solve using least squares
    reconstructed, _, _, _ = np.linalg.lstsq(A.T, photometry, rcond=None)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, true_spectrum, label='True Spectrum')
    plt.plot(wavelengths, reconstructed, label='Reconstructed Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
```