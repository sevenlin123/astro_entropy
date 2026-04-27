import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge


def spectral_model(wave, coeffs, base):
    """
    Simple linear combination of basis spectra.
    :param wave: array of wavelengths
    :param coeffs: coefficients for each basis
    :param base: base spectra matrix (n_basis x n_wave)
    :return: combined spectrum
    """
    return coeffs @ base


def generate_synthetic_spectra(n_spec, n_wave, n_basis):
    """
    Generate synthetic spectra by random mixing of basis spectra.
    :param n_spec: number of spectra to create
    :res