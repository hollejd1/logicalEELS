import matplotlib.pyplot as plt
import numpy as np
from .preprocess import findEdgeIndex

def plotSpectra(ax: plt.axes, energy: np.ndarray, 
                spectra: np.ndarray, title: str = None, window: list = None) -> plt.axes:
    """
    Plots the mean of 100 spectra from the 24 scans in the MXenes EELS dataset.
    
    Args:
        ax (plt.axes): matplotlib axis in which the data will be plotted 
        energy (np.ndarray): 1D numpy array of the energy range with shape (1600, ) or (240, )
        spectra (np.ndarray): 2D numpy array of the stacked EELS data with shape (24, 570, 1600) or (13680, 240, 1)
        title (str): title used for this subplot
        window (list): array which contains the lower and upper limits the energy axis with shape (2, 1)

    Returns:
        ax (plt.axes): matplotlib axis with plotted data
    """
    if window:
        inds = findEdgeIndex(energy, window[0], window[1])
    else:
        inds = slice(0,spectra.shape[-1])

    for i in range(24):
        if spectra.shape[-1] == 1:
            # spectral dataset has been pre-processed and concatentated
            # expected shape is (13680, 240, 1)
            ax.plot(energy, spectra[i*570+300:(i+1)*570+400].mean(axis=(0)))
        else:
            # spectral dataset has not been concatentated
            # expected shape is (24, 570, 1600)
            ax.plot(energy[inds], spectra[i, 300:400, inds].mean(axis=(0)))

    ax.set_xlabel('Energy Loss (eV)')
    ax.set_ylabel('Average Counts')

    if title:
        ax.set_title(title)

    return ax