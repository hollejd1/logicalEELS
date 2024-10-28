import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    L = np.arange(24)
    cmap=cm.copper(L/np.mean(L))

    for i in L:
        if spectra.shape[-1] == 1:
            # spectral dataset has been pre-processed and concatentated
            # expected shape is (13680, 240, 1)
            i1, i2 = i*570+300,i*570+400
            mean_spectrum = spectra[i1:i2].mean(axis=(0))
            ax.plot(energy, mean_spectrum, color=cmap[i])
            # ax.plot(energy, spectra[i*570:(i+1)*570].mean(axis=(0)), color=cmap[i])
        else:
            # spectral dataset has not been concatentated
            # expected shape is (24, 570, 1600)
            mean_spectrum = spectra[i, 300:400, inds].mean(axis=(0))
            ax.plot(energy[inds], mean_spectrum, color=cmap[i])
            # ax.plot(energy[inds], spectra[i, :, inds].mean(axis=(0)), color=cmap[i])

    ax.set_xlabel('Energy Loss (eV)')
    ax.set_ylabel('Average Counts')

    if title:
        ax.set_title(title)

    return ax

def plot_SI_class(color, xlabels=None, ylabels=None):
    '''

    '''
    # colors = [mpl.colormaps['viridis'](255), mpl.colormaps['viridis'](127),mpl.colormaps['viridis'](63), mpl.colormaps['viridis'](0)]
    def get_img_loc(imgNum):
        img_loc = np.zeros(2, dtype=int)
        img_loc[0] = imgNum % 3
        img_loc[1] = imgNum // 3
        if img_loc[1]>=7:
            img_loc[1] -= 8
        img_loc[1] += 1
        return img_loc

    SI_color = color.reshape(-1,30,19)
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(15, 8), constrained_layout=False)

    for i in range(SI_color.shape[0]):
        xx,yy = get_img_loc(i)
        axes[xx,yy].imshow(SI_color[i], vmin=0, vmax=3, cmap=mpl.colormaps['viridis'].reversed())
        axes[xx,yy].xaxis.set_ticklabels([])
        axes[xx,yy].xaxis.set_ticks([])
        axes[xx,yy].yaxis.set_ticks([])
        # axes[xx,yy].axis('off')

    if xlabels!=None:
        for i,xlabel in enumerate(xlabels):
            axes[-1,i].set_xlabel(xlabel)

    if ylabels!=None:
        for i,ylabel in enumerate(ylabels):
            axes[i,0].set_ylabel(ylabel)

    return fig, axes