import os
import numpy as np

import hyperspy.api_nogui as hs

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import shift

def readDM4(fname, file_path, crop=(0,0)):
    """Read in a file and directory of data and return a numpy array of the spectral data
    fname - Filename of data DM4 file \\
    file_path - Directory of data files"""
    spectra = hs.load(os.path.join(file_path, fname))
    if crop!=(0,0):
        if hs.__version__[0]=='1':
            spectra.crop_signal1D(left_value=crop[0], right_value=crop[1])
        else:
            spectra.crop_signal(left_value=crop[0], right_value=crop[1])
    spectra_data = spectra.data.reshape(-1, np.shape(spectra)[-1])
    print(spectra_data.shape)
    return spectra_data

def getEnergyAxis(fname, crop=(0,0)):
    """Get Energy Axis array from a DM4 file"""
    s=hs.load(fname)
    if crop!=(0,0):
        if hs.__version__[0]=='1':
            s.crop_signal1D(left_value=crop[0], right_value=crop[1])
        else:
            s.crop_signal(left_value=crop[0], right_value=crop[1])
    emin = s.axes_manager[2].offset
    estep = s.axes_manager[2].scale
    ebins = np.shape(s)[-1]
    energy_axis = np.arange(0,ebins) * estep + emin
    return energy_axis

def loadDirDM4(filepath, crop=(0,0)):
    """Load a full directory of DM4 files into a np array
    """
    
    filename_list = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    print('Files found: {}'.format(len(filename_list)))
    energy_axis = getEnergyAxis(os.path.join(filepath, filename_list[0]), crop)
    spectras = []
    for fname in filename_list:
        print('Loading {}...'.format(fname))
        spectras.append(readDM4(fname, filepath, crop))
    spectras = np.array(spectras)
    return energy_axis, spectras, filename_list


def alignSpectra(energy_axis, spectra, alignment_window, X, edge_energy):
    """Align spectra along the energy axis using
    """
    aligned = np.zeros_like(spectra)
    peaks0, properties = find_peaks(spectra[-1].mean(axis=0)[alignment_window], width=10, prominence=40)
    sEdge = np.where(energy_axis==edge_energy)[0][0] - properties['left_bases'][0]-alignment_window.start
    for i in range(spectra.shape[0]):
        for j in range(spectra.shape[1]//X):
            peaks, _ = find_peaks(spectra[i][j*X:(j+1)*X].mean(axis=0)[alignment_window], width=10, prominence=40)
            try:
                sVal = (peaks[:2] - peaks0[:2]).mean()
            except:
                print(i,j, peaks.shape)
            aligned[i][j*X:(j+1)*X] = shift(spectra[i][j*X:(j+1)*X],[0, sEdge-1*sVal],prefilter=True, mode='nearest')
    return aligned

def backgroundSubtract(energy_axis, spectra, edge_window, fitting_window):
    def power_law(x, a, b):
        return a*np.power(x,b)
    
    backSub = np.zeros_like(spectra[:,:,edge_window])
    for i in range(spectra.shape[0]):
        print('Fitting Img:',i)
        for j in range(spectra.shape[1]):
            pars, cov = curve_fit(f=power_law,
                                xdata=energy_axis[fitting_window],
                                ydata=spectra[i,j,:][fitting_window],
                                maxfev=100000)
            line_fit = power_law(energy_axis, pars[0], pars[1])
            backSub[i,j] = np.subtract(spectra[i,j], line_fit)[edge_window]
    return backSub

def findEdgeIndex(energy_axis, emin, emax):
    ind_min = np.where(energy_axis==emin)[0][0]
    ind_max = np.where(energy_axis==emax)[0][0]+1
    return slice(ind_min, ind_max, 1)

def normAUC_inds(X, Ys, inds):
    orig_shape = Ys.shape
    Ys = Ys.reshape((-1, orig_shape[-1]))
    Y_norm = np.zeros_like(Ys)
    for i,Y in enumerate(Ys):
        auc = np.trapz(Y[inds], X[inds])
        Y_norm[i] = Y/(auc*0.03)
    return Y_norm.reshape(orig_shape)