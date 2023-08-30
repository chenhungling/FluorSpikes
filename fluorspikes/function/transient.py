# -*- coding: utf-8 -*-
'''
Functions computing transients, activity rates and active transients from fluorescence traces.

@author: Hung-Ling
'''
import numpy as np
from scipy.ndimage import gaussian_filter1d 

# %%
def transient_mask(trace, sigma=None, nsigma=3, mindur=0.2, fps=20):
    '''
    Parameters
    ----------
    trace : np.ndarray (1D)
        Fluorescence trace.
    sigma : float
        Standard deviation of the baseline fluctuation (noise sigma).
        Provide this value using calcium.normalize().
        1 if the trace is z-scored.
    nsigma : int
        Number times the noise sigma above which signal will be considered as
        candidate transients.
    mindur : float
        Minimum transient duration in sec.
    fps : float
        Frames per second.
        
    Returns
    -------
    transient : np.ndarray (1D)
        Boolean array of transient (same length as trace).
    '''
    if sigma is None:
        sigma = np.std(trace)
        trace = trace - np.mean(trace)
        
    transient = (trace > nsigma*sigma)
    
    ## Check for minimum transient width
    T = len(trace)
    minwidth = int(mindur*fps)  # Minimum width in number of data points
    j = 0  # Index pointer
    while (j < T):
        if transient[j]:
            # Candidate transient starts, find its end point
            k = next((idx for idx, val in enumerate(transient[j:]) if not val), None)
            if k is not None:
                # Found transient[j+k] the next first False
                if (k < minwidth):
                    transient[j:j+k] = False
                j = j+k+1
            else:
                # k is None, i.e. transient is True until the end
                if (T-j) < minwidth:
                    transient[j:] = False
                j = T
        else:
            # Not transient, skip to the next data point
            j = j+1
    
    return transient

# %%
def transient_rate(trace, transient, fps=20):
    '''
    Parameters
    ----------
    trace : np.ndarray (1D)
        Fluorescence trace (normalized).
    transient : np.ndarray (1D)
        Boolean array of transient (same length as trace).
    fps : float, optional
        Frames per second. The default is 20.

    Returns
    -------
    transient_event : np.ndarray (1D)
        Time when transient event occurs (defined as transient mask switches from False to True).
    rate : float
        Number of events per second.
    auc : float
        Area-under-curve, here the sum of trace divided by the time duration in sec.
    '''
    T = len(trace)
    t = np.arange(T)/fps
    signal = trace * transient  # Significant calcium transient
    event = np.diff(np.hstack([0, transient])) > 0  # np.diff(int32 array)
    transient_event = t[event]  # Time of transient event (switch from False to True)
    n_transient = np.sum(event)  # Number of events
    rate = n_transient/(T/fps)  # Number of events per sec
    auc = np.sum(signal)/(T/fps)  # Area-under-curve (unit of trace per sec)
    
    return transient_event, rate, auc
    
# %%
def binarize(trace, transient, sig=None):
    '''
    Parameters
    ----------
    trace : np.ndarray (1D)
        Fluorescence trace (normalized)
    transient : np.ndarray (1D)
        Boolean array of transient.
    sig : float
        1D Gaussian kernel used to smooth the trace. The default is None.

    Returns
    -------
    active : np.ndarray (1D)
        Boolean array of active transient (rising part of the transient).
    '''
    trace2 = trace.copy()
    trace2[~transient] = 0
    if sig is not None:
        trace2 = gaussian_filter1d(trace2, sigma=sig, mode='nearest')
    
    active = np.logical_and(transient, np.gradient(trace2)>0)
    
    return active

def transient_rise(*args, **kwargs):
    '''Alias of the function binarize
    '''
    return binarize(*args, **kwargs)