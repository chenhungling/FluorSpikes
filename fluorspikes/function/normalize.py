# -*- coding: utf-8 -*-
"""
Normalization of fluorescence traces.
Methods for estimate noise standard deviation (sigma) and baseline.

@author: Hung-Ling
"""
import numpy as np
from scipy.ndimage import percentile_filter, gaussian_filter1d, median_filter
from scipy.signal import welch
from scipy.stats import gaussian_kde, exponnorm

# %%
def correct_drift(F, method='prct', percentile=8, window=600, niter=10):
    '''
    Correction for slow drift of the baseline.
    
    Parameters
    ----------
    F : np.ndarray (1D or 2D)
        Fluorescence trace. If 2D array, shape is (# cells, # time points)
    method : str 'prct'|'iter'
        Method used to derive the drift
        - 'prct' : sliding percentile filter
        - 'iter' : iterative smoothing process
    percentile : int
        In method='prct', percentile between 0 and 100. The default is 8.
    window : int
        Time scale over which the baseline fluctuates (# data points, i.e. int(fps*sec))
        Small window produces a flexible baseline so fluorescence trace is strongly flattened.
        Large window leads to a globally more rigid baseline.
        In method='prct', this is the window used to calculate the percentile.
        In method='iter', the Gaussian smoothing kernel (sigma) takes the value window//4.
        The default is 600, e.g. 10(Hz)*60(sec)
    niter : int
        In method='iter', number of iteration. The default is 10.
        
    Returns
    -------
    flatF : np.ndarray (1D or 2D)
        Drift-corrected trace (same shape as F).
    drift : np.ndarray (1D or 2D)
        Inferred baseline fluctuation (same shape as F).
    '''
    if method == 'prct':
        if F.ndim == 1:
            drift = percentile_filter(F, percentile, size=window)  # Default mode='reflect'
            flatF = F - drift + drift.mean()  # Make mean F unchanged
        elif F.ndim == 2:
            drift = percentile_filter(F, percentile, size=(1,window))  # Do nothing along axis 0 
            flatF = F - drift + drift.mean(axis=1)[:,np.newaxis]
        else:
            raise TypeError('Invalid dimension for fluorescence array, need to be 1D or 2D')
    elif method == 'iter':
        sigma = window//4
        if F.ndim == 1:
            drift = gaussian_filter1d(F, sigma=sigma)  # Default mode='reflect'
            for it in range(niter):
                deviate = np.maximum(drift-F, 0)
                drift -= gaussian_filter1d(deviate, sigma=sigma)
            flatF = F - drift + drift.mean()  # Make mean F unchanged
        elif F.ndim == 2:
            drift = gaussian_filter1d(F, sigma=sigma, axis=1)
            for it in range(niter):
                deviate = np.maximum(drift-F, 0)
                drift -= gaussian_filter1d(deviate, sigma=sigma, axis=1)
            flatF = F - drift + drift.mean(axis=1)[:,np.newaxis]
        else:
            raise TypeError('Invalid dimension for fluorescence array, need to be 1D or 2D')    
    else:
        raise ValueError('Unknown method, choose "prct" or "iter".')
        
    return flatF, drift

# %%
def correct_drift_interp(F, method='prct', percentile=8, window=600, niter=10):
    '''Fast implementation of correct_drift with downsampling and linear interpolation
    '''
    if F.ndim == 1:
        F = F.reshape((1,-1))  # Row vector
    ncell, T = F.shape
    ds = int(window/200)*2 + 1  # Downsampling factor (odd number), roughly <100 new data points in the time window
    T2 = int(T/ds)*ds
    F2 = F[:,:T2].reshape((ncell,int(T/ds),ds)).mean(axis=2)
    t2 = np.arange(T2).reshape((int(T/ds),ds)).mean(axis=1)
    
    if method == 'prct':
        drift2 = percentile_filter(F2, percentile, size=(1,window//ds))
    elif method == 'iter':
        sigma = int(window/ds/4)
        drift2 = gaussian_filter1d(F2, sigma=sigma, axis=1)
        for it in range(niter):
            deviate = np.maximum(drift2-F2, 0)
            drift2 -= gaussian_filter1d(deviate, sigma=sigma, axis=1)
    
    drift = np.zeros_like(F)
    for i in range(ncell):
        drift[i] = np.interp(np.arange(T), t2, drift2[i])
    flatF = F - drift + drift.mean(axis=1)[:,np.newaxis]
    if ncell == 1:
        drift = drift.squeeze()
        flatF = flatF.squeeze()
    
    return flatF, drift
    
# %%
def normalize_trace(trace, baseline=None, sigma=None, method='iter', nstd=2.0,
                    frange=(0.25,0.5), faverage='mean', medfilt=0,
                    norm_by='baseline'):
    '''
    Normalize the fluorescence trace by substracting a constant baseline then 
    dividing the baseline (so-called "Delta F over F") or
    dividing the noise standard deviation (i.e. sigma, so-called "z-score").
    
    Parameters
    ----------
    trace : np.ndarray (1D), shape (T,)
        Fluorescence trace.
    baseline : None or float
        The constant baseline to be subtracted.
        If None (default), it is estimated using {method}
    sigma : None or float
        In norm_by='sigma', the noise level used.
        If None (default), it is estimated using {method}
    method : str 'iter'|'psd-kde'|'psd-emg'
        Method for calculating the constant baseline and noise sigma.
        - 'iter' for iterative thresholding method (default)
        - 'psd-kde' for sigma from power spectral density and baseline from kernel density estimate
        - 'psd-emg' for sigma from power spectral density and baseline from exponentially modified Gaussian
    nstd : float
        In method='iter', threshold used (mean + nstd*std). The default is 2.0
    frange : tuple of float (fmin, fmax)
        In method='psd-', range of frequency over which the spectrum is averaged.
        (Nyquist rate, positive and max value <= 0.5). The default is (0.25,0.5).
    faverage : str 'mean'|'median'|'logmexp'
        Method of averaging. The default is 'mean'.
        'logmexp' for exponential of the mean of log values.
    medfilt : int
        If > 0, preprocess trace with a median filter over {medfilt} data points
        to remove outliers (only in method='psd-emg', applied before fitting baseline).
        The default is 0
    norm_by : str 'baseline'|'sigma'
        Normalize (divide) the fluorescence by
        - 'baseline', that is "Delta F over F" for two-photon recordings
        - 'sigma', that is "z-score" or "signal-to-noise ratio"

    Returns
    -------
    norm_trace : np.ndarray (1D)
        Normalized fluorescence trace.
    baseline : float
        Substracted mean baseline level.
    sigma : float
        Standard deviation of the noisy baseline (assuming Gaussian noise).
    '''        
    ## Attention: method 'iter' overestimates baseline and sigma when the firing rate is high!!
    if method == 'iter':
        if baseline is None or sigma is None:
            s0 = trace.max() - trace.min()
            m1 = trace.mean()
            s1 = trace.std()
            F = trace.copy()
            while (s0-s1)/s1 > 0.001:  # Accept if convergence to <0.1% relative change
                s0 = s1
                F = F[F < m1+nstd*s1]  # Exclude signal above {nstd} times std
                m1 = F.mean()
                s1 = F.std()
        if baseline is None:
            baseline = m1
        if sigma is None:
            sigma = s1
        
    elif method == 'psd-kde':  # Relatively fast, possibly overestimate baseline
        if sigma is None:
            sigma = noise_level(trace, frange=frange, faverage=faverage)
        if baseline is None:
            baseline = baseline_level(trace, method='kde')
            
    elif method == 'psd-emg':  # Higher computation load, robust noise baseline estimate
        if sigma is None:
            sigma = noise_level(trace, frange=frange, faverage=faverage)
        if baseline is None:
            if medfilt > 0:
                trace2 = median_filter(trace, size=medfilt)
                baseline = baseline_level(trace2, method='emg', sigma=sigma)
            else:
                baseline = baseline_level(trace, method='emg', sigma=sigma)
    else:
        raise ValueError('Unknown method, choose "iter", "psd-kde" or "psd-emg".')
        
    if norm_by == 'baseline':
        norm_trace = (trace - baseline) / baseline
    elif norm_by == 'sigma':
        norm_trace = (trace - baseline) / sigma
    else:
        raise ValueError('Unknown norm_by, choose "baseline" or "sigma".')
    
    return norm_trace, baseline, sigma

# %%
def noise_level(trace, frange=(0.25,0.5), faverage='mean'):
    '''
    Estimate noise level (standard deviation) through the power spectral density
    over the range of large frequencies.

    Parameters
    ----------
    trace : 1D array
        Typically fluorescence intensities with one entry per time bin.
    frange : tuple of float (fmin, fmax)
        Range of frequency over which the spectrum is averaged
        (Nyquist rate, positive and max value <= 0.5). The default is (0.25,0.5)
    faverage : str 'mean'|'median'|'logmexp'
        Method of averaging. The default is 'mean'.
        'logmexp' is exponential of the mean of log values, this might be more
        robust against unexpectly large power (?)
    
    Returns
    -------
    sigma : float
        Noise standard deviation
    '''
    ## Compute the power spectrum density using Welch’s method
    freqs, Pxx = welch(trace)  # Default fs=1.0, freqs is between 0 and 0.5 (Nyquist or folding frequency)
    indices = (freqs>frange[0]) & (freqs<frange[1])
    ## Estimate the noise variance by averaging high frequency part of the PSD
    if faverage == 'mean':
        noise_power = np.mean(Pxx[indices])/2
    elif faverage == 'median':
        noise_power = np.median(Pxx[indices])/2
    elif faverage == 'logmexp':
        noise_power = np.exp(np.mean(np.log(Pxx[indices]/2)))
    else:
        raise ValueError('Unknown faverage method, choose "mean", "median" or "logmexp"')
        
    return np.sqrt(noise_power)

# %%
def baseline_level(trace, method='kde', sigma=None):
    '''
    Estimate constant baseline level of fluorescence trace.

    Parameters
    ----------
    trace : 1D array
        Time serie of fluorescence intensity.
    method : str 'kde' (default)|'emg'
        - 'kde' for mode of the Gaussian Kernel Density Estimation
        - 'emg' for mode of the fitted Exponentially Modified Gaussian distribution
    sigma : None or float
        In method='emg', noise standard deviation used to fix the Gaussian tail.
        The default is None, the Gaussian tail is fitted.

    Returns
    -------
    baseline : float
        Mean baseline of fluorescence.
    '''
    if method == 'kde':
        kernel = gaussian_kde(trace)
        # min_ = np.min(trace)
        # max_ = np.max(trace)
        min_, max_ = np.percentile(trace, [1,99])
        x = np.linspace(min_, max_, 64)  # Reduce the number of points as evaluation on large amount of data points is costly 
        ind = np.argmax(kernel.evaluate(x))
        x2 = np.linspace(x[max(0,ind-1)], x[min(ind+1,63)], 32)  # Refinement, 64*32/2=1024
        baseline = x2[np.argmax(kernel.evaluate(x2))]  # Mode (max) of the distribution
    elif method == 'emg':
        if sigma is None:
            K, loc, scale = exponnorm.fit(trace)
        else:
            K, loc, scale = exponnorm.fit(trace, fscale=sigma)  # Fixed scale
        rv = exponnorm(K, loc=loc, scale=scale)
        x = np.linspace(rv.ppf(0.05), rv.ppf(0.95), 1024)
        baseline = x[np.argmax(rv.pdf(x))]  # Mode of the EMG distribution
    else:
        raise ValueError('Unknown method, choose "kde" or "emg"')
    
    return baseline
        