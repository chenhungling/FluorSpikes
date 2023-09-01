# -*- coding: utf-8 -*-
"""
GUI for inspecting post-analysis of calcium imaging fluorescence traces and
performing spike deconvolution using caiman/oasis package.

@author: Hung-Ling
"""
import os
import sys
from glob import glob
import numpy as np
import h5py
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.parametertree import Parameter, ParameterTree
from queue import SimpleQueue

from function import normalize
from function import transient
from function import utils

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi, estimate_time_constant

pg.setConfigOptions(imageAxisOrder='row-major')

# %% Functions        
def subtract_channel(x, y, bad_frames=None, percentile=75):
    if bad_frames is not None:  # Exclude outlier frames
        x2 = np.copy(x)[~bad_frames]
        y2 = np.copy(y)[~bad_frames]
    else:
        x2 = np.copy(x)
        y2 = np.copy(y)
    ix = x2<np.percentile(x2, percentile)  # Avoid taking saturated red-channel values
    iy = y2<np.percentile(y2, percentile)  # Avoid taking visible calcium transients
    idx = np.logical_and(ix, iy)
    if np.sum(idx) > 32:
        a, b = np.polyfit(x2[idx], y2[idx], 1)  # Linear regression
        background = a*x
        ysub = y - background + np.mean(background)  # Subtract linearly scaled x from y, without changing mean of y
    else:
        background = np.zeros_like(x)
        ysub = y
        print('Warning: too few good frames, recommend lowering corr_thr, or increase prct_thr')
    return ysub, background

# %% Implement computation intensive tasks that can be run in separate threads
class WorkerBackground(QtCore.QThread):
    '''Step 0: subtract background signal
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, data, queue):
        super(WorkerBackground, self).__init__()
        self.data = data
        self.queue = queue
        
    def run(self):
        F, Fneu, Fred, corr_frame, iplane, badframes, nframes_, params = self.data
        
        ntrials = len(nframes_)-1
        nplanes = len(set(iplane))
        ncells, T = F.shape
        
        if params['background_signal'] == 'red':
            Fsub = np.zeros_like(F)
            background = np.zeros_like(F)
            if corr_frame is None:
                corr_frame = np.ones((nplanes,T))
            if params['bad_frames']:
                bad_frames = (corr_frame<params['corr_thr']) | badframes
            else:
                bad_frames = (corr_frame<params['corr_thr'])  # Shape ()
            for j in range(ntrials):
                seg = slice(nframes_[j], nframes_[j+1])
                for i in range(ncells):
                    p = iplane[i]
                    Fsub[i,seg], background[i,seg] = subtract_channel(
                        Fred[i,seg], F[i,seg], bad_frames=bad_frames[p,seg],
                        percentile=params['prct_thr'])
                self.progress.emit(j+1)
                    
        elif params['background_signal'] == 'neuropil':
            background = params['neucoeff']*Fneu
            Fsub = F - background + np.mean(background, axis=1)[:,np.newaxis]
            self.progress.emit(ntrials)
            
        self.queue.put((Fsub, background))
        self.finished.emit()
    
# %% 
class WorkerDrift(QtCore.QThread):
    '''Step 1: correction for slow drift of the baseline
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, data, queue):
        super(WorkerDrift, self).__init__()
        self.data = data
        self.queue = queue

    def run(self):
        F, nframes_, params = self.data
        
        ntrials = len(nframes_)-1
        kwargs = {k: params[k] for k in ['method','percentile','window','niter']}
        
        Fcorr = np.zeros_like(F)
        drift = np.zeros_like(F)
        for j in range(ntrials):
            seg = slice(nframes_[j], nframes_[j+1])
            Fcorr[:,seg], drift[:,seg] = normalize.correct_drift_interp(F[:,seg], **kwargs)
            self.progress.emit(j+1)  # Send signal to the connection in the main thread
            
        self.queue.put((Fcorr, drift))  # Put results in queue allows communication with the main thread
        self.finished.emit()
        
# %%
class WorkerNormalize(QtCore.QThread):
    '''Step 2: estimate the constant baseline and noise sigma
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, data, queue):
        super(WorkerNormalize, self).__init__()
        self.data = data
        self.queue = queue

    def run(self):
        F, nframes_, params = self.data
        
        ncells = F.shape[0]
        ntrials = len(nframes_)-1
        frange = (params['fmin'], params['fmax'])
        kwargs = {k: params[k] for k in ['method','nstd','faverage','medfilt','norm_by']}
        kwargs.update({'frange': frange})
        
        Fnorm = np.zeros_like(F)  # Normalized traces
        baseline = np.zeros((ncells, ntrials))  # Mean baseline for each cell and each trial
        sigma = np.zeros((ncells, ntrials))
        
        if params['psd_sigma'] == 'cell-by-cell':  # Noise sigma for each cell and each trial
            for i in range(ncells):
                for j in range(ntrials):
                    seg = slice(nframes_[j], nframes_[j+1])
                    Fnorm[i,seg], baseline[i,j], sigma[i,j] = \
                        normalize.normalize_trace(F[i,seg], **kwargs)
                self.progress.emit(i+1)
        else:  # Compute one noise level for all cells at each trial
            for j in range(ntrials):
                seg = slice(nframes_[j], nframes_[j+1])
                if params['psd_sigma'] == 'pool':
                    sigma[:,j] = normalize.noise_level(
                        F[:,seg].ravel(order='C'), frange=frange, faverage=params['faverage'])
                else:
                    temp = np.zeros(ncells)
                    for i in range(ncells):
                        temp[i] = normalize.noise_level(
                            F[i,seg], frange=frange, faverage=params['faverage'])
                    if params['psd_sigma'] == 'mean':
                        sigma[:,j] = np.mean(temp)
                    elif params['psd_sigma'] == 'median':
                        sigma[:,j] = np.median(temp)
            for i in range(ncells):
                for j in range(ntrials):
                    seg = slice(nframes_[j], nframes_[j+1])
                    Fnorm[i,seg], baseline[i,j], _ = \
                        normalize.normalize_trace(F[i,seg], sigma=sigma[i,j], **kwargs)
                self.progress.emit(i+1)
        
        self.queue.put((Fnorm, baseline, sigma))
        self.finished.emit()
        
# %%
class WorkerSpike(QtCore.QThread):
    '''Step 3: Spike deconvolution
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, data, queue):
        super(WorkerSpike, self).__init__()
        self.data = data
        self.queue = queue
        
    def run(self):
        F, baseline, sigma, fps, nframes_, params, norm_by = self.data
        
        ncells, T = F.shape
        ntrials = len(nframes_)-1
        Sp = np.zeros((ncells, T))
        Ca = np.zeros((ncells, T))
        
        kwargs = {k: params[k] for k in ['p','lags','fudge_factor']}        
        if params['ar_coeff'] == 'trial-by-trial':
            g = np.zeros((ncells, params['p'], ntrials))
        else:
            g = np.zeros((ncells, params['p']))
            
        for i in range(ncells):
            if params['ar_coeff'] == 'trial-by-trial':
                for j in range(ntrials):
                    if norm_by == 'baseline':
                        sn = sigma[i,j]/baseline[i,j]
                    elif norm_by == 'sigma':
                        sn = 1.0
                    if params['s_min'] == 0:
                        s_min = None  # Standard L1 penalty
                    else:
                        s_min = params['s_min']*sn
                    seg = slice(nframes_[j], nframes_[j+1])
                    Ca[i,seg], _, _, g[i,:,j], _, Sp[i,seg], _ = constrained_foopsi(
                        F[i,seg], bl=0., c1=None, g=None, sn=sn, s_min=s_min,
                        method_deconvolution='oasis', optimize_g=0, **kwargs)
            else:  # ar_coeff in {'mean','median','pool'}:
                ## Estimate time constant (AR coefficient g)
                if params['ar_coeff'] == 'pool':
                    g[i,:] = estimate_time_constant(F[i], sn=None, **kwargs)
                else:
                    gs = np.zeros((ntrials, params['p']))
                    for j in range(ntrials):
                        if norm_by == 'baseline':
                            sn = sigma[i,j]/baseline[i,j]
                        elif norm_by == 'sigma':
                            sn = 1.0
                        seg = slice(nframes_[j], nframes_[j+1])
                        gs[j] = estimate_time_constant(F[i,seg], sn=sn, **kwargs)
                    if params['ar_coeff'] == 'mean':
                        g[i,:] = np.mean(gs, axis=0)
                    elif params['ar_coeff'] == 'median':
                        g[i,:] = np.median(gs, axis=0)
                ## Trial-by-trial spike deconvolution
                for j in range(ntrials):
                    if norm_by == 'baseline':
                        sn = sigma[i,j]/baseline[i,j]
                    elif norm_by == 'sigma':
                        sn = 1.0
                    if params['s_min'] == 0:
                        s_min = None  # Standard L1 penalty
                    else:
                        s_min = params['s_min']*sn
                    seg = slice(nframes_[j], nframes_[j+1])
                    Ca[i,seg], _, _, _, _, Sp[i,seg], _ = constrained_foopsi(
                        F[i,seg], bl=0., c1=None, g=g[i,:], sn=sn, s_min=s_min,
                        method_deconvolution='oasis', optimize_g=0, **kwargs)
            self.progress.emit(i+1)
            
        self.queue.put((Sp, Ca, g))
        self.finished.emit()
        
# %%
class WorkerTransient(QtCore.QThread):
    '''Step 4: Significant transient
    '''
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, data, queue):
        super(WorkerTransient, self).__init__()
        self.data = data
        self.queue = queue
        
    def run(self):
        F, baseline, sigma, fps, nframes_, params, norm_by = self.data
        
        ncells = F.shape[0]
        ntrials = len(nframes_)-1
        G = np.zeros(F.shape, dtype=bool)  # Transient
        G_ = np.zeros(F.shape, dtype=bool)  # Transient 1st 0-to-1 peak
        Gr = np.zeros(F.shape, dtype=bool)  # Transient (rising part)
        Gr_ = np.zeros(F.shape, dtype=bool)  # Transient (rising part) 1st 0-to-1 peak
        for j in range(ntrials):
            seg = slice(nframes_[j], nframes_[j+1])
            for i in range(ncells):
                sn = 1.0 if norm_by=='sigma' else sigma[i,j]/baseline[i,j]
                G[i,seg] = transient.transient_mask(
                    F[i,seg], sigma=sn, nsigma=params['nsigma'], fps=fps,
                    mindur=params['mindur'])
                Gr[i,seg] = transient.transient_rise(
                    F[i,seg], G[i,seg], sig=params['gsigma'])  
            G_[:,seg] = np.diff(np.hstack([np.zeros((ncells,1)),G[:,seg]]),axis=1) > 0
            Gr_[:,seg] = np.diff(np.hstack([np.zeros((ncells,1)),Gr[:,seg]]),axis=1) > 0
            self.progress.emit(j+1)
            
        self.queue.put((G, G_, Gr, Gr_))
        self.finished.emit()
        
# %% Subclass QtWidgets.QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, datapath=None):
        super(MainWindow, self).__init__()  # Inherit the constructor, methods and properties of the parent class
        self.resize(1400,1000)  # width, height
        self.setWindowTitle('FluorSpikes (suite2p)')
        self.statusBar().showMessage('Ready')
        
        ## ------------ Initialize internal variables ------------------------
        self.params = dict()  # Processing parameters
        self.loaded = False
        self.loaded_corr = False
        self.loaded_red = False
        self.done_background = False
        self.done_drift = False  
        self.done_normalize = False
        self.done_spike = False
        self.done_transient = False
        
        ## ------------ Central Widget Layout --------------------------------
        cw = QtWidgets.QWidget()  # Create a central widget to hold everything
        self.hlayout = QtWidgets.QHBoxLayout(cw)
        self.setCentralWidget(cw)  
        
        ## ------------ Add Widgets (column left) ----------------------------
        self.vlayout1 = QtWidgets.QVBoxLayout()
        self.partree = ParameterTree(showHeader=False)  # For processing parameters
        self.partree.setHeaderLabels(['Parameter'+' '*34, 'Value'])  # Workaround to force larger column width
        self.vlayout1.addWidget(self.partree)
        self.label1 = QtWidgets.QLabel('Total cells :')
        self.label2 = QtWidgets.QLabel('Spike (events/min) :')
        self.label3 = QtWidgets.QLabel('Transient (events/min) :')
        self.vlayout1.addWidget(self.label1)
        self.vlayout1.addWidget(self.label2)
        self.vlayout1.addWidget(self.label3)
        self.progbar = QtWidgets.QProgressBar()
        self.progbar.setProperty('value', 100)
        self.vlayout1.addWidget(self.progbar)
        self.hlayout.addLayout(self.vlayout1)
        
        ## ------------ Add Widgets (column right) ---------------------------
        self.vlayout2 = QtWidgets.QVBoxLayout()
        self.p0 = pg.PlotWidget()  # Raw fluorescence and background signal
        self.p1 = pg.PlotWidget()  # Background subtracted traces & drift
        self.p2 = pg.PlotWidget()  # Drift corrected traces & baseline
        self.p3 = pg.PlotWidget()  # Normalization & deconvolved calcium traces
        self.p4 = pg.PlotWidget()  # Spikes
        self.p5 = pg.PlotWidget()  # Transients
        ## Add widgets to the layout in their proper positions
        self.vlayout2.addWidget(self.p0)
        self.vlayout2.addWidget(self.p1)
        self.vlayout2.addWidget(self.p2)
        self.vlayout2.addWidget(self.p3)
        self.vlayout2.addWidget(self.p4)
        self.vlayout2.addWidget(self.p5)
        self.hlayout.addLayout(self.vlayout2)
        self.hlayout.setStretch(0,1)
        self.hlayout.setStretch(1,4)
        
        ## ------------ Create plot area (ViewBox + axes) --------------------
        self.p0.setLabel('left', 'Intensity (a.u.)')
        self.p1.setLabel('left', 'Background subtracted')
        self.p2.setLabel('left', 'Drift corrected')
        self.p3.setLabel('left', 'Normalized')
        self.p4.setLabel('left', 'Spikes')
        self.p5.setLabel('left', 'Transients')
        self.p0.setMouseEnabled(x=True, y=False)  # Enable only horizontal zoom for displaying traces
        self.p1.setMouseEnabled(x=True, y=False)
        self.p2.setMouseEnabled(x=True, y=False)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p4.setMouseEnabled(x=True, y=False)
        self.p5.setMouseEnabled(x=True, y=False)
        self.p0.getPlotItem().hideAxis('bottom')
        self.p1.getPlotItem().hideAxis('bottom')
        self.p2.getPlotItem().hideAxis('bottom')
        self.p3.getPlotItem().hideAxis('bottom')
        self.p4.getPlotItem().hideAxis('bottom')
        self.p1.setXLink(self.p0)  # link PlotWidget axis
        self.p2.setXLink(self.p0)
        self.p3.setXLink(self.p0)
        self.p4.setXLink(self.p0)
        self.p5.setXLink(self.p0)
        self.p5.setLabel('bottom', 'Time (s)')
        ## ------------ Create list to store plot item (curve) ---------------
        ## Use c1.append(p1.plot(...)) to add a plot and 
        ## p1.removeItem(c1[-1]) to remove last plot added to p1
        self.c0 = []  # Original fluorescence trace & background in self.p0  
        self.c1 = []  # Calculated drift in self.p1
        self.c2 = []  # Calculated baseline in self.p2
        self.c3 = []  # Fitted calcium trace in self.p3
        
        ## ------------ Setup munu and parameter tree ------------------------
        self.make_menu()
        self.make_parameters()
        #     self.store_params()
        
        ## ------------ Load data --------------------------------------------
        if datapath is not None:
            self.fname = datapath
            self.load_suite2p(click=False)
    
    # %% Setup menu structure
    def make_menu(self):
        menu = self.menuBar()
        menuFile = menu.addMenu('&File')
        ## Load suite2p processed data
        loadSuite2p = QtWidgets.QAction('Open ...', self)
        loadSuite2p.setShortcut('Ctrl+O')
        loadSuite2p.setStatusTip('Open suite2p processed data (*/suite2p/)')
        loadSuite2p.triggered.connect(lambda: self.load_suite2p(click=True))
        menuFile.addAction(loadSuite2p)
        ## Export results to the suite2p folder
        saveSuite2p = QtWidgets.QAction('Save', self)
        saveSuite2p.setShortcut('Ctrl+S')
        saveSuite2p.setStatusTip('Save results to suite2p folder')
        saveSuite2p.triggered.connect(self.save_suite2p)
        menuFile.addAction(saveSuite2p)

    # %% Setup parameters and link actions
    def make_parameters(self):
        params = [
            {'name':'Background', 'type':'group','children':[
                {'name':'subtract_background', 'type':'bool', 'value':True},
                {'name':'background_signal', 'type':'list', 'values':['red','neuropil'], 'value':'red'},
                {'name':'bad_frames', 'type':'bool', 'value':True},
                {'name':'corr_thr', 'type':'float', 'value':0., 'limits':(-1.0,1.0), 'step':0.01},
                {'name':'prct_thr', 'type':'int', 'value':90, 'limits':(0,100), 'step':1},
                {'name':'neucoeff', 'type':'float', 'value':0.7, 'limits':(0.0,1.0), 'step':0.01}]},
            {'name':'BACKGROUND', 'type':'action'},
            {'name':'Drift', 'type':'group','children':[
                {'name':'correct_drift', 'type':'bool', 'value':True},
                {'name':'method', 'type':'list', 'values':['prct','iter'], 'value':'iter'},
                {'name':'percentile', 'type':'int', 'value':8, 'limits':(0,100), 'step':1},
                {'name':'window', 'type':'int', 'value':400, 'limits':(10,10000), 'step':10},
                {'name':'niter', 'type':'int', 'value':10, 'limits':(1,200), 'step':5}]},
            {'name':'DRIFT', 'type':'action'},
            {'name':'Normalize', 'type':'group','children':[
                {'name':'method', 'type':'list', 'values':['iter','psd-kde','psd-emg'], 'value':'iter'},
                {'name':'nstd', 'type':'float', 'value':2.0, 'limits':(0.1,10.0), 'step':0.1},
                {'name':'psd_sigma', 'type':'list', 'values':['cell-by-cell','mean','median','pool'], 'value':'cell-by-cell'},
                {'name':'fmin', 'type':'float', 'value':0.25, 'limits':(0.0,0.49), 'step':0.01},
                {'name':'fmax', 'type':'float', 'value':0.5, 'limits':(0.01,0.5), 'step':0.01},
                {'name':'faverage', 'type':'list', 'values':['mean','median','logmexp'], 'value':'mean'},
                {'name':'medfilt', 'type':'int', 'value':0, 'limits':(0,20), 'step':1},
                {'name':'norm_by', 'type':'list', 'values':['baseline','sigma'], 'value':'baseline'}]},
            {'name':'NORMALIZE', 'type':'action'},
            {'name':'Spike', 'type':'group','children':[
                {'name':'p', 'type':'int', 'value':1, 'limits':(1,2), 'step':1},
                {'name':'ar_coeff', 'type':'list', 'values':['trial-by-trial','mean','median','pool'], 'value':'pool'},
                {'name':'lags', 'type':'int', 'value':8, 'limits':(1,100), 'step':1},
                {'name':'fudge_factor', 'type':'float', 'value':0.95, 'limits':(0.0,1.0), 'step':0.01},
                {'name':'s_min', 'type':'float', 'value':3.0, 'limits':(0.0,10.0), 'step':0.1}]},
            {'name':'SPIKE', 'type':'action'},
            {'name':'Transient', 'type':'group', 'children':[
                {'name':'denoised', 'type':'bool', 'value':False},
                {'name':'nsigma', 'type':'float', 'value':3.0, 'limits':(0.0,10.0), 'step':0.1},
                {'name':'mindur', 'type':'float', 'value':0.3, 'limits':(0.0,5.0), 'step':0.05, 'suffix':'s'},
                {'name':'rising', 'type':'bool', 'value':False},
                {'name':'gsigma', 'type':'float', 'value':1.0, 'limits':(0.0,10.0), 'step':0.1, 'suffix':'point'}]},
            {'name':'TRANSIENT', 'type':'action'},
            {'name':'Cell', 'type':'int', 'value':0, 'limits':(0,10000), 'step':1}
        ]
        self.par = Parameter.create(name='Processing Parameters', type='group', children=params)
        self.partree.setParameters(self.par, showTop=True)
        # self.par.child('Background').sigValueChanged.connect(self.change_params)  # lambda: (name='Background')
        # self.par.child('Drift'). sigTreeStateChanged.connect(self.change_params)  # (name='Drift')
        # self.par.child('Normalize'). sigTreeStateChanged.connect(self.change_params)  # (name='Normalize')
        # self.par.child('Spike'). sigTreeStateChanged.connect(self.change_params)  # (name='Spike')
        # self.par.child('Transient').sigTreeStateChanged.connect(self.change_params)  # (name='Transient')
        self.par.param('BACKGROUND').sigActivated.connect(self.button_background)
        self.par.param('DRIFT').sigActivated.connect(self.button_drift)
        self.par.param('NORMALIZE').sigActivated.connect(self.button_normalize)
        self.par.param('SPIKE').sigActivated.connect(self.button_spike)
        self.par.param('TRANSIENT').sigActivated.connect(self.button_transient)
        # for name in ['Background','Drift','Normalize','Spike','Transient']:
        #     par_dict = self.par.child(name).getValues()
        #     for key in par_dict.keys():
        #         self.par.child(name).param(key).sigValueChanged.connect(
        #             lambda: self.change_param(name,key))
        self.par.param('Cell').sigValueChanged.connect(self.change_cell)

    # %% IO suite2p            
    def load_suite2p(self, click=True):
        if click:
            self.fname = QFileDialog().getExistingDirectory(caption='Choose suite2p directory...')
            self.fname = QtCore.QDir.toNativeSeparators(self.fname)
        self.loaded = False
        self.loaded_corr = False
        self.loaded_red = False
        self.done_background = False  # For cleaning up previously loaded data
        self.done_drift = False  
        self.done_normalize = False
        self.done_spike = False
        self.done_transient = False
        
        if 'combined' in os.listdir(self.fname):
            self.subfolder = os.path.join(self.fname,'combined')
        else:
            self.subfolder = os.path.join(self.fname,'plane0')
            
        ## Get some information
        ops = np.load(os.path.join(self.subfolder,'ops.npy'), allow_pickle=True).item()
        self.fps = ops['fs']
        self.nplanes = ops['nplanes']
        nframes = ops['nframes_per_file'].astype(int)
        self.ntrials = len(nframes)
        self.nframes_ = np.hstack([0,np.cumsum(nframes)])
        self.badframes = ops['badframes']
        
        ## Get ROI plane
        iscell = np.load(os.path.join(self.subfolder,'iscell.npy'))[:,0].astype(bool)  # 0|1 for all ROIs
        stat = np.load(os.path.join(self.subfolder,'stat.npy'), allow_pickle=True)
        iplane = np.array([stat[i]['iplane'] for i in range(len(stat))])  # Plane index of all ROIs
        
        ## Check merged cells
        stat_list = [stat[iplane==p] for p in range(self.nplanes)]
        iscell_list = [iscell[iplane==p] for p in range(self.nplanes)]
        for p in range(self.nplanes):
            for i in range(len(stat_list[p])):
                if 'imerge' in stat_list[p][i].keys():
                    if len(stat_list[p][i]['imerge']) > 0:
                        for j in stat_list[p][i]['imerge']:
                            iscell_list[p][j] = False  # Cells that were merged
        iscell = np.hstack(iscell_list)
        self.iplane = iplane[iscell]
        
        ## Load fluorescence data and check iscell
        self.F0 = np.load(os.path.join(self.subfolder,'F.npy'))[iscell].astype(float)  # (ncell,T)
        self.Fneu = np.load(os.path.join(self.subfolder,'Fneu.npy'))[iscell].astype(float)
        self.ncells, self.T = self.F0.shape
        
        self.loaded = True
        self.statusBar().showMessage('Loaded: '+self.subfolder)
        
        try:  ## Load frame correlation
            self.corr_frame = np.vstack(
                [np.load(os.path.join(self.fname,'plane'+str(p),'correlation_frame.npy')) 
                 for p in range(self.nplanes)])
            self.loaded_corr = True
        except FileNotFoundError:
            self.corr_frame = None
            self.statusBar().showMessage('Frame correlation not found...')
            
        try:  ## Load red channel signal
            Fred = [np.load(os.path.join(self.fname,'plane'+str(p),'F_chan2.npy'))
                    for p in range(self.nplanes)]
            self.Fred = np.vstack(Fred)[iscell]
            self.loaded_red = True
        except FileNotFoundError:
            self.Fred = None
            self.statusBar().showMessage('Red channel not found...')
        
        if self.loaded:
            ## Set Cell ID range
            self.par.param('Cell').setLimits((0,self.ncells-1))
            self.label1.setText('Total cells : %d' % self.ncells)
            if self.nframes_[-1] != self.T:
                self.statusBar().showMessage(
                    'Sum of nframes per file is different than the total time points, check data')
            self.plot_init()
            self.load_fluorspikes()
            
    def save_suite2p(self):
        self.store_params()
        savename = QFileDialog().getSaveFileName(filter='HDF5 (*.h5 *.hdf5)')[0]
        fluorspikes = dict(background=self.background, drift=self.drift,
                           baseline=self.baseline, sigma=self.sigma,
                           F=self.F3, C=self.Ca, S=self.Sp, g=self.g,
                           G=self.G, G_=self.G_, Gr=self.Gr, Gr_=self.Gr_,
                           params=self.params)
        with h5py.File(savename, 'a') as f:
            if 'fluorspikes' in f:
                del f['fluorspikes']
            utils.recursively_save_dict_to_group(f, 'fluorspikes/', fluorspikes)
            
        self.statusBar().showMessage('Saved: '+savename)
    
    # %%
    def load_fluorspikes(self):
        try:
            files = []
            for ext in ['*.h5','*.hdf5']:
                files.extend(glob(os.path.join(self.fname, ext)))
            savename = files[0]
            with h5py.File(savename, 'r') as f:
                data = utils.recursively_load_dict_from_group(f, 'fluorspikes/')
                
            self.params = data['params']
            self.set_params()
            self.background = data['background']
            if self.params['Background']['subtract_background']:
                self.F1 = np.zeros_like(self.F0)
                for j in range(self.ntrials):
                    seg = slice(self.nframes_[j], self.nframes_[j+1])
                    self.F1[:,seg] = self.F0[:,seg] - self.background[:,seg] +\
                        self.background[:,seg].mean(axis=1)[:,np.newaxis]
            else:
                self.F1 = self.F0
            self.done_background = True
            
            self.drift = data['drift']
            if self.params['Drift']['correct_drift']:
                self.F2 = np.zeros_like(self.F1)
                for j in range(self.ntrials):
                    seg = slice(self.nframes_[j], self.nframes_[j+1])
                    self.F2[:,seg] = self.F1[:,seg] - self.drift[:,seg] +\
                        self.drift[:,seg].mean(axis=1)[:,np.newaxis]
            else:
                self.F2 = self.F1
            self.done_drift = True
            
            self.F3 = data['F']
            self.baseline = data['baseline']
            self.sigma = data['sigma']
            self.done_normalize = True
            
            self.Ca = data['C']
            self.Sp = data['S']
            self.g = data['g']
            self.done_spike = True
            
            self.G = data['G']
            self.G_ = data['G_']
            self.Gr = data['Gr']
            self.Gr_ = data['Gr_']
            self.done_transient = True
        except Exception:
            self.statusBar().showMessage('Fluorspikes result is not yet loaded')
            self.store_params()
            return
        
        self.plot_background()
        self.plot_drift()        
        self.plot_normalize()
        self.plot_spike()
        self.plot_transient()
    
    # %% Parameter action
    def store_params(self):
        '''Create Python dictionary to store parameters from Pyqtgraph's Parameter class
        '''
        for name in ['Background','Drift','Normalize','Spike','Transient']:
            par_dict = dict()
            for key, val in self.par.child(name).getValues().items():
                par_dict.update({key: val[0]})  # Pyqtgraph format key, (value, OrderedDict())
            self.params.update({name: par_dict})
    
    # def change_param(self, name, key):
    #     '''Update self.params dictionary when user changes parameter settings
    #     '''
    #     # for name in ['Background','Drift','Normalize','Spike','Transient']:
    #     #     par_dict = self.par.child(name).getValues()
    #     #     for k, val in par_dict.items():
    #     #         self.params[name][k] = val[0]
    #     par_dict = self.par.child(name).getValues()
    #     self.params[name][key] = par_dict[key][0]
        
    def set_params(self):
        '''Set parameter tree according to self.params when loaded processed data
        '''
        for name in ['Background','Drift','Normalize','Spike','Transient']:
            for key, val in self.params[name].items():
                self.par.child(name).param(key).setValue(val)
                        
    # %% Button action (put computation into worker thread and get results from queue)
    def button_background(self):
        if self.par.child('Background').param('subtract_background').value():
            params = dict()
            for key, val in self.par.child('Background').getValues().items():
                params.update({key: val[0]})
            self.progbar.setValue(0)
            self.queue = SimpleQueue()
            self.thread = WorkerBackground(
                (self.F0, self.Fneu, self.Fred, self.corr_frame, self.iplane,
                 self.badframes, self.nframes_, params),
                self.queue)
            self.thread.progress.connect(
                lambda count: self.progbar.setValue(int(100*count/self.ntrials)))
            self.thread.finished.connect(self.get_background)
            self.thread.finished.connect(self.thread.deleteLater)  # Delete thread object when control returns the the event loop
            self.thread.start()
            self.par.param('BACKGROUND').setOpts(enabled=False)
        else:
            self.F1 = self.F0
            self.background = np.zeros_like(self.F0)
            self.done_background = True
            self.plot_background()
    
    def get_background(self):
        self.F1, self.background = self.queue.get()
        self.done_background = True
        self.par.param('BACKGROUND').setOpts(enabled=True)
        self.plot_background()
        self.statusBar().showMessage('Background subtraction done')
        
    def button_drift(self):
        if self.par.child('Drift').param('correct_drift'):
            params = dict()
            for key, val in self.par.child('Drift').getValues().items():
                params.update({key: val[0]})
            self.progbar.setValue(0)
            self.queue = SimpleQueue()
            self.thread = WorkerDrift(
                (self.F1, self.nframes_, params), self.queue)
            self.thread.progress.connect(
                lambda count: self.progbar.setValue(int(100*count/self.ntrials)))
            self.thread.finished.connect(self.get_drift)
            self.thread.finished.connect(self.thread.deleteLater)  # Delete thread object when control returns the the event loop
            self.thread.start()
            self.par.param('DRIFT').setOpts(enabled=False)
        else:
            self.F2 = self.F1
            self.drift = np.zeros_like(self.F1)
            self.done_drift = True
            self.plot_drift()
        
    def get_drift(self):
        self.F2, self.drift = self.queue.get()
        self.done_drift = True
        self.par.param('DRIFT').setOpts(enabled=True)
        self.plot_drift()
        self.statusBar().showMessage('Drift correction done')
        
    def button_normalize(self):
        params = dict()
        for key, val in self.par.child('Normalize').getValues().items():
            params.update({key: val[0]})
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        self.thread = WorkerNormalize(
            (self.F2, self.nframes_, params), self.queue)
        self.thread.progress.connect(
            lambda count: self.progbar.setValue(int(100*count/self.ncells)))
        self.thread.finished.connect(self.get_normalize)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.par.param('NORMALIZE').setOpts(enabled=False)
        
    def get_normalize(self):
        self.F3, self.baseline, self.sigma = self.queue.get()
        self.done_normalize = True
        self.par.param('NORMALIZE').setOpts(enabled=True)
        self.plot_normalize()
        self.statusBar().showMessage('Normalization done')
    
    def button_spike(self):
        params = dict()
        for key, val in self.par.child('Spike').getValues().items():
            params.update({key: val[0]})
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        self.thread = WorkerSpike(
            (self.F3, self.baseline, self.sigma, self.fps, self.nframes_, params,
             self.par.child('Normalize').param('norm_by').value()),
             self.queue)
        self.thread.progress.connect(
            lambda count: self.progbar.setValue(int(100*count/self.ncells)))
        self.thread.finished.connect(self.get_spike)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.par.param('SPIKE').setOpts(enabled=False)
    
    def get_spike(self):
        self.Sp, self.Ca, self.g = self.queue.get()
        self.done_spike = True
        self.par.param('SPIKE').setOpts(enabled=True)
        self.plot_spike()
        self.statusBar().showMessage('Spike deconvolution done')
    
    def button_transient(self):
        params = dict()
        for key, val in self.par.child('Transient').getValues().items():
            params.update({key: val[0]})
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        F = self.Ca if self.par.child('Transient').param('denoised').value() else self.F3
        self.thread = WorkerTransient(
            (F, self.baseline, self.sigma, self.fps, self.nframes_, params,
             self.par.child('Normalize').param('norm_by').value()),
             self.queue)
        self.thread.progress.connect(
            lambda count: self.progbar.setValue(int(100*count/self.ntrials)))
        self.thread.finished.connect(self.get_transient)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.par.param('TRANSIENT').setOpts(enabled=False)
    
    def get_transient(self):
        self.G, self.G_, self.Gr, self.Gr_ = self.queue.get()
        self.done_transient = True
        self.par.param('TRANSIENT').setOpts(enabled=True)
        self.plot_transient()
        self.statusBar().showMessage('Transient done')
    
    # %% Drawing
    def plot_init(self):
        '''Plot Fig. 0 (self.p0)
        Put vertical lines separating adjacent trials/sesisons (draw once per dataset).
        '''
        self.p0.clear()  # Remove all items
        for t in self.nframes_:  
            self.p0.addLine(x=(t-0.5)/self.fps, pen=0.5)  # Gray
    
    def plot_background(self):
        '''Plot Fig. 1 (self.p1)
        Draw original fluorescence trace and background signal in Fig. 0 (place
        the object handle in self.c0 list so that items can be replotted/updating
        without clearing everything).
        Draw background-subtracted trace in Fig. 1
        '''
        this_cell = self.par.param('Cell').value()
        ts = np.arange(self.T)/self.fps
        for c in self.c0:
            self.p0.removeItem(c)
        self.c0 = []
        self.c0.append(self.p0.plot(ts, self.F0[this_cell], pen=(160,160,0)))  # Yellow Green
        self.c0.append(self.p0.plot(ts, self.background[this_cell], pen=0.7))
        self.p1.clearPlots()
        self.p1.plot(ts, self.F1[this_cell], pen=(0,160,0))  # Green   
        
    def plot_drift(self):
        '''Plot Fig. 1 (self.p1) and Fig. 2 (self.p2)
        Draw calculated drift in Fig. 1 (object handle in self.c1) and 
        drift-corrected trace in Fig. 2
        '''
        this_cell = self.par.param('Cell').value()
        ts = np.arange(self.T)/self.fps
        for c in self.c1:
            self.p1.removeItem(c)
        self.c1 = []
        if self.par.child('Drift').param('correct_drift').value():
            for j in range(self.ntrials):
                seg = slice(self.nframes_[j], self.nframes_[j+1])
                self.c1.append(self.p1.plot(ts[seg], self.drift[this_cell,seg], pen='w'))
        self.p2.clearPlots()
        self.p2.plot(ts, self.F2[this_cell], pen=(0,128,255))  # Blue

    def plot_normalize(self):
        '''Plot Fig. 2 (self.p2) and Fig. 3 (self.p3)
        Draw calculated baseline in Fig. 2 (object handle in self.c2) and 
        normalized trace in Fig. 3
        '''
        this_cell = self.par.param('Cell').value()
        ts = np.arange(self.T)/self.fps
        for c in self.c2:
            self.p2.removeItem(c)
        self.c2 = []
        for j in range(self.ntrials):
            x = [self.nframes_[j+i]/self.fps for i in [0,1]]
            self.c2.append(self.p2.plot(x, [self.baseline[this_cell,j]]*2, pen='w'))
        self.p3.clearPlots()
        self.p3.plot(ts, self.F3[this_cell], pen=0.5)  # Gray
    
    def plot_spike(self, clear_last=False):
        '''Plot Fig. 3 (self.p3) and Fig. 4 (self.p4)
        Draw fitted calcium trace in Fig. 3 (object handle in self.c3) and spikes in Fig. 4
        '''
        this_cell = self.par.param('Cell').value()
        ts = np.arange(self.T)/self.fps
        
        for c in self.c3:
            self.p3.removeItem(c)
        self.c3 = []
        self.c3.append(self.p3.plot(ts, self.Ca[this_cell], pen=(255,128,0)))  # Orange
        self.p4.clearPlots()
        st = self.Sp[this_cell]>0  # Spike time
        n_events = np.sum(st)  # Total events
        stems = np.vstack([np.zeros(n_events), self.Sp[this_cell,st]]).ravel(order='F')
        self.p4.plot(np.repeat(ts[st],2), stems, connect='pairs', pen='m')  # Magenta stem plot
        self.label2.setText('Spike (events/min) : %.4g' % (60*n_events/(self.T/self.fps)))
    
    def plot_transient(self):
        '''Plot Fig. 5 (self.p5) normalized trace and inferred transient
        '''
        this_cell = self.par.param('Cell').value()
        ts = np.arange(self.T)/self.fps
        self.p5.clearPlots()
        if self.par.child('Transient').param('denoised').value():
            fs = self.Ca[this_cell]
        else:
            fs = self.F3[this_cell]
        self.p5.plot(ts, fs, pen=0.5)  # Gray
        if self.par.child('Transient').param('rising').value():
            tt = self.Gr_[this_cell]  # Indicate the transient start time
            self.p5.plot(ts, fs, connect=self.Gr[this_cell], pen='r')  # Red
        else:
            tt = self.G_[this_cell]
            self.p5.plot(ts, fs, connect=self.G[this_cell], pen='r')  # Red
        n_events = np.sum(tt)
        # stems = np.vstack([np.zeros(n_events), fs[tt]]).ravel(order='F')
        # self.p5.plot(np.repeat(ts[tt],2), stems, connect='pairs', pen='r')  # Stem plot
        self.label3.setText('Transient (events/min) : %.4g' % (60*n_events/(self.T/self.fps)))

    def change_cell(self):
        '''Update plots when Cell ID is changed
        '''
        if self.done_background:
            self.plot_background()
        if self.done_drift:        
            self.plot_drift()
        if self.done_normalize:    
            self.plot_normalize()
        if self.done_spike:    
            self.plot_spike()
        if self.done_transient:
            self.plot_transient()

# %% Execute application event loop
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    