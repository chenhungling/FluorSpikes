# -*- coding: utf-8 -*-
"""
GUI for inspecting post-analysis of calcium imaging fluorescence traces and
performing spike deconvolution using caiman/oasis package.

@author: Hung-Ling
"""
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.parametertree import Parameter, ParameterTree
from queue import SimpleQueue

from function import normalize
from function import transient

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi, estimate_time_constant

pg.setConfigOptions(imageAxisOrder='row-major')

# %% Implement computation intensive tasks that can be run in separate threads
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
        srate = np.zeros((ncells, ntrials))
        
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
                    T1 = nframes_[j+1] - nframes_[j]
                    seg = slice(nframes_[j], nframes_[j+1])
                    Ca[i,seg], _, _, g[i,:,j], _, Sp[i,seg], _ = constrained_foopsi(
                        F[i,seg], bl=0., c1=None, g=None, sn=sn, s_min=s_min,
                        method_deconvolution='oasis', optimize_g=0, **kwargs)
                    srate[i,j] = np.sum(Sp[i,seg]>0)/(T1/fps)  # events/sec
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
                    T1 = nframes_[j+1] - nframes_[j]
                    seg = slice(nframes_[j], nframes_[j+1])
                    Ca[i,seg], _, _, _, _, Sp[i,seg], _ = constrained_foopsi(
                        F[i,seg], bl=0., c1=None, g=g[i,:], sn=sn, s_min=s_min,
                        method_deconvolution='oasis', optimize_g=0, **kwargs)
                    srate[i,j] = np.sum(Sp[i,seg]>0)/(T1/fps)  # events/sec
            self.progress.emit(i+1)
            
        self.queue.put((Sp, Ca, g, srate))
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
        self.setWindowTitle('FluorSpikes')
        self.resize(1350,900)  # width, height
        self.statusBar().showMessage('Ready')
        
        ## ------------ Initialize internal variables ------------------------
        self.params = dict()  # Processing parameters
        self.loaded = False
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
        self.partree.setHeaderLabels(['Parameter'+' '*24, 'Value'])  # Workaround to force larger column width
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
        self.p1 = pg.PlotWidget()  # Raw fluorescence and drift
        self.p2 = pg.PlotWidget()  # Drift corrected traces
        self.p3 = pg.PlotWidget()  # Normalized traces & transients
        self.p4 = pg.PlotWidget()  # Normalized traces & deconvolution
        self.p5 = pg.PlotWidget()  # Spikes
        ## Add widgets to the layout in their proper positions
        self.vlayout2.addWidget(self.p1)
        self.vlayout2.addWidget(self.p2)
        self.vlayout2.addWidget(self.p3)
        self.vlayout2.addWidget(self.p4)
        self.vlayout2.addWidget(self.p5)
        self.hlayout.addLayout(self.vlayout2)
        self.hlayout.setStretch(0,1)
        self.hlayout.setStretch(1,4)
        
        ## ------------ Create plot area (ViewBox + axes) --------------------
        self.p1.setLabel('left', 'Intensity (a.u.)')
        self.p2.setLabel('left', 'Drift corrected')
        self.p3.setLabel('left', 'Normalized')
        self.p4.setLabel('left', 'Spikes')
        self.p5.setLabel('left', 'Transient')
        self.p1.setMouseEnabled(x=True, y=False)  # Enable only horizontal zoom for displaying traces
        self.p2.setMouseEnabled(x=True, y=False)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p4.setMouseEnabled(x=True, y=False)
        self.p5.setMouseEnabled(x=True, y=False)
        self.p2.setXLink(self.p1)  # link PlotWidget axis
        self.p3.setXLink(self.p1)  # link PlotWidget axis
        self.p4.setXLink(self.p1)  # link PlotWidget axis
        self.p5.setXLink(self.p1)  # link PlotWidget axis
        self.p5.setLabel('bottom', 'Time (s)')
        ## ------------ Create list to store plot item (curve) ---------------
        ## Use c1.append(p1.plot(...)) to add a plot and 
        ## p1.removeItem(c1[-1]) to remove last plot added to p1
        self.c0 = []  # Original fluorescence trace in self.p1  
        self.c1 = []  # Calculated drift in self.p1
        self.c2 = []  # Calculated baseline in self.p2
        self.c3 = []  # Fitted calcium trace in self.p3
        ## ------------ Load data --------------------------------------------
        if datapath is not None:
            self.fname = datapath
            self.load_caiman_data(click=False)
        ## ------------ Setup munu and parameter tree ------------------------
        self.make_menu()
        self.make_parameters()
        self.store_params()
    
    # %% Setup menu structure
    def make_menu(self):
        menu = self.menuBar()
        menuFile = menu.addMenu('&File')
        ## Load caiman processed data
        actionOpen = QtWidgets.QAction('Open...', self)
        actionOpen.setShortcut('Ctrl+O')
        actionOpen.setStatusTip('Open caiman processed data (*.hdf5)')
        actionOpen.triggered.connect(lambda: self.load_caiman_data(click=True))
        menuFile.addAction(actionOpen)
        # ## Save (overwrite original hdf5 file)
        actionSave = QtWidgets.QAction('Save', self)
        actionSave.setShortcut('Ctrl+S')
        actionSave.setStatusTip('Save caiman hdf5 file')
        actionSave.triggered.connect(lambda: self.save_caiman(new=False))
        menuFile.addAction(actionSave)
        # ## Save as (create new hdf5 file)
        actionSaveAs = QtWidgets.QAction('Save as...', self)
        actionSaveAs.setShortcut('Ctrl+Shift+S')
        actionSaveAs.setStatusTip('Save as caiman hdf5 file')
        actionSaveAs.triggered.connect(lambda: self.save_caiman(new=True))
        menuFile.addAction(actionSaveAs)

    # %% Setup parameters and link actions
    def make_parameters(self):
        params = [
            {'name':'Drift', 'type':'group', 'children':[
                {'name':'correct_drift', 'type':'bool', 'value':True},
                {'name':'method', 'type':'list', 'value':'prct', 'values':['prct','iter']},
                {'name':'percentile', 'type':'int', 'value':8, 'limits':(0,100), 'step':1},
                {'name':'window', 'type':'int', 'value':600, 'limits':(10,10000), 'step':10, 'suffix':'point'},
                {'name':'niter', 'type':'int', 'value':10, 'limits':(1,200), 'step':5}]},
            {'name':'DRIFT', 'type':'action'},
            {'name':'Normalize', 'type':'group', 'children':[
                {'name':'method', 'type':'list', 'value':'iter', 'values':['iter','psd-kde','psd-emg']},
                {'name':'nstd', 'type':'float', 'value':2.0, 'limits':(0.1,10.0), 'step':0.1},
                {'name':'psd_sigma', 'type':'list', 'value':'cell-by-cell', 'values':['cell-by-cell','mean','median','pool']},
                {'name':'fmin', 'type':'float', 'value':0.25, 'limits':(0.0,0.49), 'step':0.01},
                {'name':'fmax', 'type':'float', 'value':0.5, 'limits':(0.01,0.5), 'step':0.01},
                {'name':'faverage', 'type':'list', 'value':'mean', 'values':['mean','median','logmexp']},
                {'name':'medfilt', 'type':'int', 'value':0, 'limits':(0,20), 'step':1, 'suffix':'point'},
                {'name':'norm_by', 'type':'list', 'value':'sigma', 'values':['baseline','sigma']}]},
            {'name':'NORMALIZE', 'type':'action'},
            {'name':'Spike', 'type':'group', 'children':[
                {'name':'p', 'type':'int', 'value':1, 'limits':(1,2), 'step':1},
                {'name':'ar_coeff', 'type':'list', 'value':'trial-by-trial', 'values':['trial-by-trial','mean','median','pool']},
                {'name':'lags', 'type':'int', 'value':6, 'limits':(1,100), 'step':1, 'suffix':'point'},
                {'name':'fudge_factor', 'type':'float', 'value':1.0, 'limits':(0.0,1.0), 'step':0.01},
                {'name':'s_min', 'type':'float', 'value':3.0, 'limits':(0.0,10.0), 'step':0.1, 'suffix':'sigma'}]},
            {'name':'SPIKE', 'type':'action'},
            {'name':'Transient', 'type':'group', 'children':[
                {'name':'denoised', 'type':'bool', 'value':False},
                {'name':'nsigma', 'type':'float', 'value':3.0, 'limits':(0.0,10.0), 'step':0.1},
                {'name':'mindur', 'type':'float', 'value':0.2, 'limits':(0.0,5.0), 'step':0.05, 'suffix':'s'},
                {'name':'rising', 'type':'bool', 'value':False},
                {'name':'gsigma', 'type':'float', 'value':1.0, 'limits':(0.0,10.0), 'step':0.1, 'suffix':'point'}]},
            {'name':'TRANSIENT', 'type':'action'},
            {'name':'Cell ID', 'type':'int', 'value':0, 'limits':(0,1000), 'step':1}
        ]
        self.par = Parameter.create(name='Processing Parameters', type='group', children=params)
        self.partree.setParameters(self.par, showTop=True)
        self.par.child('Drift').sigTreeStateChanged.connect(lambda: self.change_params(name='Drift'))
        self.par.child('Normalize').sigTreeStateChanged.connect(lambda: self.change_params(name='Normalize'))
        self.par.child('Spike').sigTreeStateChanged.connect(lambda: self.change_params(name='Spike'))
        self.par.child('Transient').sigTreeStateChanged.connect(lambda: self.change_params(name='Transient'))
        self.par.param('DRIFT').sigActivated.connect(self.button_drift)
        self.par.param('NORMALIZE').sigActivated.connect(self.button_normalize)
        self.par.param('SPIKE').sigActivated.connect(self.button_spike)
        self.par.param('TRANSIENT').sigActivated.connect(self.button_transient)
        self.par.param('Cell ID').sigValueChanged.connect(self.change_cell)
        
    # %% IO caiman
    def load_caiman_data(self, click=True):
        if click:
            self.fname = QFileDialog().getOpenFileName(
                caption='Choose caiman output', filter='HDF5 (*.h5 *.hdf5)')[0]
        self.done_drift = False
        self.done_normalize = False
        self.done_spike = False
        self.done_transient = False
        try:
            self.cnmf = load_CNMF(self.fname)  # Load all caiman output may be heavy and not everything is used here...
            estimates = self.cnmf.estimates
            self.fps = self.cnmf.params.data['fr']
            if len(self.cnmf.dims) == 2:
                self.nplanes = 1
            else:
                self.nplanes = self.cnmf.dims[2]
            if hasattr(self.cnmf,'nframes'):  # Created nframes in personal version of caiman pipeline
                self.nframes = self.cnmf.nframes
                self.ntrials = len(self.nframes)
            else:
                self.nframes = np.array([estimates.C.shape[1]])
                self.ntrials = 1  # Treat as single trial/session
            self.nframes_ = np.hstack([0,np.cumsum(self.nframes)])  # Convenient for slicing
            ## Check accepted list and read fluprescence data
            if hasattr(estimates,'accepted_list') and len(estimates.accepted_list)>0:
                accepted_list = estimates.accepted_list
            else:
                accepted_list = estimates.idx_components
            self.F0 = estimates.C[accepted_list] + estimates.YrA[accepted_list]
            self.ncells = self.F0.shape[0]
            self.par.param('Cell ID').setLimits((0, self.ncells-1))
            self.label1.setText('Total cells : %d' % self.ncells)
            self.statusBar().showMessage('Loaded: '+self.fname)
            self.loaded = True
            self.plot_init()
        except Exception:
            self.statusBar().showMessage('Loading '+self.fname+' failed. Try other file...')
            return
        if hasattr(self.cnmf, 'fluorspikes'):
            self.load_fluorspikes()

    def save_caiman(self, new=True):
        if new:
            fname_save = QFileDialog().getSaveFileName(filter='HDF5 (*.h5 *.hdf5)')[0]
        else:
            fname_save = self.fname
        fluorspikes = dict(drift=self.drift, bl=self.baseline, sn=self.sigma,
                           F=self.F2, C=self.Ca, S=self.Sp, g=self.g, srate=self.srate,
                           G=self.G, G_=self.G_, Gr=self.Gr, Gr_=self.Gr_,
                           params=self.params)
        self.cnmf.fluorspikes = fluorspikes
        self.cnmf.save(fname_save)
        self.statusBar().showMessage('Saved: '+fname_save)
        
    def load_fluorspikes(self):
        try:
            self.params = self.cnmf.fluorspikes['params']    
            if self.params['Drift']['correct_drift']:
                self.drift = self.cnmf.fluorspikes['drift']
                self.F1 = np.zeros_like(self.F0)
                for j in range(self.ntrials):
                    seg = slice(self.nframes_[j], self.nframes_[j+1])
                    self.F1[:,seg] = self.F0[:,seg] - self.drift[:,seg] +\
                        self.drift[:,seg].mean(axis=1)[:,np.newaxis]
            else:
                self.F1 = self.F0
            self.done_drift = True
            self.F2 = self.cnmf.fluorspikes['F']
            self.baseline = self.cnmf.fluorspikes['bl']
            self.sigma = self.cnmf.fluorspikes['sn']
            self.done_normalize = True
            self.Ca = self.cnmf.fluorspikes['C']
            self.Sp = self.cnmf.fluorspikes['S']
            self.g = self.cnmf.fluorspikes['g']
            self.srate = self.cnmf.fluorspikes['srate']
            self.done_spike = True
            self.G = self.cnmf.fluorspikes['G']
            self.G_ = self.cnmf.fluorspikes['G_']
            self.Gr = self.cnmf.fluorspikes['Gr']
            self.Gr_ = self.cnmf.fluorspikes['Gr_']
            self.done_transient = True
        except Exception:
            self.statusBar().showMessage('Fluorspikes result is not yet loaded')
            self.store_params()
            return
        self.set_params()
        self.plot_drift()        
        self.plot_normalize()
        self.plot_spike()
        self.plot_transient()
        
    # %% Parameter action
    def store_params(self):
        '''Create Python dictionary to store parameters from Pyqtgraph's Parameter class
        '''
        for name in ['Drift','Normalize','Spike','Transient']:
            par_dict = dict()
            for k, val in self.par.child(name).getValues().items():
                par_dict.update({k: val[0]})  # Pyqtgraph format key, (value, OrderedDict())
            self.params.update({name: par_dict})
    
    def change_params(self, name='Drift'):
        '''Update self.params dictionary when user changes parameter settings
        '''
        par_dict = self.par.child(name).getValues()
        for k, val in par_dict.items():
            self.params[name][k] = val[0]
        self.statusBar().showMessage(name+' setting changed')
    
    def set_params(self):
        '''Set parameter tree according to self.params when loaded processed data
        '''
        for name in ['Drift','Normalize','Spike','Transient']:
            for k, val in self.params[name].items():
                self.par.child(name).param(k).setValue(val)  # blockSignal=(lambda: self.change_params(name=name))
        
    # %% Button action (put computation into worker thread and get results from queue)
    def button_drift(self):
        if self.par.child('Drift').param('correct_drift').value():
            self.progbar.setValue(0)
            self.queue = SimpleQueue()
            self.thread = WorkerDrift(
                (self.F0, self.nframes_, self.params['Drift']), self.queue)
            self.thread.progress.connect(
                lambda count: self.progbar.setValue(int(100*count/self.ntrials)))
            self.thread.finished.connect(self.get_drift)
            self.thread.finished.connect(self.thread.deleteLater)  # Delete thread object when control returns the the event loop
            self.thread.start()
            self.par.param('DRIFT').setOpts(enabled=False)
        else:
            self.F1 = self.F0
            self.done_drift = True
            self.plot_drift()
        
    def get_drift(self):
        self.F1, self.drift = self.queue.get()
        self.done_drift = True
        self.par.param('DRIFT').setOpts(enabled=True)
        self.plot_drift()
        self.statusBar().showMessage('Drift correction done')
        
    def button_normalize(self):
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        self.thread = WorkerNormalize(
            (self.F1, self.nframes_, self.params['Normalize']), self.queue)
        self.thread.progress.connect(
            lambda count: self.progbar.setValue(int(100*count/self.ncells)))
        self.thread.finished.connect(self.get_normalize)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.par.param('NORMALIZE').setOpts(enabled=False)
        
    def get_normalize(self):
        self.F2, self.baseline, self.sigma = self.queue.get()
        self.done_normalize = True
        self.par.param('NORMALIZE').setOpts(enabled=True)
        self.plot_normalize()
        self.statusBar().showMessage('Normalization done')
    
    def button_spike(self):
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        self.thread = WorkerSpike(
            (self.F2, self.baseline, self.sigma, self.fps, self.nframes_, 
             self.params['Spike'], self.params['Normalize']['norm_by']),
             self.queue)
        self.thread.progress.connect(
            lambda count: self.progbar.setValue(int(100*count/self.ncells)))
        self.thread.finished.connect(self.get_spike)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.par.param('SPIKE').setOpts(enabled=False)
    
    def get_spike(self):
        self.Sp, self.Ca, self.g, self.srate = self.queue.get()
        self.done_spike = True
        self.par.param('SPIKE').setOpts(enabled=True)
        self.plot_spike()
        self.statusBar().showMessage('Spike deconvolution done')
    
    def button_transient(self):
        self.progbar.setValue(0)
        self.queue = SimpleQueue()
        F = self.Ca if self.params['Transient']['denoised'] else self.F2
        self.thread = WorkerTransient(
            (F, self.baseline, self.sigma, self.fps, self.nframes_, 
             self.params['Transient'], self.params['Normalize']['norm_by']),
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
    def plot_init(self, clear_all=True):
        '''Plot Fig. 1 (self.p1)
        Put vertical lines separating adjacent trials/sesisons (draw once per dataset).
        Draw original fluorescence trace and place the object handle in the list (self.c0)
        so that items can be replotted/updating without clearing everything.
        '''
        this_cell = self.par.param('Cell ID').value()
        ts = np.arange(self.F0.shape[1])/self.fps
        if clear_all:  # Redraw vertical line separating trial/session
            self.p1.clear()  # Remove all items
            for t in self.nframes_:  
                self.p1.addLine(x=(t-0.5)/self.fps, pen=0.5)  # Gray
        for c in self.c0:
            self.p1.removeItem(c)
        self.c0 = []
        self.c0.append(self.p1.plot(ts, self.F0[this_cell], pen=(0,128,255)))  # Blue
        
    def plot_drift(self):
        '''Plot Fig. 1 (self.p1) and Fig. 2 (self.p2)
        Draw calculated drift in Fig. 1 (object handle in self.c1) and 
        drift-corrected trace in Fig. 2
        '''
        this_cell = self.par.param('Cell ID').value()
        ts = np.arange(self.F0.shape[1])/self.fps
        for c in self.c1:
            self.p1.removeItem(c)
        self.c1 = []
        if self.par.child('Drift').param('correct_drift').value():
            for j in range(self.ntrials):
                seg = slice(self.nframes_[j], self.nframes_[j+1])
                self.c1.append(self.p1.plot(ts[seg], self.drift[this_cell,seg], pen='w'))
        self.p2.clearPlots()
        self.p2.plot(ts, self.F1[this_cell], pen=(0,160,0))  # Green

    def plot_normalize(self):
        '''Plot Fig. 2 (self.p2) and Fig. 3 (self.p3)
        Draw calculated baseline in Fig. 2 (object handle in self.c2) and 
        normalized trace in Fig. 3
        '''
        this_cell = self.par.param('Cell ID').value()
        ts = np.arange(self.F0.shape[1])/self.fps
        for c in self.c2:
            self.p2.removeItem(c)
        self.c2 = []
        for j in range(self.ntrials):
            x = [self.nframes_[j+i]/self.fps for i in [0,1]]
            self.c2.append(self.p2.plot(x, [self.baseline[this_cell,j]]*2, pen='w'))
        self.p3.clearPlots()
        self.p3.plot(ts, self.F2[this_cell], pen=0.5)  # Gray
    
    def plot_spike(self, clear_last=False):
        '''Plot Fig. 3 (self.p3) and Fig. 4 (self.p4)
        Draw fitted calcium trace in Fig. 3 (object handle in self.c3) and spikes in Fig. 4
        '''
        this_cell = self.par.param('Cell ID').value()
        ts = np.arange(self.F0.shape[1])/self.fps
        self.label2.setText('Spike (events/min) : %.4g' % np.mean(60*self.srate[this_cell]))
        for c in self.c3:
            self.p3.removeItem(c)
        self.c3 = []
        self.c3.append(self.p3.plot(ts, self.Ca[this_cell], pen=(255,128,0)))  # Orange
        self.p4.clearPlots()
        st = self.Sp[this_cell]>0  # Spike time
        stems = np.vstack([np.zeros(st.sum()), self.Sp[this_cell,st]]).ravel(order='F')
        self.p4.plot(np.repeat(ts[st],2), stems, connect='pairs', pen='m')  # Stem plot
    
    def plot_transient(self):
        '''Plot Fig. 5 (self.p5) normalized trace and inferred transient
        '''
        this_cell = self.par.param('Cell ID').value()
        T = self.F0.shape[1]
        ts = np.arange(T)/self.fps
        self.p5.clearPlots()
        if self.params['Transient']['denoised']:
            fs = self.Ca[this_cell]
        else:
            fs = self.F2[this_cell]
        self.p5.plot(ts, fs, pen=0.5)  # Gray
        if self.params['Transient']['rising']:
            tt = self.Gr_[this_cell]  # Indicate the transient start time
            self.p5.plot(ts, fs, connect=self.Gr[this_cell], pen='r')  # Red
        else:
            tt = self.G_[this_cell]
            self.p5.plot(ts, fs, connect=self.G[this_cell], pen='r')  # Red
        n_events = np.sum(tt)
        # stems = np.vstack([np.zeros(n_events), fs[tt]]).ravel(order='F')
        # self.p5.plot(np.repeat(ts[tt],2), stems, connect='pairs', pen='r')  # Stem plot
        self.label3.setText('Transient (events/min) : %.4g' % (60*n_events/(T/self.fps)))

    def change_cell(self):
        '''Update plots when Cell ID is changed
        '''
        if self.loaded:
            self.plot_init(clear_all=False)
        if self.done_drift:        
            self.plot_drift()
        if self.done_normalize:    
            self.plot_normalize()
        if self.done_spike:    
            self.plot_spike()
        if self.done_transient:
            self.plot_transient()
            
# %%
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    