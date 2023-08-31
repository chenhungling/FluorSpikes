# FluorSpikes
This package implements GUIs in Python 3 for post-treatment of temporal sequences from calcium imaging data. Neuroscientists often use softwares such as [Suite2p](https://github.com/MouseLand/suite2p), [CaImAn](https://github.com/flatironinstitute/CaImAn) to extract calcium activity sequences. However, these temporal sequences are prone to baseline fluctuation or in- and out-of-focus motion during recording, which may hinder the correct identification of calcium transients and inferred spikes. FluorSpikes provides the following treatments in a user-freindly graphical interface:
* Background correction (require recording of a second non-functional channel)
* Drift correction (deal with baseline fluctuation)
* Normalization
* Spike deconvolution (with CaImAn's algorithm)
* Transient detection

## Installation
FluorSpikes uses CaImAn's algorithm to perform spike deconvolution, so you need to first install CaImAn following the instructions [here](https://github.com/flatironinstitute/CaImAn/blob/main/docs/source/Installation.rst).

Then, you can download the FluorSpikes source codes or
```
git clone https://github.com/chenhungling/FluorSpikes
cd FluorSpikes/fluorspikes
```

FluorSpikes uses Python packages such as [PyQt5](https://doc.bccnsoft.com/docs/PyQt5/), [pyqtgraph](https://pyqtgraph.readthedocs.io/en/latest/) and [h5py](https://www.h5py.org/) that come with CaImAn's installation. Thus, you can then run FluorSpikes in `caiman` environment (assume you call `caiman` for your CaImAn installation):
```
conda activate caiman
python fluorspikes_caiman.py
```

We also support reading Suite2p's output data, simply evoke the corresponding script (always in `caiman` environment):
```
python fluorspikes_suite2p.py
```

You can also run `fluorspikes_caiman.py` and `fluorspikes_suite2p.py` under [Spyder](https://www.spyder-ide.org/). However, you will need to set: menu Run/Configuration per file/Execute in an external system termal, to avoid conflict between Spyder's interactive console and PyQt5's event loop.

## Getting started

### Using the GUI
<p align="center" width=100%>
  <img src="images/Mainwindow_Caiman.png" width="95%">
</p>

After running CaImAn or Suite2p and curating your cells, you can load the results into FluorSpikes by selecting (menu File/Open...) the `.hdf5` file from CaImAn or the `suite2p` folder from Suite2p. Treatments of fluorescence traces for all accepted cells are controlled by the settings in the left panel and results are display in the right panels of the GUI.

### Outputs
<img align="right" width="25%" src="images/Output_hdf5_file.png">

FluorSpikes result is saved by clicking menu File/Save and it will be appended in the original `.hdf5` file (in case of loading suite2p folder, a new `.hdf5` file will be created). The naming of the outputs follows mainly CaImAn's:
* `C` (cells, time points): Fitted (denoised) fluorescence traces
* `F` (cells, time points): Normalized (raw) fluorescence traces
* `S` (cells, time points): Inferred spikes
* `g` (cells, p, sessions): Coefficients of the auto-regressive model
* `sn` (cells, sessions): Estimated noise level ($\sigma$)
* `bl` (cells, sessions): Estimated baseline level
* `params`: Parameter settings