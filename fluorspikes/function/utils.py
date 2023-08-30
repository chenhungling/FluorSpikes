# -*- coding: utf-8 -*-
"""
Adapted from caiman/utils/utils.py

@author: Hung-Ling
"""
import h5py
import numpy as np

# %%
# with h5py.File('test.hdf5','a') as f:
#     recursively_save_dict_to_group(f, 'params/', params)

def recursively_save_dict_to_group(h5file:h5py.File, path:str, dic:dict) -> None:
    '''
    Args:
        h5file: hdf5 object
            hdf5 file where to store the dictionary
        path: str
            Path within the hdf5 file structure
        dic: dict
            Dictionary to save
    '''
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    ## Save items to the hdf5 file
    for key, item in dic.items():
        
        if isinstance(item, (list, tuple)):
            if len(item) > 0 and all(isinstance(elem, str) for elem in item):
                item = np.string_(item)
            else:
                item = np.array(item)
        
        ## Save strings, np.int64, np.int32, and np.float64 types
        if isinstance(item, (np.int64, np.int32, np.float64, str, np.float, float, np.float32, int)):
            h5file[path + key] = item
            
        ## Save numpy array
        elif isinstance(item, np.ndarray):
            h5file[path + key] = item
        
        ## Save dictionary
        elif isinstance(item, dict):
            recursively_save_dict_to_group(h5file, path + key + '/', item)
        
        else:
            raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")
            
# %%
# with h5py.File(datapath, 'r') as f:
#     data = recursively_load_dict_from_group(f, 'fluorspikes/')
    
def recursively_load_dict_from_group(h5file:h5py.File, path:str) -> dict:
    '''Load dictionary from hdf5 object
    Args:
        h5file: hdf5 object
            Object where dictionary is stored
        path: str
            Path within the hdf5 file
    '''
    ans:dict = {}
    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            
            if isinstance(item[()], str):
                if item[()] == 'NoneType':
                    ans[key] = None
                else:
                    ans[key] = item[()]
            if isinstance(item[()], np.bool_):
                ans[key] = bool(item[()])
            else:
                ans[key] = item[()]  # np.ndarray

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_from_group(h5file, path + key + '/')
    return ans
