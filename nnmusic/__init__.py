# __init__.py
# Author: Boo Mew Mew

"""Trains LSTM recursive neural networks to write music."""

import importlib as _il

import os as _os

_global_dict = globals()

for _s in _os.listdir(__path__[0]):
    if _s[-3:] != ".py" or _s[:2] == "__":
        continue
        
    _module = _il.import_module("nnmusic.{}".format(_s[:-3]))

    for _t in dir(_module):
        if _t[:1] == "_":
            continue
    
        _global_dict[_t] = getattr(_module, _t)

del _module