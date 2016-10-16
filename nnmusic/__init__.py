# __init__.py
# Author: Boo Mew Mew

"""Initialize nnmusic package."""

import importlib as il

import os

_global_dict = globals()

for s in os.listdir(__path__[0]):
    if s[-3:] != ".py" or s[:2] == "__":
        continue

    module = il.import_module("nnmusic.{}".format(s[:-3]))

    try:
        attribute_list = module.__all__
    except AttributeError:
        attribute_list = [t for t in dir(module) if t == "" or t[0] != "_"]
    
    for t in attribute_list:
        _global_dict[t] = getattr(module, t)
