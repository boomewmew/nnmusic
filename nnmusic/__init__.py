# nnmusic, a library for composing music using artificial neural networks.
# Copyright (C) 2016  Boo Mew Mew

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Address correspondence about this library to boomewmew@gmail.com.

"""Trains LSTM recurrent neural networks to write music."""

import importlib as _il
import os        as _os

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
