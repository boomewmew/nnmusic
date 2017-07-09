#!/usr/bin/env python

# nnmusic, a program for composing music using artificial neural networks.
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

"""Compose music using artificial neural networks."""

import importlib as _il
import os        as _os
import re        as _re

for _file_name in _os.listdir(__path__[0]):
    if _re.match("(?!__init__).*\.py", _file_name):
        globals().update(
            _il.import_module("nnmusic.{}".format(_file_name[:-3])).__dict__
        )
