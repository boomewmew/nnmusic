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

"""Type aliases."""

import numpy      as _np
import tensorflow as _tf

tensor_size        = _tf.int64
size_list          = _tf.train.Int64List
SIZE_LIST_ARG_NAME = "int64_list"

amplitude        = _np.float64
tensor_amplitude = _tf.float64
to_amplitude     = _tf.to_double
