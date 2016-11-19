# types.py
# Author: Boo Mew Mew

"""Data types used in nnmusic.

Types:
    amplitude -- Floating-point number representing the amplitude of a sound
                 wave.
"""

import numpy as _np

import tensorflow as _tf

amplitude        = _np.float64
tensor_amplitude = _tf.float64
to_amplitude     = _tf.to_double
