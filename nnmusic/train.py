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

"""Neural-net training."""

import nnmusic.io    as _io
import nnmusic.types as _types
import tflearn       as _tfl

DEFAULT_EPOCHS = 5000

def train_nn(dir_name, out_file_name, n_threads, sample_rate=_io.DEFAULT_RATE,
             n_channels=_io.DEFAULT_CHANNELS, n_hidden=24,
             n_epochs=DEFAULT_EPOCHS):
    """Train an LSTM recurrent neural network for writing music.
    
    Keyword arguments:
        dir_name      -- Directory containing audio files for training.
        out_file_name -- File to write neural-net state to.
        n_threads     -- Number of threads to run.
        sample_rate   -- Expected sample rate in Hz of audio files.
        n_channels    -- Expected number of audio channels.
        n_hidden      -- Number of nodes per hidden layer.
        n_epochs      -- Number of training epochs.
    """
    
