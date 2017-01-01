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

import h5py          as _h5py
import nnmusic.io    as _io
import nnmusic.types as _types
import tflearn       as _tfl

DEFAULT_EPOCHS  = 5000
DEFAULT_THREADS = 1

def train_nn(in_file_name, out_file_name, log_dir, n_threads=DEFAULT_THREADS,
             n_hidden=24, n_epochs=DEFAULT_EPOCHS):
    """Train an LSTM recurrent neural network for writing music.
    
    Keyword arguments:
        in_file_name  -- HDF5 file containing audio data for training.
        out_file_name -- File to write neural-net state to.
        log_dir       -- Directory for writing logs.
        n_threads     -- Number of threads to run.
        n_hidden      -- Number of nodes per hidden layer.
        n_epochs      -- Number of training epochs.
    """
    _io.print_now(
        "Training neural net with audio from file {}.".format(in_file_name)
    )
    
    dataset = _h5py.File(in_file_name, "r")[_io.DATASET_NAME]
    
    max_len    = dataset.shape[1]
    n_channels = dataset.shape[2]
    
    net = _tfl.input_data((None, max_len, n_channels),
                          dtype=_types.tensor_amplitude)

    net = _tfl.lstm(net, n_hidden, return_seq=True)

    net = _tfl.fully_connected(net, n_channels)

    net = _tfl.regression(net, loss="mean_square", metric="r2",
                          dtype=_types.tensor_amplitude)
    
    generator = _tfl.SequenceGenerator(net, seq_maxlen=max_len,
                                       tensorboard_verbose=3,
                                       tensorboard_dir=log_dir,
                                       checkpoint_path=out_file_name)
    generator.fit(dataset[:, :-1, :], dataset[:, 1:, :], n_epochs,
                  validation_set=0.5, show_metric=True)
