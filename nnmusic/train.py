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
import tensorflow    as _tf
import tflearn       as _tfl
import time          as _time

DEFAULT_EPOCHS  = 5000
DEFAULT_THREADS = 1

def train_nn(in_file_name, out_dir, log_dir, n_threads=DEFAULT_THREADS,
             n_epochs=DEFAULT_EPOCHS):
    """Train an LSTM recurrent neural network for writing music.
    
    Keyword arguments:
        in_file_name  -- HDF5 file containing audio data for training.
        out_dir       -- Directory to write neural-net state to.
        log_dir       -- Directory for writing logs.
        n_threads     -- Number of threads to run.
        n_epochs      -- Number of training epochs.
    """
    _io.print_now(
        "Training neural net with audio from file {}.".format(in_file_name)
    )
    
    dataset = _h5py.File(in_file_name, "r")[_io.DATASET_NAME]
    in_data = dataset[:, :-1, :]
    target  = dataset[:, 1:,  :]
    
    chunk_size = in_data.shape[1]
    n_channels = in_data.shape[2]
    
    net = _tfl.input_data((None, chunk_size, n_channels),
                          dtype=_types.tensor_amplitude)

    net = _tfl.lstm(net, n_channels, return_seq=True)
    net = _tf.pack(net, 1)
    
    net = _tfl.regression(net, loss=_tfl.mean_square, metric=_tfl.R2(),
                          dtype=_types.tensor_amplitude)

    _tfl.SequenceGenerator(
        net, {
            x: i
            for i, x in enumerate((y for a in dataset for b in a for y in b))
        }, chunk_size, tensorboard_verbose=3, tensorboard_dir=log_dir,
        checkpoint_path="{}/nn-state".format(out_dir)
    ).fit(in_data, target, n_epochs, validation_set=0.5, show_metric=True,
          run_id=_time.strftime("%Y-%m-%d-%H:%M:%S"))
