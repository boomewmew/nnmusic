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

import math          as _math
import nnmusic.io    as _io
import nnmusic.types as _types
import numpy         as _np
import tensorflow    as _tf

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
    weight = _tf.Variable(
        _tf.truncated_normal((n_hidden, n_channels),
                             dtype=_types.tensor_amplitude),
        dtype=_types.tensor_amplitude
    )
    bias = _tf.Variable(
        _tf.constant(0.1, shape=(n_channels,), dtype=_types.tensor_amplitude),
        dtype=_types.tensor_amplitude
    )
    
    data_shape = (None, n_channels)
    
    file_name = _tf.placeholder(_tf.string,              ()        )
    data      = _tf.placeholder(_types.tensor_amplitude, data_shape)
    target    = _tf.placeholder(_types.tensor_amplitude, data_shape)
    
    cell = _tf.nn.rnn_cell.LSTMCell(n_hidden)
    
    batch = _tf.contrib.training.batch_sequences_with_states(
        file_name, {"data": data}, {}, None,
        {
            "lstm_state": _tf.zeros(cell.state_size[0],
                                    dtype=_types.tensor_amplitude),
            "lstm_hidden": _tf.zeros(cell.state_size[1],
                                     dtype=_types.tensor_amplitude)
        }, 10000, 1, capacity=1
    )
    
    output = _tf.nn.state_saving_rnn(
        cell, _tf.unpack(batch.sequences["data"], axis=2), state_saver=batch,
        state_name=("lstm_state", "lstm_hidden")
    )[0]
    
    sum_square_error = _tf.reduce_sum(
        _tf.squared_difference(
            _tf.matmul(weight, _tf.reshape(_tf.pack(output),
                       (n_channels, -1))) + bias, target
        )
    )
    
    minimize = _tf.train.AdamOptimizer().minimize(sum_square_error)
    
    saver = _tf.train.Saver()

    sess = _tf.Session(
        config=_tf.ConfigProto(intra_op_parallelism_threads=n_threads)
    )
    sess.run(_tf.initialize_all_variables())
    
    for i in range(n_epochs):
        _io.print_now("Epoch {}/{}.".format(i, n_epochs))
        for f in _io.read_dir(dir_name, sample_rate, n_channels):
            input_data = f.data
            sess.run(
                minimize,
                {file_name: f.file_name, data: input_data,
                 target: input_data[1:, :]}
            )

    saver.save(sess, out_file_name)
