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

"""Composes music using a neural net."""

import nnmusic.defaults as _defaults
import nnmusic.types    as _types
import nnmusic.io       as _io
import tensorflow       as _tf

N_HIDDEN   = 24
BATCH_SIZE = 100

def train(records_file_name, state_file_name,
          n_epochs=_defaults.DEFAULT_EPOCHS,
          n_channels=_defaults.DEFAULT_CHANNELS,
          n_batch_threads=_defaults.DEFAULT_THREADS,
          n_train_threads=_defaults.DEFAULT_THREADS):
    """Train a neural net.
    
    Keyword arguments:
        records_file_name -- Name of TFRecords file containing training
                             examples.
        state_file_name   -- Name of file to write neural-net state to.
        n_epochs          -- Number of training epochs.
        n_channels        -- Number of audio channels.
        n_batch_threads   -- Number of threads for enqueueing training
                             examples.
        n_train_threads   -- Number of threads for executing training ops.
    """
    file_name, audio_data = _io.read_record(records_file_name, n_epochs,
                                            n_channels)
    
    cell = _tf.nn.rnn_cell.LSTMCell(N_HIDDEN)
    
    batch = _tf.contrib.training.batch_sequences_with_states(
        file_name, {"audio_data": audio_data}, {}, None, {
            "lstm_state" : _tf.zeros(cell.state_size[0],
                                     dtype=_types.tensor_amplitude),
            "lstm_hidden": _tf.zeros(cell.state_size[1],
                                     dtype=_types.tensor_amplitude)
        }, 10000, BATCH_SIZE, n_batch_threads, BATCH_SIZE
    )
    
    n_channels = audio_data.get_shape()[1]
    
    batched_data = _tf.unpack(batch.sequences["audio_data"])
    
    output = _tf.concat(
        0, _tf.nn.state_saving_rnn(cell, batched_data, state_saver=batch,
                                   state_name=("lstm_state", "lstm_hidden"))[0]
    )[:-1, :]
    
    rms_error = _tf.sqrt(
        _tf.reduce_sum(
            _tf.squared_difference(
                _tf.matmul(
                    output, _tf.Variable(
                        _tf.truncated_normal(
                            (N_HIDDEN, n_channels),
                            dtype=_types.tensor_amplitude
                        ), dtype=_types.tensor_amplitude
                    )
                ) + _tf.tile(
                    _tf.Variable(
                        _tf.constant(0.0, shape=(1, n_channels),
                                     dtype=_types.tensor_amplitude),
                        dtype=_types.tensor_amplitude
                    ), (output.get_shape()[0], 1)
                ), _tf.concat(0, batched_data)[1:, :]
            )
        )
    )

    minimize = _tf.train.AdamOptimizer().minimize(rms_error)
    
    saver = _tf.train.Saver()
    
    with _tf.Session(
        config=_tf.ConfigProto(intra_op_parallelism_threads=n_train_threads)
    ) as s:
        s.run(_tf.initialize_all_variables())
        
        coord   = _tf.train.Coordinator()
        threads = _tf.start_queue_runners(sess=s, coord=coord)
        
        while not coord.should_stop():
            _io.print_now("RMS error = "
                          "{}.".format(s.run([minimize, rms_error])[1]))

        coord.request_stop()
        coord.join(threads)

        saver.save(s, state_file_name)
