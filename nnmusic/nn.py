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

N_HIDDEN    = 24
N_UNROLL    = 10
BATCH_SIZE  = 10
TIME_OFFSET = 1

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
    _io.print_now("Training neural net.")
    
    file_name, audio_data = _io.read_record(records_file_name, n_epochs,
                                            n_channels)
    
    cell = _tf.nn.rnn_cell.LSTMCell(N_HIDDEN)
    
    batch = _tf.contrib.training.batch_sequences_with_states(
        file_name, {"audio_data": audio_data}, {}, None, {
            "lstm_state" : _tf.zeros(cell.state_size[0],
                                     dtype=_types.tensor_amplitude),
            "lstm_hidden": _tf.zeros(cell.state_size[1],
                                     dtype=_types.tensor_amplitude)
        }, N_UNROLL, BATCH_SIZE, n_batch_threads, BATCH_SIZE
    )
    
    batched_data = batch.sequences["audio_data"]
    
    batch_size   = batch.batch_size
    n_time_steps = N_UNROLL * batch_size
    
    rms_error = _tf.sqrt(
        _tf.reduce_sum(
            _tf.squared_difference(
                _tf.matmul(
                    _tf.reshape(
                        _tf.stack(
                            _tf.nn.state_saving_rnn(
                                cell, _tf.unstack(batched_data, axis=1),
                                state_saver=batch,
                                state_name=("lstm_state", "lstm_hidden")
                            )[0], 1
                        ), (n_time_steps, N_HIDDEN)
                    )[:-TIME_OFFSET, :],
                    _tf.Variable(
                        _tf.truncated_normal((N_HIDDEN, n_channels),
                                             dtype=_types.tensor_amplitude),
                        name="weights", dtype=_types.tensor_amplitude
                    )
                ) + _tf.tile(
                    _tf.Variable(
                        _tf.constant(0.0, shape=(1, n_channels),
                                     dtype=_types.tensor_amplitude),
                        name="biases", dtype=_types.tensor_amplitude
                    ), (batch_size - TIME_OFFSET, 1)
                ),
                _tf.reshape(batched_data, (n_time_steps, n_channels))[
                    TIME_OFFSET:, :
                ]
            )
        )
    )
    
    minimize = _tf.train.AdamOptimizer().minimize(rms_error)
    
    with _tf.Session(
        config=_tf.ConfigProto(intra_op_parallelism_threads=n_train_threads)
    ) as s:
        s.run(_tf.global_variables_initializer())
        s.run(_tf.local_variables_initializer ())
        
        coord   = _tf.train.Coordinator()
        threads = _tf.train.start_queue_runners(sess=s, coord=coord)
        
        epoch = 0
        try:
            while not coord.should_stop():
                _io.print_now(
                    "Epoch {}.  RMS error = "
                    "{}.".format(epoch, s.run([minimize, rms_error])[1])
                )
                epoch += 1
        except _tf.errors.OutOfRangeError:
            pass

        coord.request_stop()
        coord.join(threads)

        _tf.train.Saver().save(s, state_file_name)
