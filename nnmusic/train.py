# train.py
# Author: Boo Mew Mew

"""Neural-net training."""

import math as _math

import nnmusic.io    as _io
import nnmusic.types as _types

import numpy as _np

import tensorflow as _tf

def _trim(wave):
    return wave[1:, :]

def train_and_test(train_dir, test_dir, batch_size, out_file_name,
                   sample_rate=_io.DEFAULT_RATE,
                   n_channels=_io.DEFAULT_CHANNELS, n_hidden=24,
                   n_epochs=5000):
    """Train and test an LSTM recurrent neural network for writing music.
    
    Keyword arguments:
        train_dir     -- Directory containing audio files for training.
        test_dir      -- Directory containing audio files for testing.
        batch_size    -- Number of audio files per batch.
        out_file_name -- File to write neural-net state to.
        sample_rate   -- Expected sample rate in Hz of audio files.
        n_channels    -- Expected number of audio channels.
        n_hidden      -- Number of nodes per hidden layer.
        n_epochs      -- Number of training epochs.
    """
    data_shape = (None, None, n_channels)
    data       = _tf.placeholder(_types.tensor_amplitude, data_shape)
    target     = _tf.placeholder(_types.tensor_amplitude, data_shape)
    
    weight = _tf.Variable(_tf.truncated_normal((n_hidden, n_channels)),
                          dtype=_types.tensor_amplitude)
    bias   = _tf.Variable(_tf.constant(0.1, shape=(n_channels,)),
                          dtype=_types.tensor_amplitude)
    
    nonzero = _tf.to_int32(
        _tf.not_equal(target, _tf.constant(0., dtype=_types.tensor_amplitude))
    )
    n_nonzero         = _tf.reduce_sum(nonzero)
    mean_square_error = _tf.reduce_sum(
        _tf.squared_difference(
            _tf.einsum(
                "ijk,kl->ijl",
                _tf.nn.dynamic_rnn(
                    _tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True),
                    data, dtype=_types.tensor_amplitude
                )[0], weight
            ) + bias, target
        ) * nonzero
    ) / n_nonzero
    
    minimize  = _tf.train.AdamOptimizer().minimize(mean_square_error)
    
    saver = _tf.train.Saver()
    
    sess    = _tf.Session()
    sess.run(_tf.initialize_all_variables())
    
    def run(obj, batch):
        if len(batch):
            zeros = _np.zeros((1, batch[0].shape[1]), _types.amplitude)
        return sess.run(
            obj,
            {data  : batch,
             target: [_np.vstack((_trim(a), zeros)) for a in batch]}
        )
    
    for i in range(n_epochs):
        for l in _io.read_dir(train_dir, batch_size, sample_rate, n_channels):
            run(minimize, l)

    sum_sq     = _types.amplitude(0.)
    sum_points = 0
    for l in _io.read_dir(test_dir, batch_size, sample_rate, n_channels):
        n_points    = run(n_nonzero, l)
        sum_sq     += run(mean_square_error, l) * n_points
        sum_points += n_points
    print(
        "RMS error on testing sample = {}.".format(
            _math.sqrt(sum_sq / sum_points)
        )
    )
    
    saver.save(sess, out_file_name)
