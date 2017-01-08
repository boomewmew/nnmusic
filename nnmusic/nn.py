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

import h5py          as _h5py
import nnmusic.io    as _io
import nnmusic.types as _types
import pickle        as _pickle
import random        as _random
import tensorflow    as _tf
import tflearn       as _tfl
import time          as _time

DEFAULT_EPOCHS  = 5000
DEFAULT_THREADS = 1

class Composer:
    """Uses LSTM neural net to compose music."""
    
    def __init__(self, dictionary, log_dir, chunk_size=_io.DEFAULT_CHUNK_SIZE,
                 n_channels=_io.DEFAULT_CHANNELS):
        """Constructor.
        
        Keyword arguments:
            dictionary -- Dictionary mapping each amplitude value from the
                          dataset to a unique integer.
            log_dir    -- Directory for log files.
            chunk_size -- Number of time steps per chunk.
            n_channels -- Number of audio channels.
        """
        _io.print_now("Initializing neural net.")
        
        net = _tfl.input_data((None, chunk_size, n_channels),
                              dtype=_types.tensor_amplitude)
        net = _tfl.lstm(net, n_channels, return_seq=True)
        net = _tf.pack(net, 1)
        net = _tfl.regression(net, loss=_tfl.mean_square, metric=_tfl.R2(),
                              dtype=_types.tensor_amplitude)
                              
        local_dict = locals()

        self._generator = _tfl.SequenceGenerator(net, dictionary, chunk_size,
                                                 tensorboard_verbose=3,
                                                 tensorboard_dir=log_dir)
        self._cucumbers = [local_dict[s] for s in self._CUCUMBER_NAMES]

    _CUCUMBER_NAMES = __init__.__code__.co_varnames[1:]

    def save(self, model_file_name, pickle_file_name):
        """Store Composer state.
        
        The Composer object is not actually pickleable because its _generator
        attribute is not pickleable. Instead, the model state is stored in a
        model file, and other information required to restore the Composer
        state is pickled.
        
        Keyword argument:
            model_file_name  -- Name of file to write neural net model into.
            pickle_file_name -- Name of file to serialize auxiliary data into.
        """
        message = "Writing neural-net state to files {} and {}.".format(
            model_file_name, pickle_file_name
        )
        _io.print_now(message)
        
        with open(pickle_file_name, "wb") as f:
            for o in self._cucumbers:
                _pickle.dump(o, f)
    
        with open(model_file_name, "wb") as f:
            self._generator.save(f)
        
    @staticmethod
    def load(model_file_name, pickle_file_name):
        """Load Composer from model and pickle files.
        
        The model state is loaded from the model file, and the other necessary
        objects to restore the state of the Composer are deserialized from the
        pickle file and used to construct the Composer object.
        
        Keyword argument:
            model_file_name  -- Name of model file.
            pickle_file_name -- Name of pickle file.
        
        Return value:
            The Composer.
        """
        message = "Reading neural-net state from files {} and {}.".format(
            model_file_name, pickle_file_name
        )
        _io.print_now(message)
        
        with open(pickle_file_name, "rb") as f:
            composer = Composer(_pickle.load(f)
                                for s in Composer._CUCUMBER_NAMES)
            
        with open(model_file_name, "rb") as f:
            composer.load(f)
        
        return composer
        
    def train(self, dataset, n_epochs=DEFAULT_EPOCHS):
        """Train the contained neural net.
        
        Keyword arguments:
            dataset  -- Training data. Array of amplitudes. Shape is
                        (n_chunks, n_time_steps, n_channels).
            n_epochs -- Number of training epochs.
        """
        _io.print_now("Training neural net.")
        
        self._generator.fit(dataset[:, :-1, :], dataset[:, 1:, :], n_epochs,
                            validation_set=0.5, show_metric=True,
                            run_id=_time.strftime("%Y-%m-%d-%H:%M:%S"))

    def compose(self, duration, seed):
        """Compose a piece of music using the neural net.
        
        Keyword arguments:
            duration -- Number of time steps to compose for.
            seed     -- Seed sequence. Array of amplitudes. Shape is
                        (n_time_steps, n_channels).

        Return value:
            The composition. Array of amplitudes. Shape is
            (duration, n_channels).
        """
        _io.print_now("Composing.")
        
        return self._generator.generate(duration, seq_seed=seed)

def train(hdf5_file_name, model_file_name, pickle_file_name, log_dir,
          n_threads=DEFAULT_THREADS, n_epochs=DEFAULT_EPOCHS):
    """Train an LSTM recurrent neural network for writing music.
    
    Keyword arguments:
        hdf5_file_name   -- Name of HDF5 file containing audio data for
                            training.
        model_file_name  -- Name of file to store model weights in.
        pickle_file_name -- Name of file for pickling generator attributes.
        log_dir          -- Directory for writing logs.
        n_threads        -- Number of threads to run.
        n_epochs         -- Number of training epochs.
    """
    _io.print_now("Reading training data from file {}.".format(hdf5_file_name))
    
    with _h5py.File(hdf5_file_name, "r") as f:
        dataset    = f[_io.DATASET_NAME]
        data_shape = dataset.shape
        
        dictionary = {
            x: i
            for i, x in enumerate({y for a in dataset for b in a for y in b})
        }
        
        composer = Composer(dictionary, log_dir, data_shape[1] - 1,
                            data_shape[2])
        composer.train(dataset, n_epochs)
        
    composer.save(model_file_name, pickle_file_name)

def compose(model_file_name, pickle_file_name, audio_file_name, duration,
            hdf5_file_name, sample_rate=_io.DEFAULT_RATE):
    """Compose a piece of music and write the result to an audio file.
    
    Keyword arguments:
        model_file_name  -- Name of file to write neural-net state into.
        pickle_file_name -- Name of file for pickling generator.
        audio_file_name  -- Output audio file to write music into.
        duration         -- Number of time steps of composed music.
        hdf5_file_name   -- HDF5 file from which a random chunk will be
                            selected as a seed.
        sample_rate      -- Sampling frequency in Hz.
    """
    message = "Seeding composition with random chunk from file {}.".format(
        hdf5_file_name
    )
    _io.print_now(message)
    
    composer = Composer.load(model_file_name, pickle_file_name)
    
    with _h5py.File(hdf5_file_name, "r") as f:
        _io.write_audio(
            audio_file_name,
            composer.compose(duration, _random.choice(f[_io.DATASET_NAME])),
            sample_rate
        )
