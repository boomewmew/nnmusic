# io.py
# Author: Boo Mew Mew

"""Reads and writes audio files."""

import soundfile

from nnmusic.types import Amplitude

from os import listdir

from sys import stderr

class SampleRateError(Exception):
    """Exception indicating unexpected sample rate.
    
    Attributes:
        sample_rate   -- Actual sample rate of file.
        expected_rate -- Desired sample rate.
    """
    
    def __init__(self, file_name, sample_rate, expected_rate):
        """Constructor."""
        self.file_name     = file_name
        self.sample_rate   = sample_rate
        self.expected_rate = expected_rate
        
    def __str__(self):
        """Error message."""
        return "File {} has sample rate {} Hz. Expected {} Hz.".format(
            self.file_name, 
            self.sample_rate,
            self.expected_rate
        )

class ChannelError(Exception):
    """Exception indicating bad number of channels.
    
    Attributes:
        n_channels        -- Number of channels in the file.
        expected_channels -- Desired number of channels.
    """
    
    def __init__(self, file_name, n_channels, expected_channels):
        """Constructor."""
        self.n_channels        = n_channels
        self.expected_channels = expected_channels
        self.file_name         = file_name
    
    def __str__(self):
        """Error message."""
        return "File {} has {} channels. Expected {}.".format(
            self.file_name,
            self.n_channels,
            self.expected_channels
        )

DEFAULT_RATE     = 44100
DEFAULT_CHANNELS = 2

def read(file_name,
         expected_rate     = DEFAULT_RATE,
         expected_channels = DEFAULT_CHANNELS):
    """Read an audio file and return the data as a tensor.
    
    Keyword arguments:
        file_name         -- Name of input file.
        expected_rate     -- Throws exception if file's sample rate in Hz
                             differs from this value.
        expected_channels -- Throws exception if file contains number of
                             channels differing from this value.
    
    Return value:
        A 2D array of amplitudes. The first index runs over audio channels. The
        second runs over time steps.
    """
    data, sample_rate = soundfile.read(
        file_name,
        dtype     = Amplitude,
        always_2d = True
    )
    
    if sample_rate != expected_rate:
        raise SampleRateError(file_name, sample_rate, expected_rate)
    
    n_channels = data.shape[0]
    if n_channels != expected_channels:
        raise ChannelError(file_name, n_channels, expected_channels)
        
    return data

def write(file_name, data, sample_rate = DEFAULT_RATE):
    """Write an audio file.
    
    Keyword arguments:
        file_name   -- Name of output file.
        data        -- A 2D array of amplitudes. The first index runs over audio
                       channels. The second runs over time steps.
        sample_rate -- Sample rate in Hz.
    """
    soundfile.write(file_name, data, sample_rate)

def read_dir(dir_name,
             expected_rate     = DEFAULT_RATE,
             expected_channels = DEFAULT_CHANNELS):
    """Read audio files from a directory.
    
    The list of files present in the directory is parsed at the beginning of the
    generator's execution. If a file is removed before it is read, the file is
    skipped, with a warning printed to stderr.
    
    If a file does not have the desired sample rate or number of channels, it is
    likewise skipped with a warning sent to stderr.
    
    Keyword arguments:
        dir_name          -- Directory to be read from.
        expected_rate     -- Desired sample rate in Hz.
        expected_channels -- Desired number of audio channels.
    """
    err_stream = stderr
    
    for s in listdir(dir_name):
        file_name = "{}/{}".format(dir_name, s)

        try:
            yield read(file_name)
        except SampleRateError as e:
            print(
                "File {} has sample rate {} Hz (wanted {} Hz). Skipping."
                    .format(file_name, e.sample_rate, expected_rate),
                file = err_stream
            )
        except ChannelError as e:
            print(
                "File {} has {} channels (wanted {}). Skipping.".format(
                    file_name,
                    e.n_channels,
                    expected_channels
                ),
                file = err_stream
            )
        except RuntimeError:
            print(
                "File {} was removed from directory {}. Skipping.".format(
                    s,
                    dir_name
                ),
                file = err_stream
            )
