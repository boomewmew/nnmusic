# io.py
# Author: Boo Mew Mew

import soundfile

from nnmusic.types import Amplitude

"""Reads and writes audio files."""

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

DEFAULT_RATE = 44100

def read(file_name, expected_rate = DEFAULT_RATE, expected_channels = 2):
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
