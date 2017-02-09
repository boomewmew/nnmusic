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

"""Performs reading and writing of audio files and log formatting."""

import nnmusic.defaults as _defaults
import nnmusic.types    as _types
import os               as _os
import soundfile        as _sf
import sys              as _sys
import tensorflow       as _tf

STR_ENCODING = "utf-8"

class ANSICodeError(Exception):
    """Exception indicating an unknown ANSI color code."""
    
    def __init__(self, color):
        """Constructor.
        
        Attribute:
            color -- Name of unrecognized color.
        """
        self.color = color
    
class ANSICode:
    """One or more ANSI color or text-formatting codes."""
    
    _COLOR_DICT = {"black": 30, "red": 31, "green": 32, "yellow": 33,
                   "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
                   "default": 39}

    @staticmethod
    def _get_color(color, bg=False):
        """Get a foreground color code.
        
        Keyword argument:
            color  -- Name of desired color or None.
            bg     -- Boolean value.  True indicates background color.
        
        Return value:
            The integer color code.
        """
        if color is None:
            return None
        
        try:
            ret = ANSICode._COLOR_DICT[color]
        except KeyError:
            raise ANSICodeError(color)
    
        if bg:
            ret += 10

        return ret

    def __init__(self, fg_color=None, bg_color=None, bright=None, italics=None,
                 underline=None, inverse=None, strikethrough=None,
                 reset=False):
        """Constructor.
        
        Keyword arguments:
            fg_color      -- Name of foreground color.
            bg_color      -- Name of background color.
            bright        -- Turn on/off bright colors.
            italics       -- Turn on/off italics.
            underline     -- Turn on/off underlining.
            inverse       -- Turn on/off inverting of foreground and background
                             colors.
            strikethrough -- Turn on/off strikethrough.
            reset         -- Reset to default style.  All other options are
                             ignored.
        """
        self._reset = reset
        if reset:
            return
        
        self._fg_color = self._get_color(fg_color)
        self._bg_color = self._get_color(bg_color, True)
        
        self._bright        = bright
        self._italics       = italics
        self._underline     = underline
        self._inverse       = inverse
        self._strikethrough = strikethrough
    
    @staticmethod
    def _encode(code):
        """Format an integer value for printing."""
        return "\033[{}m".format(code)
    
    def __str__(self):
        """Get escape code."""
        if self._reset:
            return self._encode(0)
        
        ret = ""
        
        for s in ("f", "b"):
            color = getattr(self, "_{}g_color".format(s))
            if color is not None:
                ret += self._encode(color)

        for s, n, m in (("bright", 1, 22), ("italics", 3, 23),
                        ("underline", 4, 24), ("inverse", 7, 27),
                        ("strikethrough", 9, 29)):
            attr = getattr(self, "_{}".format(s))
            if attr is not None:
                ret += self._encode(n if attr else m)
        
        return ret

ANSI_NONE  = ANSICode()
ANSI_RESET = ANSICode(reset=True)
ANSI_OK    = ANSICode("green")
ANSI_WARN  = ANSICode("yellow")
ANSI_ERR   = ANSICode("red", bright=True)

def wrap_ansi(message, ansi_code):
    """Format text with an ANSI code."""
    return "{}{}{}".format(ansi_code, message, ANSI_RESET)

def wrap_err(message):
    """Format text as an error message."""
    return wrap_ansi(message, ANSI_ERR)

def wrap_warn(message):
    """Format text as a warning."""
    return wrap_ansi(message, ANSI_WARN)

setattr(ANSICodeError, "__str__",
        lambda e: wrap_err("Unrecognized color {}.".format(e.color)))

def print_now(message, stream=_sys.stdout, ansi_code=ANSI_OK):
    """Print a message to a stream and immediately flush the stream.
    
    Keyword arguments:
        message   -- The text to print.
        stream    -- The output stream to print to.
        ansi_code -- If provided, message is printed with this formatting, and
                     default formatting is restored at the end.
    """
    if ansi_code is not None:
        message = wrap_ansi(message, ansi_code)
    
    print(message, file=stream)
    
    stream.flush()

def print_err_now(message):
    """Print an error message and immediately flush the stream.
    
    Keyword argument:
        message -- The text to print.
    """
    print_now(message, _sys.stderr, ANSI_ERR)

def print_warn_now(message):
    """Print a warning and immediately flush the stream.
    
    Keyword argument:
        message -- The text to print.
    """
    print_now(message, _sys.stderr, ANSI_WARN)

class DirNotFoundError(Exception):
    """Exception indicating a directory could not be found.
    
    Attributes:
        dir_name -- Name of directory.
    """
    
    def __init__(self, dir_name):
        """Constructor."""
        self.dir_name = dir_name
    
    def __str__(self):
        """Error message."""
        return wrap_err("Directory {} does not exist.".format(self.dir_name))

class SoundFileNotFoundError(Exception):
    """Exception indicating a file could not be found for opening.
    
    Attributes:
        file_name -- Name of file.
    """
    
    def __init__(self, file_name):
        """Constructor."""
        self.file_name = file_name
    
    def __str__(self):
        """Error message."""
        return wrap_err("File {} does not exist.".format(self.file_name))

class InvalidFileError(Exception):
    """Exception indicating a file exists but is not a valid audio file.
    
    Attributes:
        file_name -- Name of file.
    """
    
    def __init__(self, file_name):
        """Constructor."""
        self.file_name = file_name
    
    def __str__(self):
        """Error message."""
        return wrap_err("File {} is not a valid audio "
                        "file.".format(self.file_name))

class SampleRateError(Exception):
    """Exception indicating unexpected sample rate.
    
    Attributes:
        file_name     -- Name of file.
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
        return wrap_err(
            "File {} has sample rate {} Hz.  Expected {} "
            "Hz.".format(self.file_name, self.sample_rate, self.expected_rate)
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
        return wrap_err(
            "File {} has {} channels.  Expected "
            "{}.".format(self.file_name, self.n_channels,
                         self.expected_channels)
        )

def read_audio(file_name, expected_rate=_defaults.DEFAULT_RATE,
               expected_channels=_defaults.DEFAULT_CHANNELS):
    """Read an audio file and return the data as an array.
    
    Keyword arguments:
        file_name         -- Name of input file.
        expected_rate     -- Throws exception if file's sample rate in Hz
                             differs from this value.
        expected_channels -- Throws exception if file contains number of
                             channels differing from this value.
    
    Return value:
        A 2D array of amplitudes.  The zeroth index runs over time steps.  The
        first runs over audio channels.
    """
    if not _os.path.exists(file_name):
        raise SoundFileNotFoundError(file_name)
    
    print_now("Reading file {}.".format(file_name))
    
    try:
        data, sample_rate = _sf.read(file_name, dtype=_types.amplitude,
                                     always_2d=True)
    except RuntimeError:
        raise InvalidFileError(file_name)
    
    if sample_rate != expected_rate:
        raise SampleRateError(file_name, sample_rate, expected_rate)
    
    n_channels = data.shape[1]
    if n_channels != expected_channels:
        raise ChannelError(file_name, n_channels, expected_channels)
    
    return data

def read_audio_no_throw(file_name, expected_rate=_defaults.DEFAULT_RATE,
                        expected_channels=_defaults.DEFAULT_CHANNELS):
    """Read an audio file and return the data as an array, ignoring exceptions.
    
    If the file cannot be opened or has an invalid sample rate or number of
    channels, a warning is printed to stderr, and None is returned.
    
    Keyword arguments:
        file_name         -- Name of input file.
        expected_rate     -- Throws exception if file's sample rate in Hz
                             differs from this value.
        expected_channels -- Throws exception if file contains number of
                             channels differing from this value.
    
    Return value:
        If the file is read successfully, a 2D array of amplitudes is returned.
        The zeroth index runs over time steps.  The first runs over audio
        channels.
        
        If the read fails, None is returned.
    """
    try:
        return read_audio(file_name, expected_rate, expected_channels)
    except SoundFileNotFoundError:
        print_warn_now("File {} does not exist.  Skipping.".format(file_name))
    except InvalidFileError:
        print_warn_now("File {} could not be read.  "
                       "Skipping.".format(file_name))
    except SampleRateError as e:
        print_warn_now(
            "File {} has sample rate {} Hz.  Expected {} Hz.  "
            "Skipping.".format(file_name, e.sample_rate, expected_rate)
        )
    except ChannelError as e:
        print_warn_now(
            "File {} has {} channels.  Expected {}.  "
            "Skipping.".format(file_name, e.n_channels, expected_channels)
        )

def write_audio(file_name, data, sample_rate=_defaults.DEFAULT_RATE):
    """Write an audio file.
    
    Keyword arguments:
        file_name   -- Name of output file.
        data        -- A 2D array of amplitudes.  The zeroth index runs over
                       time steps.  The first runs over audio channels.
        sample_rate -- Sample rate in Hz.
    """
    print_now("Writing file {}.".format(file_name))
    
    _sf.write(file_name, data, sample_rate)

def audio_to_records(audio_dir, records_file_name,
                     expected_rate=_defaults.DEFAULT_RATE,
                     expected_channels=_defaults.DEFAULT_CHANNELS):
    """Convert several audio files to a TFRecords file.
    
    Keyword arguments:
        audio_dir         -- Directory containing audio files.
        records_file_name -- Name of TFRecords file to write.
        expected_rate     -- Expected sample rate in Hz.
        expected_channels -- Expected number of audio channels.
    """
    print_now("Converting audio files in directory {} to TFRecords file "
              "{}.".format(audio_dir, records_file_name))
    
    with _tf.python_io.TFRecordWriter(records_file_name) as w:
        for s in _os.listdir(audio_dir):
            file_name = _os.path.join(audio_dir, s)

            example = _tf.train.SequenceExample()
            example.context.feature["file_name"].bytes_list.value.append(
                file_name.encode(STR_ENCODING)
            )
            audio_data = example.feature_lists.feature_list["audio_data"]
            for x in read_audio(file_name, expected_rate,
                                expected_channels).reshape((-1,)):
                getattr(audio_data.feature.add(),
                        _types.AMPLITUDE_LIST_ARG_NAME).value.append(x)

            w.write(example.SerializeToString())

def read_record(records_file_name, n_epochs=_defaults.DEFAULT_EPOCHS,
                n_channels=_defaults.DEFAULT_CHANNELS):
    """Read a record from a TFRecords file.
    
    Keyword arguments:
        records_file_name -- TFRecords file to read.
        n_epochs          -- Number of training epochs.
        n_channels        -- Number of audio channels.
    
    Return value:
        A tuple whose zeroth element is the name of the original sound file
        used to produce the example and whose first element is a 2D array of
        amplitudes with shape (duration, n_channels).
    """
    print_now("Reading audio data from TFRecords file "
              "{}.".format(records_file_name))

    context, sequence = _tf.parse_single_sequence_example(
        serialized=_tf.TFRecordReader().read(
            _tf.train.string_input_producer([records_file_name],
                                            num_epochs=n_epochs)
        )[1],
        context_features={
            "file_name": _tf.FixedLenFeature((), dtype=_tf.string)
        },
        sequence_features={
            "audio_data": _tf.FixedLenSequenceFeature((),
                                                      dtype=_types.amplitude)
        }
    )
    
    return (context["file_name"],
            _tf.reshape(sequence["audio_data"], (-1, n_channels)))
