# nnmusic, a program for composing music using artificial neural networks.
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

import sys as _sys

class Format:
    """ANSI codes for formatting terminal output."""
    
    def __init__(self, code=None):
        """Initialize format object.
        
        code: Numeric code or iterable of numeric codes.
        """
        try:
            self._codes = set() if code is None else set(code)
        except TypeError:
            self._codes = set((code,))
    
    def __add__(self, other):
        """Combine two formats.
        
        other: Another format object.
        """
        return Format(self._codes.union(other._codes))
        
    def __str__(self):
        """Convert to string."""
        return "".join("\033[{}m".format(code) for code in self._codes)

RESET = Format(0)
BOLD  = Format(1)
RED   = Format(31)
GREEN = Format(32)
ERR   = BOLD + RED

DEFAULT_OUTPUT = ""

def print_now(output=DEFAULT_OUTPUT, form=GREEN, stream=_sys.stdout):
    """Print and immediately flush output stream.
    
    output: Text or object to print.
    form:   Text format.
    stream: Output stream.
    """
    print("{}{}{}".format(form, output, RESET), file=stream)
    stream.flush()

def print_err(output=DEFAULT_OUTPUT, form=ERR):
    """Print error and immediately flush output stream.
    
    output: Text or object to print.
    form:   Text format.
    """
    print_now(output, form, _sys.stderr)
