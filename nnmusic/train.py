# train.py
# Author: Boo Mew Mew

"""Neural-net training."""

import tensorflow as tf

def train(train_dir, valid_dir, test_dir):
    """Train an LSTM recurrent neural network to write music.
    
    Keyword argument:
        train_dir -- Directory containing audio files for training.
        valid_dir -- Directory containing audio files for validation.
        test_dir  -- Directory containing audio files for testing.
    """
