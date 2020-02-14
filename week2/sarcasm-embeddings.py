#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import json

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plot

# hyper parameters
vocab_size = 10000
embedded_dimensions = 16
max_seq_length = 50
trunc_type = "post"
padding_type = "post"
training_size = 20000
oov_token = "<OOV>"
num_training_epochs = 30

with open( "data/sarcasm.json" ) as file:
    content = json.load( file )

training_data = content[ 0: training_size ]
testing_data = content[ training_size: len( content ) ]

training_sentences = [ ]
training_labels = [ ]
for item in training_data:
    training_sentences.append( item[ 'headline' ] )
    training_labels.append( item[ 'is_sarcastic' ] )

test_sentences = [ ]
test_labels = [ ]
for item in testing_data:
    test_sentences.append( item[ 'headline' ] )
    test_labels.append( item[ 'is_sarcastic' ] )

tokenizer = Tokenizer( num_words = vocab_size, oov_token = oov_token )

tokenizer.fit_on_texts( training_sentences )

training_sequences = pad_sequences( tokenizer.texts_to_sequences( training_sentences ), maxlen = max_seq_length, padding = padding_type, truncating = trunc_type )
test_sequences = pad_sequences( tokenizer.texts_to_sequences( test_sentences ), maxlen = max_seq_length, padding = padding_type, truncating = trunc_type )

model = tf.keras.Sequential( [
    tf.keras.layers.Embedding( vocab_size, embedded_dimensions, input_length = max_seq_length ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense( 6, activation = 'relu' ),
    tf.keras.layers.Dense( 1, activation = 'sigmoid' )
] )
model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )

results = model.fit( x = training_sequences, y = training_labels, epochs = num_training_epochs, validation_data = (test_sequences, test_labels) )


def plot_results( history, label ):
    plot.plot( history.history[ label ] )
    plot.plot( history.history[ f"val_{label}" ] )
    plot.xlabel( "Epochs" )
    plot.ylabel( label )
    plot.show()


plot_results( results, 'acc' )
plot_results( results, 'loss' )