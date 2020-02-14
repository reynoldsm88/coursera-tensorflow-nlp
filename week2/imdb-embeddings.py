#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import random
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

from os.path import exists
from os import remove

tf.enable_eager_execution()
write_word_index = False

imdb, info = tfds.load( "imdb_reviews", with_info = True, as_supervised = True )

train, test = imdb[ 'train' ], imdb[ 'test' ]

training_sentences = [ ]
training_labels = [ ]
for sentence, label in train:
    training_sentences.append( str( sentence.numpy() ) )
    training_labels.append( label.numpy() )
# tensorflow requires a numpy array
training_labels = np.array( training_labels )

testing_sentences = [ ]
testing_labels = [ ]
for sentence, label in test:
    testing_sentences.append( str( sentence.numpy() ) )
    testing_labels.append( label.numpy() )
# tensorflow requires a numpy array
testing_labels = np.array( testing_labels )

# hyperparameters
vocab_size = 90000
embedding_dim = 16
max_length = 120
oov_token = "<OOV>"
trunc_type = 'post'
padding_type = 'post'

tokenizer = Tokenizer( num_words = vocab_size, oov_token = oov_token )
tokenizer.fit_on_texts( training_sentences )

word_index = tokenizer.word_index

word_index_filename = "word_index.csv"

if exists( word_index_filename ):
    remove( word_index_filename )

if write_word_index is True:
    with open( word_index_filename, "w+" ) as file:
        for item in word_index.items():
            key, value = item
            file.write( f"{key},{value}\n" )

training_sequences = pad_sequences( tokenizer.texts_to_sequences( training_sentences ), maxlen = max_length, truncating = trunc_type, padding = padding_type )
testing_sequences = pad_sequences( tokenizer.texts_to_sequences( testing_sentences ), maxlen = max_length, truncating = trunc_type, padding = padding_type )

reverse_word_index = { }
for wi in word_index.items():
    word, num = wi
    reverse_word_index.update( { num: word } )

print( reverse_word_index )


def decode_review( text ):
    return ' '.join( [ reverse_word_index.get( i, '?' ) for i in text ] )


print( decode_review( training_sequences[ 1 ] ) )
print( training_sentences[ 1 ] )

model = tf.keras.Sequential( [
    tf.keras.layers.Embedding( vocab_size, embedding_dim, input_length = max_length ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense( 6, activation = 'relu' ),
    tf.keras.layers.Dense( 1, activation = 'sigmoid' )
] )

model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )
model.summary()

num_training_epochs = 10

model.fit( x = training_sequences,
           y = training_labels,
           epochs = num_training_epochs,
           validation_data = (testing_sequences, testing_labels) )