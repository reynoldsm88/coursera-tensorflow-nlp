#!/usr/bin/env python3

import random
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sys import argv
from os.path import dirname

with open( f"{dirname( argv[ 0 ] )}/sarcasm.json" ) as file:
    data = json.loads( file.read() )
    for entry in data:
        del entry[ 'article_link' ]

records = [ ]
for item in data:
    records.append( (item[ 'headline' ], item[ 'is_sarcastic' ]) )

print( records )

train_set = records[ 0:20000 ]
test_set = records[ 19999: -1 ]
# setting the number means that the tokenizer will take the top num_words, this is an important hyperparamter
# that can affect accuracy and training time
tokenizer = Tokenizer( oov_token = "<OOV>" )

training_sentences = [ ]
training_labels = [ ]
for record in train_set:
    sentence, label = record
    training_sentences.append( sentence )
    training_labels.append( label )

tokenizer.fit_on_texts( training_sentences )

seq = tokenizer.texts_to_sequences( training_sentences )
# default is to prepend 0's, so padding = post adds 0's to the end, truncating chooses the strategy for where to lost information
# if a sentence of a different size is encountered in the test_data
padded = pad_sequences( seq, padding = "post", truncating = "post" )
print( padded )