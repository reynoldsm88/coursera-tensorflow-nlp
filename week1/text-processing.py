#!/usr/bin/env python3

import random
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

data = [ ]
with open( "sarcasm.json" ) as file:
    data = json.loads( file.read() )
    for entry in data:
        del entry[ 'article_link' ]

sentences = [ ]
for item in data:
    sentences.append( (item[ 'headline' ], item[ 'is_sarcastic' ]) )

train_set = sentences[ 0:20000 ]
test_set = sentences[ 19999: -1 ]
# setting the number means that the tokenizer will take the top num_words, this is an important hyperparamter
# that can affect accuracy and training time
tokenizer = Tokenizer( oov_token = "<OOV>" )
tokenizer.fit_on_texts( train_set )
word_index = tokenizer.word_index

test_sentences = [
    "I love my cat",
    "I love my dog",
    "You love my dog!",
    "Do you think my dog is amazing?",
]

seq = tokenizer.texts_to_sequences( test_sentences )
# default is to prepend 0's, so padding = post adds 0's to the end, truncating chooses the strategy for where to lost information
# if a sentence of a different size is encountered in the test_data
padded = pad_sequences( seq, padding = "post", truncating = "post" )
print( word_index )
print( padded )