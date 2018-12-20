import tensorflow as tf
import numpy as np

import re
import collections 

import pickle

tf.enable_eager_execution()

"""
This performs the regular vectorization of the comment base if we are using a
character based model. It can also save and load previously computed vocabularies.

Inputs:
 - comments: A list of (string, int) pairs that represent a comment and score.
 - load: A boolean denoting whether or not we would like to load a previous
   vectorized vocabulary.
 - save: A boolean denoting whether or not we would like to save our vocab.
Returns:
 - vocab: A set of characters denoting our vocabulary. Note that this set of 
   chars can be outside the range of the traditional ASCII range. This is due to
   the fact that Reddit comments can cover a wide range of characters. For
   simplicity's sake, we read them with UTF-8 encoding.
 - char2idx: A dictionary of mappings between characters and integers.
 - idx2char: A numpy array of characters where the mapping is in the indices of 
   the list.
 - text_as_int: A numpy array of integers representing our vectorized comment
   base.
"""
def vectorize(comments, load_vocab=False, save_vocab=False):
    vocab, char2idx, idx2char, text_as_int = None, None, None, None
    if load_vocab:
        print("Loading from vocab file instead of vectorizing manually")
        with open("vocab.pkl", "rb") as input:
            packet = pickle.load(input)
            vocab, char2idx, idx2char, text_as_int = packet
            return vocab, char2idx, idx2char, text_as_int
    """
    Builds vocabulary in terms of characters. This methdology runs many orders
    of magnitudes faster than running a loop over the entire comment base. I
    don't know why. Probably some inner Pythonic black magic.
    """
    vocab = None
    text = "".join([comment[0] for comment in comments])
    vocab = sorted(set(text))

    """
    Creates two maps in the following formats:
    char2idx: {char: idx,...}
    idx2char: [char, char,...]
    """
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    """
    Performs the full vectorization of the comments by mapping each character to
    an integer.
    """
    text_as_int = np.array([char2idx[c] for c in text])

    print("Finished vectorizing")

    packet = (vocab, char2idx, idx2char, text_as_int)
    if save_vocab:
        print("Saving vocabulary file")
        with open("vocab.pkl", "wb") as output:
            pickle.dump(packet, output, pickle.HIGHEST_PROTOCOL)

    """
    Returns the structures that we've created.
    """
    return vocab, char2idx, idx2char, text_as_int

"""
This performs a modified vectorization of the comment base if we are using a
word based model. Much of the functionality of this is identical to that of
the method above.
Note: Within this method, 

Inputs:
 - comments: A list of (string, int) pairs that represent a comment and score.
Returns:
 - final_words: A list of strings that represent our word set.
 - char2idx: A dictionary of mappings between characters and integers.
 - idx2char: A numpy array of characters where the mapping is in the indices of 
   the list.
 - text_as_int: A numpy array of integers representing our vectorized comment
   base. 
"""
def vectorize_word(comments):
    print("Beginning to vectorize")

    words = []
    for comment in comments:
        words_in_comment = comment[0].lower().split()
        words += words_in_comment

    final_words = []
    counter = collections.Counter(words)
    for k,v in dict(counter).items():
        if v >= 5:
            final_words.append(k)

    char2idx = {u:i for i, u in enumerate(final_words)}
    idx2char = np.array(final_words)
    
    text_as_int = np.array([char2idx[c] for c in final_words])

    print("Finished vectorizing")
    return final_words, char2idx, idx2char, text_as_int

"""
This object represents the first model that we produced. It follows a simple
    Embedding - LSTM - Dense
architecture. This does produce decent results but we wanted to try a more
complicated architecture.
"""
class ModelOld(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(ModelOld, self).__init__()

        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        """
        If CUDA is available on the current machine, we use it.
        """
        if tf.test.is_gpu_available():
            self.LSTM1 = tf.keras.layers.CuDNNLSTM(self.units, 
                                                return_sequences=True, 
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        else:
            self.LSTM1 = tf.keras.layers.LSTM(self.units, 
                                            return_sequences=True, 
                                            recurrent_activation='sigmoid', 
                                            recurrent_initializer='glorot_uniform', 
                                            stateful=True)
        self.fc1 = tf.keras.layers.Dense(vocab_size)

    """
    Forward pass.
    """
    def call(self, x):
        embedding = self.embedding(x)
        output1 = self.LSTM1(embedding)
        fc1_out = self.fc1(output1)

        return fc1_out
"""
This is the second model that we used. This follows a much more complicated
model:
    Embedding w/ Dropout - LSTM1 - LSTM2
where the outputs of each layer is then concatenated together in another layer
which is then passed into a Dense layer.
We use a standard dropout probability of 0.5.
"""
class ModelTest(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, dropout=0.5):
        super(ModelTest, self).__init__()

        self.dropout = dropout
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
            self.LSTM1 = tf.keras.layers.CuDNNLSTM(self.units, 
                                                return_sequences=True, 
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
            self.LSTM2 = tf.keras.layers.CuDNNLSTM(self.units, 
                                                return_sequences=True, 
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        else:
            self.LSTM1 = tf.keras.layers.LSTM(self.units, 
                                            return_sequences=True, 
                                            recurrent_activation='sigmoid', 
                                            recurrent_initializer='glorot_uniform', 
                                            stateful=True)
            self.LSTM2 = tf.keras.layers.LSTM(self.units, 
                                            return_sequences=True, 
                                            recurrent_activation='sigmoid', 
                                            recurrent_initializer='glorot_uniform', 
                                            stateful=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    """
    Forward pass.
    """
    def call(self, x):
        embedding = self.embedding(x)

        if self.dropout > 0.0:
            embedding = tf.keras.layers.SpatialDropout1D(self.dropout, name='dropout')(embedding)

        output1 = self.LSTM1(embedding)
        output2 = self.LSTM2(output1)

        conc = tf.keras.layers.concatenate([embedding, output1, output2])
        output = self.dense(conc)

        return output
