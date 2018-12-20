from rnn import *
from preprocessor import Preprocessor

import tensorflow as tf
import numpy as np

import os
import sys
import time

class Runner(object):
    """
    The Runner class is the main class that will perform the entirety of the 
    training and the predicting.

    The constructor will take in all of the relevant information:
     - subreddit: A string denoting the subreddit that we would like to train
       on. This string should not include the "r/" prefix of the subreddit.
     - sample_size: An integer denoting how many of the input comments we would
       like to train on. If this value is None, we train on the entire dataset.
     - percentile: An integer denoting the percentile of comments that will be
       accepted. For example, an input of 90 will ensure that the comments
       we train on will only be in the top 10% of rated comments.
     - custom: A boolean that denotes whether or not we are using a file that 
       is not in the format of a reddit comment. For example, if we simply just
       want to train on any body of text, we can specify that here. The name
       of the input file is specified in the following parameter.
     - custom_file: An optional file that is only considered if custom=True.
       Specifies the name of the file that we want to train on.
     - seq_length: An integer denoting how many comments we would train on at 
       once.
     - load: Specifies whether or not we're loading from a previously trained
       checkpoint.
    """
    def __init__(
            self, subreddit, sample_size, percentile, custom=False, 
            custom_file="", seq_length=100, load_vocab=False, save_vocab=False
        ):

        self.subreddit = subreddit
        self.sample_size = sample_size
        self.percentile = percentile
        self.seq_length = seq_length

        dataset, vocab, char2idx, idx2char, text_as_int = None, None, None, None, None

        """
        If we are using a custom file, we pass it into the preprocessor so that
        it'll know that we won't be taking in text in the format of Reddit 
        comments
        """
        if custom:
            dataset, vocab, char2idx, idx2char, text_as_int = self.preprocess(
                custom=True, custom_file=custom_file
            )
        else:
            dataset, vocab, char2idx, idx2char, text_as_int = self.preprocess(
                load_vocab=load_vocab, save_vocab=save_vocab
            )
            
        self.dataset = dataset
        self.vocab = vocab
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.text_as_it = text_as_int

    """
    Splits a portion of the input text into a input, target pair. To only be
    used internally within this module. 
    Inputs:
     - chunk: A Tensor of varying shape
    Returns:
     - Two tensors
    """
    def _split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text

    """
    After we vectorize the data, we format it so that it is in a form that is
    acceptable by tensorflow.
    Input:
     - The values returned by vectorize() found in rnn.py
    Output:
     - Returns the same values but including a dataset parameter, representing
       the input data that is easily readable by Tensorflow.
    """
    def setup_vectorized_data(self, vocab, char2idx, idx2char, text_as_int):
        chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(
            self.seq_length+1, drop_remainder=True
        )
        
        dataset = chunks.map(self._split_input_target)

        return dataset, vocab, char2idx, idx2char, text_as_int

    """
    This performs the second batch of preprocessing. 
    Inputs:
     - custom_file: A string denoting the custom file that we wish to use. Only
       considered if custom=True.
     - custom: A boolean denoting whether or not we would like to use a custom
       file.
     - load_vocab: A boolean denoting whether or not we would like to load a 
       vocabulary from checkpoint.
     - save_vocab: A boolean denoting whether or not we would like to save a 
       vocabulary from a checkpoint.
    """
    def preprocess(
        self, custom_file="", custom=False, load_vocab=False, save_vocab=False
    ):
        print("Preprocessing")

        """
        The main function of this is that we would like to vectorize our input.
        We have the choice of two vectorization methodologies, but for the 
        purpose of our final product, we choose a char based vectorization as
        opposed to a word based one, which is still defined in rnn.py for
        reference.
        """
        if custom:
            pp = Preprocessor(self.subreddit, self.sample_size, self.percentile,
                custom=True, custom_file=custom_file
            )

            """
            If we do use a custom file, we need to map each instance to a 
            string, integer pair where the integer denotes a fictitious score
            that would've been used if it were a comment instead.
            """
            output, _ = pp.process(custom=True, custom_file=custom_file)
            output = [(str(output[i]), 0) for i in range(len(output))]
            vocab, char2idx, idx2char, text_as_int = vectorize(output)

            """
            Perform additional processing.
            """
            return self.setup_vectorized_data(
                vocab, char2idx, idx2char, text_as_int
            )
        else:
            good_comments = None
            if not load_vocab:
                pp = Preprocessor(self.subreddit, self.sample_size, self.percentile)
                comments, num = pp.process()
                good_comments = pp.statistics(comments)
    
            vocab, char2idx, idx2char, text_as_int = vectorize(
                good_comments, load_vocab=load_vocab, save_vocab=save_vocab
            )
            print("Vocab size of ", len(vocab))
            return self.setup_vectorized_data(
                vocab, char2idx, idx2char, text_as_int
            )

    """
    This function performs the entirety of the training.
    Inputs:
     - save: A boolean denoting whether or not we would like to save the model
       as a checkpoint.
     - epochs: An integer denoting the number of epochs we would like to train
       over. 
    """
    def regular_train(self, save=True, epochs=5):
        print("Regular training")
        """
        We shuffle the data to reduce bias.
        """
        batch_size = 1
        buffer_size = 10000
        self.dataset = self.dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True
        )

        """
        We specify some parameters specific to the neural network such as the 
        input dimensions, the number of units and which architecture we would
        like to use.

        We have the choice between two architectures. One is our original 3
        layer simple network defined in ModelOld in rnn.py that we used for
        our progress report. The second is a much more complicated network that
        is being used here for our final product.

        We use the Adam optimizer because it's good and its the best one that
        was taught in cs682. We also define the loss function with a softmax
        loss function.
        """
        vocab_size = len(self.vocab)
        embedding_dim = 256
        units = 1024
        model = ModelTest(vocab_size, embedding_dim, units)
        optimizer = tf.train.AdamOptimizer()
        def loss_function(real, preds):
            return tf.losses.sparse_softmax_cross_entropy(
                labels=real, logits=preds
            )

        model.build(tf.TensorShape([batch_size, self.seq_length]))

        """
        We define some structures that we need to keep and some final variables.
        """
        checkpoint_dir = './training_checkpoints/'
        losses = []
        iterations = []
        iteration = 1
        batchsize = len(list(self.dataset))
        for epoch in range(epochs):
            hidden = model.reset_states()
            """
            Setup some variables for timekeeping purposes
            """
            start = time.time()
            first_batch = True
            total_time = 0
            avg_batch_time = 0
            begin = time.time()
            for (batch, (inp, target)) in enumerate(self.dataset):
                """
                Perform forward and backpropagation.
                """
                with tf.GradientTape() as tape:
                    predictions = model(inp)
                    loss = loss_function(target, predictions)
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

                """
                Print our progress every 100 batches.
                """
                if batch % 100 == 0:
                    """
                    Here, we calculate the approximate time remaining in minutes.
                    """
                    end = time.time()
                    total_time += end - begin
                    avg_batch_time = int(total_time/float(iteration))
                    remaining_time = (((batchsize - batch)/100)*avg_batch_time)/60.
                    print ('Epoch {} Batch {} of {} Loss {:.4f} Time {:.4f} secs, remaining time {:.4f} mins'.format(epoch+1,
                                                                    batch,
                                                                    batchsize,
                                                                    loss, 
                                                                    end-begin,
                                                                    remaining_time))
                    """
                    Add our progress.
                    """
                    losses.append(loss)
                    iterations.append(iteration)
                    iteration += 1
                    begin = time.time()
            print ('Epoch {}'.format(epoch+1))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

        if save:
            model.save_weights(checkpoint_dir + "checkpoint")

        return model, losses, iterations

    """
    This function loads a model from a perviously trained checkpoint.
    Inputs:
     - dir: A string denoting the directory of the checkpoint. 
    Returns:
     - model: The model object with the trained weights.
    """
    def load(self, dir):
        embedding_dim = 256
        units = 1024
        vocab_size = len(self.vocab)
        batch_size = 1
        buffer_size = 10000

        model = ModelTest(vocab_size, embedding_dim, units)
        model.build(tf.TensorShape([batch_size, self.seq_length]))

        model.load_weights(dir)
        return model

    """
    This function performs the entirety of the predictions. We essentially take
    our model, specify a start string and parse it through or model. 
    Inputs:
     - model: A Tensorflow object denoting our trained model.
     - num_generate: The number of samples that we wish to generate. In the case
       of a char based model, this denotes the number of characters to generate.
       In the case of a word based model, this denotes the number of words to
       generate.
     - start_string: In the case of a char based model, this denotes the start
       char of our prediction. In the case of a word based model, this denotes
       the start word of our prediction.
     - out: A boolean denoting whether or not we would like to write our output
       to a file.
     - temperature: A float in the range [0.0, 1.0] that denotes the temperature
       of our prediction. This value corresponds to the softmax equation denoted
       by the value T:
                        q_i = exp(z_i/T)/sum(exp(z_j/T))
       The smaller the temperature value is, the higher the above probability
       is. Our prediction will be more confident but it will also be more 
       conservative and boring. A high temperature value will yield a smaller
       probability, rendering a more varied prediction. But a too high
       temperature could give a prediction so varied that it will no longer make
       sense.
    """
    def predict(self, model, num_generate, start_string, out=False, temperature=1.0):
        print("Predicting...")
        num_generate = num_generate
        start_string = "a"
        input_eval = [self.char2idx[s] for s in start_string]
        # input_eval = [self.char2idx[start_string]]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
            
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])

        print (start_string + ''.join(text_generated))
        
        print("Writing to output")

        if out:
            if not os.path.exists("outputs/" + "Out111" + ".txt"):
                output = open("outputs/" + "Out111" + ".txt", "w+", encoding="utf-8")
                output.write(start_string + ''.join(text_generated))
                output.close()
            else:
                count = 1
                while os.path.exists("outputs/" + "Out111" + str(count) + ".txt"):
                    count += 1
                output = open("outputs/" + "Out111" + str(count) + ".txt", "w+", encoding="utf-8")
                output.write(start_string + ''.join(text_generated))
                output.close()
        