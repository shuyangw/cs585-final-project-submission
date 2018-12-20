from runner import Runner

import matplotlib.pyplot as plt

import os
import sys

import numpy as np

"""
Loads a model and vocabulary and predicts 6 samples from T=0.0 to 1.0 and saves 
the predictions.
"""
def load_and_predict():
    exp = Runner("leagueoflegends", None , 90, load_vocab=True, save_vocab=False)
    model = exp.load("training_checkpoints/ckpt")

    r = np.linspace(0.0, 1.0, num=6)
    for temp in r:
        print(80*"#")
        print("PREDICTING FOR TEMPERATURE", temp)
        exp.predict(model, 1000, "s", out=False, temperature=temp)
        print(80*"#")

"""
Trains a model with only 1e5 samples and 1 epoch and outputs a prediction and
loss graph.
"""
def shortened_train():
    exp = Runner("leagueoflegends", 1e5 , 90)
    model, losses, iterations = exp.regular_train(epochs=1)
    exp.predict(model, 10000, "s", out=False)
    print("Finished predicting")
    plt.title("Losses")

    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    
    if not os.path.exists("plots/" + "RC2015_05" + ".png"):
        plt.savefig("plots/" + "RC2015_05" + ".png")
    else:
        count = 1
        while os.path.exists("plots/" + "RC2015_05" + str(count) + ".png"):
            count += 1
        plt.savefig("plots/" + "RC2015_05" + str(count) + ".png")

"""
Trains many models with sample sizes specified in sample_sizes.
"""
def range_train():
    sample_sizes = [1e7]
    for size in sample_sizes:
        exp = Runner("leagueoflegends", size, 75)
        model, losses, iterations = exp.regular_train(epochs=50)
        exp.predict(model, 1000, "s", True)

        plt.title("Losses for size " + str(size))

        plt.plot(iterations, losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        
        if not os.path.exists("plots/" + str(size) + ".png"):
            plt.savefig("plots/" + str(size) + ".png")
        else:
            count = 1
            while os.path.exists("plots/" + str(size) + ".png"):
                count += 1
            plt.savefig("plots/" + str(size) + ".png")

        print("#"*80)
        print("FINAL LOSS FOR SIZE ", size, losses[len(losses)-1])
        print("#"*80)

        plt.clf()

"""
Trains a model using the entire dataset and 5 epochs. Warning, this took us
17 hours on a GTX 1060.
"""
def single_complete_train():
    exp = Runner("leagueoflegends", None, 90)
    model, losses, iterations = exp.regular_train(epochs=5)
    exp.predict(model, 5000, "s", True)

    plt.title("Losses for size " + str(None))

    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

"""
Trains a model on 1e4 sample sizes.
"""
def single_partial_train():
    exp = Runner("leagueoflegends", 1e4, 90, save_vocab=True, load_vocab=True)
    model, losses, iterations = exp.regular_train(epochs=1, save=False)
    exp.predict(model, 500, "s", True)

    plt.title("Losses for size " + str(None))

    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")


"""
WRITE YOUR OWN EXPERIMENT :P
"""
def custom():
    """
    Number of samples predicted in the number of characters
    """
    samples_predicted = 500

    """
    Temperature that we would like to use.
    """
    temp = 0.5

    """
    Whether or not we would like to save our predictions to disk.
    """
    save=False

    """
    Initiates our training object
    """
    exp = Runner("leagueoflegends", None , 90, load_vocab=True, save_vocab=False)

    """
    Loads our model.
    """
    model = exp.load("training_checkpoints/ckpt")

    """
    Predicts using our parameters
    """
    exp.predict(model, samples_predicted, "s", out=False, temperature=temp)

if __name__ == '__main__':
    # custom()
    load_and_predict()
