Reddit Comment Generator Source Code
Authors: Shuyang Wang, Justin Lee, Quyen Tran, Oliver Pang

How to run:
The code is built using Python 3.6. It requires the following packages to build:
    matplotlib
    numpy
    tqdm
    json
    tensorflow
    pickle
In addition, the project was originally built using a GPU version of Tensorflow,
it may not work without a GPU version but the code does allow for CPU based
training and predicting. But without a GPU, training and predicting will be 
significantly slower. 

To run the code, simply run python on main.py using the command line command:
    python main.py
or if Python 2.7 is installed concurrently, use
    python3 main.py

Regarding reproducing our experiments:
Since we need to vectorize our input in order for our code to work, we would
need a sizable dataset for our code to reproduce our results. To help alleviate
this, we provided a feature where we can save our vectorization, cutting down
the original file size significantly. We've provided this vectorization in our 
submission in a compressed format as vocab.zip. You can specify that we would 
like to load it in our code.

We also produced a model checkpoint to use so we wouldn't have to train our 
model every time we want to predict.

Look into main.py to view our experiments. You would need to change the last 
function in the code to run your own experiments due to a multitude of 
command line arguments that we would need to incorporate.