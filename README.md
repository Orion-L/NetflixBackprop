# NetflixBackprop

A collaborative filtering implementation using a simple feed-forward MLP and backpropagation.
Implemented using Keras (https://keras.io/) and TensorFlow (https://www.tensorflow.org/).

Data set text files are to be passed in through argument 1, e.g. ./backprop_filter.py data.txt
Ratings are represented in the text file as a space separated matrix, with each row corresponding to a user and each column corresponding to a rated item.
Missing ratings are represented with a '-' and test data is identified by placing '?' after the rating.
See the provided data sets for an example.
