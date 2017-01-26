# genre-recognition-neural-net
A WIP convolutional neural network that classifies songs to music genres. Uses the [GTZAN dataset](http://marsyasweb.appspot.com/download/data_sets/), a collection of 1000 30 second songs. The content in 'Approach 1' achieves 37.5% accuracy on a testing set of 100 songs across 10 music genres. The architecture is inspired by the [VGG-net](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/). Each input into the neural net is a randomly sampled ~5 second snippet of the song files to increase the generalizability of the network.

Requires Tensorflow and Numpy to run.

More approaches in progress [here](https://github.com/AlperenAydin/GenreRecognition).
