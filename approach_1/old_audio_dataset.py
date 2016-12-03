import numpy as np
import sunau as sn
import os

import scipy.signal
from scipy.signal import spectrogram as spec



data_location = '../data/genres'
genre_locations = os.listdir(data_location)
genre_locations.sort()


class audio_dataset:
  """ Audio dataset is an encapsulation of our audio data. The class does not
  actually store all the files in memory, but rather it stores the locations of 
  the data, in addition to the target label. Audio is read in with the 
  load_batch() function when needed to conserve memory.
  """

  def __init__(self):
    """ Initializes the audio dataset wrapper class. """
    train_indices = []
    train_label = []

    valid_indices = []
    valid_label = []
    
    # Populate the arrays storing the indices and labels
    for genre in genre_locations:
      dirname = os.path.join(data_location, genre)
      filenames = os.listdir(dirname)
      filenames.sort()
      filenames = [os.path.join(dirname, fn) for fn in filenames]

      # Appends every fifth element to the test set, and hot encodes the labels
      for num in range(len(filenames)):
        label = [np.float32(genre == g) for g in genre_locations]

        if (num%5 != 0):
          train_indices.append(filenames[num])
          train_label.append(label)
        else:
          valid_indices.append(filenames[num])
          valid_label.append(label)

    # Convert the arrays into Numpy arrays for easy manipulation
    train_indices = np.array(train_indices)
    valid_indices = np.array(valid_indices)
    train_label = np.array(train_label)
    valid_label = np.array(valid_label)

    # These variables hold the training and testing metadata
    self.train_indices = train_indices
    self.train_label = train_label
    self.valid_indices = valid_indices
    self.valid_label = valid_label

    # These variables store the size of our training and testing set
    self.num_train = train_indices.shape[0]
    self.num_valid = valid_indices.shape[0]

    # These variables track our progress through the training and testing set
    self.index_in_epoch = 0
    self.index_in_valid = 0
    self.completed_epochs = 0

    # Generate a permutation representing the order of data to be trained
    self.perm = []
    self.get_new_permutation()


  def get_new_permutation(self):
    """ Generates an array that represents a permutation of the training
    elements.
    """
    self.perm = np.arange(self.num_train)
    np.random.shuffle(self.perm)


  def next_batch_train(self, batch_size):
    """ 
    """
    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    if self.index_in_epoch > self.num_train:
      self.completed_epochs += 1

      #Reshuffle the data
      self.get_new_permutation()
      start = 0
      self.index_in_epoch = batch_size

    end = self.index_in_epoch
    cur_perm = self.perm[start:end]
    return self.load_batch(self.train_indices[cur_perm]), self.train_label[cur_perm], batch_size


  def reset_batch_valid(self):
    self.index_in_valid = 0


  def next_batch_valid(self, batch_size):
    start = self.index_in_valid
    self.index_in_valid +=batch_size
    if self.index_in_valid > self.num_valid:
      self.reset_batch_valid()
      return -1, -1, -1
    end = self.index_in_valid
    return self.load_batch(self.valid_indices[start:end]), self.valid_label[start:end], batch_size


  def load_batch(self, file_arr, random = True):
    sample_size = 262144
    max_size = 660000

    if random:
      start = np.random.randint(0, max_size - sample_size)
    else:
      start = sample_size

    data = np.zeros((len(file_arr),129,1170,129), np.float32)

    for i in range(file_arr.shape[0]):
      f = sn.Au_read(str(file_arr[i]))
      sound = np.fromstring(f.readframes(660000), dtype=np.dtype('>h'))
      f, t, Sxx = spec(sound[0:2**18], 661794/30, nperseg=256)
      Sxx = np.log(Sxx)

      data[i] = Sxx
      print(data.shape)
      #data[i,:,0,0] = sound[start:start+sample_size]

    return data



dataset = audio_dataset()
dataset.next_batch_valid(4)

"""
  def load_batch(self, file_arr, random = True):
    sample_size = 262144
    max_size = 660000

    data = np.zeros((len(file_arr), 524288, 1, 1), np.float32)

    for i in range(file_arr.shape[0]):
      print(file_arr[i])
      f = sn.Au_read(str(file_arr[i]))
      sound = np.fromstring(f.readframes(524288), dtype = np.dtype('>h'))
      data[i,:,0,0] = sound

    return data

  def load_batch(self, file_arr, random = True):
    data_point, fs, enc = au.auread(file_arr[0])

    data = np.zeros((len(file_arr),262144,1,1), np.float32)

    for i in range(file_arr.shape[0]):
      data_point, fs, enc = au.auread(file_arr[i])
      data[i,:,0,0] = data_point[0:262144]

    return data

  def load_batch(self, file_arr, random = True):
    sample_size = 262144
    max_size = 660000

    if random:
      start = np.random.randint(0, max_size - sample_size)
    else:
      start = sample_size

    data = np.zeros((len(file_arr),262144,1,1), np.float32)

    for i in range(file_arr.shape[0]):
      f = sn.Au_read(str(file_arr[i]))
      sound = np.fromstring(f.readframes(660000), dtype=np.dtype('>h'))
      data[i,:,0,0] = sound[start:start+sample_size]

    return data

"""