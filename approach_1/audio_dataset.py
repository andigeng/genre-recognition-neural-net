import numpy as np
import sunau as sn
import os


data_location = '../data/genres'
genre_locations = os.listdir(data_location)
genre_locations.sort()


class audio_dataset:
  """ Audio dataset is an encapsulation of the audio data. The class does not
  actually store all the files in memory, but rather it stores the locations of 
  the data, in addition to the target label. Audio is read in with the 
  load_batch() function when needed to conserve memory.
  """

  def __init__(self):
    """ Initializes the audio dataset wrapper class. """
    
    # These variables hold the training and testing metadata, which include the
    # file locations of the samples, and their target label.
    self.train_indices = []
    self.train_label = []
    self.valid_indices = []
    self.valid_label = []
    
    # Fetch the metadata and populate the arrays.
    self.get_metadata()

    # These variables store the size of our training and testing set
    self.num_train = self.train_indices.shape[0]
    self.num_valid = self.valid_indices.shape[0]

    # These variables track our progress through the training and testing set
    self.index_in_epoch = 0
    self.index_in_valid = 0
    self.completed_epochs = 0

    # Generate a permutation representing the order of data to be trained
    self.perm = np.arange(self.num_train)
    self.shuffle_data()


  def shuffle_data(self):
    """ Shuffles perm, the order in which data will be learned. """
    np.random.shuffle(self.perm)


  def get_batch_train(self, batch_size):
    """ Given a size, returns a training batch."""
    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    if self.index_in_epoch > self.num_train:
      self.completed_epochs += 1

      # Reshuffle the data
      self.shuffle_data()
      start = 0
      self.index_in_epoch = batch_size

    end = self.index_in_epoch
    cur_perm = self.perm[start:end]
    batch = self.load_batch(self.train_indices[cur_perm])
    labels = self.train_label[cur_perm]
    return batch, labels


  def reset_batch_valid(self):
    self.index_in_valid = 0


  def get_batch_valid(self, batch_size):
    """ Given a size, returns a validation batch. """
    start = self.index_in_valid
    self.index_in_valid +=batch_size

    if self.index_in_valid > self.num_valid:
      self.reset_batch_valid()
      return None, None
    
    end = self.index_in_valid

    batch = self.load_batch(self.valid_indices[start:end])
    labels = self.valid_label[start:end]
    return batch, labels


  def load_batch(self, file_arr, random = True):
    """ Given an array of file locations, loads the corresponding files into a
    single numpy array (the batch) and returns it. The full audio is ~30 seconds
    long, but we opt to only train and test on randomized 11 second segments.
    """
    sample_size = 262144  # Approximately 11 seconds
    max_size = 660000     # Approximately 30 seconds

    if (random):
      start = np.random.randint(0, max_size - sample_size)
    else:
      start = sample_size

    batch = np.zeros((len(file_arr),262144,1,1), np.float32)

    for i in range(file_arr.shape[0]):
      # Read the .au files and convert into a numpy array.
      f = sn.Au_read(str(file_arr[i]))
      sound = np.fromstring(f.readframes(660000), dtype=np.dtype('>h'))
      
      # Truncate the sound file, and append to the batch array.
      sound = sound[start:start+sample_size]
      batch[i,:,0,0] = sound

    return batch

  def get_metadata(self):
    """ Fetches the informatation for the testing and validation set. This
    includes the file location and class label. 
    """
    for genre in genre_locations:
      dirname = os.path.join(data_location, genre)
      filenames = os.listdir(dirname)
      filenames.sort()  # Sort the filenames so they will be consistent
      filenames = [os.path.join(dirname, fn) for fn in filenames]

      # Appends every fifth element to the test set, and hot encodes the labels
      for num in range(len(filenames)):
        label = [np.float32(genre == g) for g in genre_locations]
        if (num%5 != 0):
          self.train_indices.append(filenames[num])
          self.train_label.append(label)
        else:
          self.valid_indices.append(filenames[num])
          self.valid_label.append(label)

    # Convert the arrays into Numpy arrays for easy manipulation
    self.train_indices = np.array(self.train_indices)
    self.valid_indices = np.array(self.valid_indices)
    self.train_label = np.array(self.train_label)
    self.valid_label = np.array(self.valid_label)