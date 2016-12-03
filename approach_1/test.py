import numpy as np
import sunau as sn
import os

import scipy.signal
from scipy.signal import spectrogram as spec

import matplotlib.pyplot as plt



#file1 = '../data/genres/blues/blues.00000.au'
#file1 = '../data/genres/classical/classical.00000.au'
"""
file1 = '../data/genres/disco/disco.00000.au'
f = sn.Au_read(file1)
sound = np.fromstring(f.readframes(700000), dtype=np.dtype('>h'))

f, t, Sxx = spec(sound, nperseg=256)
print(len(f))
print(len(t))
print(len(Sxx))
print(len(Sxx[0]))

print(t[:20])
print(f[:])
"""


#plt.pcolormesh(t, f, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

indices = []
label = []

data_location = '../data/genres'
genre_locations = os.listdir(data_location)
genre_locations.sort()

for genre in genre_locations:
  dirname = os.path.join(data_location, genre)
  filenames = os.listdir(dirname)
  filenames.sort()
  filenames = [os.path.join(dirname, fn) for fn in filenames]

  for num in range(1):
    f = sn.Au_read(filenames[num])
    sound = np.fromstring(f.readframes(262144), dtype=np.dtype('>h'))

    f, t, Sxx = spec(sound, nperseg=256)
    f = np.arange(0, 130, 1)
    t = np.arange(0, 1171, 1)
    print(Sxx[0][0])
    print("first: {}, last: {}".format(f[1], f[-2]))

    max = 0
    min = 100000000

    Sxx = np.log(Sxx)
    print(Sxx.shape)
    #for i in range(len(Sxx)):
      #for j in range(len(Sxx[0])):
        #Sxx[i][j] = np.log
        #if (Sxx[i][j] >= max): max = Sxx[i][j]
        #if (Sxx[i][j] <= min): min = Sxx[i][j]

    #print(max)
    #print(min)
    plt.title(filenames[num])
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()