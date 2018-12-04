# -*- coding: utf-8 -*-

# Sound
from scipy.io.wavfile import read
import wave
import array

# Recurrence plot
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

class Sound:

    def load_sound(filename):
        fs, sound = read(filename)
        if (sound.shape[1] == 2):
            load = sound[:,0]

        return load

    def save_sound(data, filename, sampling=44100):
        w = wave.Wave_write(filename)
        w.setparams((
            1,                        # channel
            2,                        # byte width
            sampling,                    # sampling rate
            len(data),            # number of frames
            "NONE", "not compressed"  # no compression
        ))
        w.writeframes(array.array('h', data).tostring())
        w.close()
        print('saving sound...')

class RecurrencePlot:

    def __init__(self):
        pass

    def plot(self, ax, data, eps=0.4):
        data = np.reshape(data, [len(data),1]) if len(np.shape(data)) == 1 else np.array(data)
        data = (data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))
        mat = np.array([list(np.sqrt(np.sum(np.power(_data-data, 2), axis=(1 if len(np.shape(data))==2 else 0)))) for _data in data])

        # Visualization of matrix
        ax.pcolor(mat<eps, cmap='Greys')

class DataProcessing:

    def __init__(self):
        pass


if __name__ == '__main__':
    x = [1,2,3,4,5]
    r = RecurrencePlot()
    r.plot(x)
    r.show_image()









