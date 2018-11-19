# -*- coding: utf-8 -*-

# Sound
from scipy.io.wavfile import read
import wave
import array

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
