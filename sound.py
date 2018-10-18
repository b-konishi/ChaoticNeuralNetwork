# -*- coding: utf-8 -*-

from scipy.io.wavfile import read

wavfile = "./music/jinglebells.wav"

fs, data = read(wavfile)

print("Sampling rate :", fs)

if (data.shape[1] == 2):
    left = data[:, 0]
    right = data[:, 1]
