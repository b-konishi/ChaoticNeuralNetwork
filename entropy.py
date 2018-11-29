# -*- coding: utf-8 -*-
import math
import numpy as np
import itertools

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

class TransferEntropy:

    def __init__(self):
        pass

    # TF(x<=y)
    def np_get_TE(self, x, y):
        if any(np.isnan(x)) or any(np.isnan(y)):
            return 0.
    
        xmax, xmin = max(x), min(x)
        ymax, ymin = max(y), min(y)

        N = min(len(x), len(y)) - 1
        Nx = int(np.log2(N) + 1)
        Ny = Nx
        print('N: ', N)
        print('bin: ', Nx)

        p, px, pxx, pxy = dict(), dict(), dict(), dict()

        for (x1, (x,y)) in zip(x[1:], zip(x[:-1],y[:-1])):
            norm_x1 = int(Nx * (x1 if xmax-xmin == 0 else (x1-xmin)/(xmax-xmin)))
            norm_x = int(Nx * (x if xmax-xmin == 0 else (x-xmin)/(xmax-xmin)))
            norm_y = int(Ny * (y if ymax-ymin == 0 else (y-ymin)/(ymax-ymin)))

            p[(norm_x1, norm_x, norm_y)] = p.get((norm_x1, norm_x, norm_y), 0) + 1/N
            px[norm_x] = px.get(norm_x, 0) + 1/N
            pxx[(norm_x1, norm_x)] = pxx.get((norm_x1, norm_x), 0) + 1/N
            pxy[(norm_x, norm_y)] = pxy.get((norm_x, norm_y), 0) + 1/N

        te = 0
        for ((xi1, xi, yi), pi) in p.items():
            a = pi * px.get(xi)
            b = pxy.get((xi, yi)) * pxx.get((xi1, xi))

            te = te + pi*np.log2(a/b)

        return te
    
    # TF(x<=y)
    def tf_get_TE(self, x, y, length):
        # print('\n##### TRANSFER ENTROPY FOR TENSORFLOW #####')

        tau = 5
        n = length-tau

        # N = int(np.log2(n) + 1)
        # print('n: ', n)
        # print('bin: ', N)

        # Occurred the error when 'scale' sets the value lower than about 0.1.
        sigma_rate = .1
        scale = (1/n)/(sigma_rate*2)
        scale = .5


        dm = tfd.Independent(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=tf.transpose([x[tau:], x[:-tau], y[:-tau]]),
                        scale_diag=[[scale]*3])),
                reinterpreted_batch_ndims=0)

        dmx = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                components_distribution=tfd.Normal(loc=x[:-tau], scale=scale)
                )

        dmxx = tfd.Independent(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=tf.transpose([x[tau:],x[:-tau]]),
                        scale_diag=[scale]*2)),
                reinterpreted_batch_ndims=0)

        dmxy = tfd.Independent(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=tf.transpose([x[:-tau],y[:-tau]]),
                        scale_diag=[scale]*2)),
                reinterpreted_batch_ndims=0)

        pdf = {'dm': dm, 'dmx': dmx, 'dmxx': dmxx, 'dmxy': dmxy}


        # WARNING: memory-occupated-size=sampling^3
        sampling = 20

        i = np.reshape(np.linspace(0,1,sampling), [sampling,1])
        idx = np.concatenate(list(itertools.product(i, repeat=3)), axis=1).T

        p, px, pxx, pxy = dm.prob(idx), dmx.prob(idx[:,1]), dmxx.prob(idx[:,:-1]), dmxy.prob(idx[:,1:])

        y = p * tf.log((p*px)/(pxy*pxx) + 1e-30)
        y = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
        entropy = tf.reduce_sum(y)

        return entropy, pdf



