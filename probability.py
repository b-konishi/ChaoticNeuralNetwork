# -*- coding: utf-8 -*-
import math
import numpy as np
import itertools

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

class InfoContent:

    def __init__(self):
        pass

    def get_prob2(self, x, y):
        px, py, p = dict(), dict(), dict()

        for (xi, yi) in zip(x, y):
            px[xi] = px.get(xi,0) + 1/len(x)
            py[yi] = py.get(yi,0) + 1/len(y)

            p[(xi,yi)] = p.get((xi,yi),0) + 1/min(len(x),len(y))
        
        return (px, py, p)

    def get_IC(self, prob):
        icx, icy, ic = dict(), dict(), dict()
        px, py, p = prob

        for (xi, pi) in px.items():
            icx[xi] = -math.log(pi, 2)

        for (yi, pi) in py.items():
            icy[yi] = -math.log(pi, 2)

        for ((xi,yi), pi) in p.items():
            ic[(xi,yi)] = -math.log(pi, 2)

        return (icx, icy, ic)


    def get_EN(self, prob):
        enx, eny, en = 0, 0, 0
        px, py, p = prob
        icx, icy, ic = self.get_IC(prob)

        for (xi, pi) in px.items():
            enx = enx + pi*icx.get(xi)

        for (yi, pi) in py.items():
            eny = eny + pi*icy.get(yi)

        for ((xi,yi), pi) in p.items():
            en = en + pi*ic.get((xi,yi))

        return (enx, eny, en)

    def get_MIC(self, x, y):
        prob = self.get_prob2(x,y)
        enx, eny, en = self.get_EN(prob)

        return enx + eny - en

    # time-delayed-content on phase space
    def get_tau(self, data, max_tau):
        mic = []
        tau = None
        for _tau in range(1,max_tau):
            unlaged = data[:-_tau]
            laged = data[_tau:]
            mic.append(self.get_MIC(unlaged, laged))

            if tau is None and len(mic)>1 and mic[-2] < mic[-1]:
                tau = _tau-1

        if tau is None:
            print('[Warning] tau is None.')
            tau = 0

        return tau, mic


    # TF(x<=y)
    def np_get_TE(self, to_x, from_x):
        x, y = to_x, from_x
        if any(np.isnan(x)) or any(np.isnan(y)):
            return 0.
    
        xmax, xmin = max(x), min(x)
        ymax, ymin = max(y), min(y)

        N = min(len(x), len(y)) - 1
        Nx = int(np.log2(N) + 1)
        Ny = Nx
        # print('N: ', N)
        # print('bin: ', Nx)

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

        tau = 1
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
                        loc=tf.transpose([x[tau:], x[:length-tau], y[:length-tau]]),
                        scale_diag=[[scale]*3])),
                reinterpreted_batch_ndims=0)

        dmx = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                components_distribution=tfd.Normal(loc=x[:length-tau], scale=scale)
                )

        dmxx = tfd.Independent(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=tf.transpose([x[tau:],x[:length-tau]]),
                        scale_diag=[scale]*2)),
                reinterpreted_batch_ndims=0)

        dmxy = tfd.Independent(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[1/n]*n),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=tf.transpose([x[:length-tau],y[:length-tau]]),
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

if __name__ == '__main__':
    ic = InfoContent()
    
    x = np.random.rand(100)*100
    x = x+x
    x = [1,1,1,1,1,0,0,0,0,0,1,1,1]

    mic = []
    tau = None
    for _tau in range(1,5+1):
        unlaged = x[:-_tau]
        laged = x[_tau:]
        print(unlaged, laged)
        mic.append(ic.get_MIC(unlaged, laged))

        if tau is None and len(mic)>1 and mic[-2] < mic[-1]:
            print(len(mic))
            tau = _tau-1

    print(mic, tau)
    plt.plot(range(1,len(mic)+1), mic)
    plt.show()


