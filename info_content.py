# -*- coding: utf-8 -*-
import math
import numpy as np

import tensorflow as tf


class info_content():

    def get_prob(self, x):
        p = dict()

        for xi in x:
            p[xi] = p.get(xi,0) + 1/len(x)
        
        return p


    def get_prob2(self, x, y):
        px, py, p = dict(), dict(), dict()

        for (xi, yi) in zip(x, y):
            px[xi] = px.get(xi,0) + 1/len(x)
            py[yi] = py.get(yi,0) + 1/len(y)

            p[(xi,yi)] = p.get((xi,yi),0) + 1/min(len(x),len(y))
        
        return (px, py, p)

    def get_prob3(self, x, y, z):
        px, py, pz, p = dict(), dict(), dict(), dict()

        for (xi, yi, zi) in zip(x, y, z):
            px[xi] = px.get(xi,0) + 1/len(x)
            py[yi] = py.get(yi,0) + 1/len(y)
            pz[zi] = pz.get(zi,0) + 1/len(z)

            p[(xi,yi,zi)] = p.get((xi,yi,zi),0) + 1/min(len(x),len(y),len(z))
        
        return (px, py, pz, p)

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
        icx, icy, ic = get_IC(prob)

        for (xi, pi) in px.items():
            enx = enx + pi*icx.get(xi)

        for (yi, pi) in py.items():
            eny = eny + pi*icy.get(yi)

        for ((xi,yi), pi) in p.items():
            en = en + pi*ic.get((xi,yi))

        return (enx, eny, en)


    def get_MIC(self, x, y):
        mic = 0
        px, py, p = get_prob2(x,y)

        for ((xi,yi),pi) in p.items():
            mic = mic + pi * math.log(pi / (px.get(xi)*py.get(yi)), 2)

        return mic


    def get_MIC2(self, x, y):
        prob = get_prob2(x,y)
        enx, eny, en = get_EN(prob)

        return enx + eny - en

    def get_TE(self, x, y):

        px = self.get_prob(x[:-1])
        _,_,p2 = self.get_prob2(x[:-1], y[:-1])
        _,_,px2 = self.get_prob2(x[1:], x[:-1])
        _, _, _, p3 = self.get_prob3(x[1:], x[:-1], y[:-1])

        te = 0
        for ((xi1,xi,yi),pi) in p3.items():
            a = pi * px.get(xi)
            b = p2.get((xi,yi)) * px2.get((xi1,xi))

            te = te + pi*math.log(a/b, 2)

        return te

    def get_TE2(self, x, y):
        xmax, xmin = max(x), min(x)
        ymax, ymin = max(y), min(y)

        N = min(len(x), len(y)) - 1
        Nx = int(np.log2(N) + 1)
        Ny = Nx
        print('N: ', N)
        print('bin: ', Nx)

        p, px, pxx, pxy = dict(), dict(), dict(), dict()

        for (x1, (x,y)) in zip(x[1:], zip(x[:-1],y[:-1])):
            norm_x1 = int(Nx * (x1-xmin)/(xmax-xmin))
            norm_x = int(Nx * (x-xmin)/(xmax-xmin))
            norm_y = int(Ny * (y-ymin)/(ymax-ymin))

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


    def get_TE_for_tf(x, y, TE, length):
        print('\n##### TRANSFER ENTROPY FOR TENSORFLOW #####')

        xmax, xmin = tf.reduce_max(x), tf.reduce_min(x)
        ymax, ymin = tf.reduce_max(y), tf.reduce_min(y)

        N = length - 1
        Nx = int(np.log2(N) + 1)
        Ny = Nx
        print('N: ', N)
        print('bin: ', Nx)

        p, px, pxx, pxy = dict(), dict(), dict(), dict()

        for i in range(length-1):
            norm_x1 = tf.cast(Nx * (x[i+1]-xmin)/(xmax-xmin), tf.int64)
            norm_x = tf.cast(Nx * (x[i]-xmin)/(xmax-xmin), tf.int64)
            norm_y = tf.cast(Ny * (y[i]-ymin)/(ymax-ymin), tf.int64)

            p[(norm_x1, norm_x, norm_y)] = p.get((norm_x1, norm_x, norm_y), 0) + 1/N
            px[norm_x] = px.get(norm_x, 0) + 1/N
            pxx[(norm_x1, norm_x)] = pxx.get((norm_x1, norm_x), 0) + 1/N
            pxy[(norm_x, norm_y)] = pxy.get((norm_x, norm_y), 0) + 1/N

        for ((xi1, xi, yi), pi) in p.items():
            a = pi * px.get(xi)
            b = pxy.get((xi, yi)) * pxx.get((xi1, xi))

            TE = TE + pi*np.log2(a/b)

        return TE

            


### TEST ###
ic = info_content()

xseq = [1,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1]
yseq = [1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1]

xseq = np.random.rand(100)*2 - 1
yseq = 2*xseq[1:] + 0.01*np.random.rand()
xseq = xseq[:-1]
'''
x: [x0, x1, ..., x(n-1)]
y: [x1, x2, ..., x(n)]
xはyの値を参考にして、次の時刻で、yの値をとっている＝yの影響をxが受けている(y->x)
'''

if len(xseq) != len(yseq):
    print('not equal length')

print('\n##### TRANSFER ENTROPY1 #####')
te_xy = ic.get_TE(yseq, xseq)
te_yx = ic.get_TE(xseq, yseq)
print('te_x->y: ', te_xy)
print('te_y->x: ', te_yx)

print('\n##### TRANSFER ENTROPY2 #####')
te_xy = ic.get_TE2(yseq, xseq)
te_yx = ic.get_TE2(xseq, yseq)
print('te_x->y: ', te_xy)
print('te_y->x: ', te_yx)

'''
print('\n##### DATA #####')
print('x: ', xseq)
print('y: ', yseq)

print('\n##### PROBABILITY #####')
p3 = get_prob3(xseq[1:], xseq[:-1], yseq[:-1])
px2 = get_prob2(xseq[1:], xseq[:-1])
p2 = get_prob2(xseq[:-1], yseq[:-1])
px, py, _ = get_prob2(xseq, yseq)

print('px: ', px)
print('px2: ', px2)
print('p2: ', p2)
print('p3: ', p3)

'''

'''
print('\n##### INFOMATION CONTENT #####')
info = get_IC(p)
print('info: ', info)

print('\n##### ENTROPY #####')
entropy = get_EN(p)
print('entropy: ', entropy)

print('\n##### MUTUAL INFORMATION CONTENT #####')
mic = get_MIC(xseq, yseq)
mic2 = get_MIC2(xseq, yseq)
print('mic: ', mic)
print('mic2: ', mic2)

'''

    
