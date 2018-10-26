# -*- coding: utf-8 -*-
import math

'''
yseq = [1,1,1,2,3]

N = 1

px, py, p = dict(), dict(), dict()

for (x, y) in zip(xseq, yseq):
    px[x] = px.get(x,0) + 1
    py[y] = py.get(y,0) + 1
    p[(x,y)] = p.get((x,y),0) + 1

print('px: ', px)
print('py: ', py)
print('p: ', p)
'''


def get_prob(x, y):
    px, py, p = dict(), dict(), dict()

    for (xi, yi) in zip(x, y):
        px[xi] = px.get(xi,0) + 1/len(x)
        py[yi] = py.get(yi,0) + 1/len(y)

        p[(xi,yi)] = p.get((xi,yi),0) + 1/min(len(x),len(y))
    
    return (px, py, p)


def get_IC(prob):
    icx, icy, ic = dict(), dict(), dict()
    px, py, p = prob

    for (xi, pi) in px.items():
        icx[xi] = -math.log(pi, 2)

    for (yi, pi) in py.items():
        icy[yi] = -math.log(pi, 2)

    for ((xi,yi), pi) in p.items():
        ic[(xi,yi)] = -math.log(pi, 2)

    return (icx, icy, ic)


def get_EN(prob):
    enx, eny, en = 0, 0, 0;
    px, py, p = prob;
    icx, icy, ic = get_IC(prob)

    for (xi, pi) in px.items():
        enx = enx + pi*icx.get(xi)

    for (yi, pi) in py.items():
        eny = eny + pi*icy.get(yi)

    for ((xi,yi), pi) in p.items():
        en = en + pi*ic.get((xi,yi))

    return (enx, eny, en)


def get_MIC(x, y):
    mic = 0
    px, py, p = get_prob(x,y)

    for ((xi,yi),pi) in p.items():
        mic = mic + pi * math.log(pi / (px.get(xi)*py.get(yi)), 2)

    return mic


def get_MIC2(x, y):
    prob = get_prob(x,y)
    enx, eny, en = get_EN(prob)

    return enx + eny - en


xseq = [3,3,3,2,2]
yseq = [2,2,2,1,1]

if len(xseq) != len(yseq):
    print('not equal length')

print('\n##### DATA #####')
print('x: ', xseq)
print('y: ', yseq)

print('\n##### PROBABILITY #####')
p = get_prob(xseq, yseq)
px, py, pp = p
print('p: ', p)

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


    
