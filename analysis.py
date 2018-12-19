import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

import my_library as my
import probability
from mpl_toolkits.mplot3d import Axes3D

class Analysis:
    def __init__(self, filename):
        self.time, self.x1, self.y1, self.x2, self.y2 = [],[],[],[],[]

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            [next(reader) for _ in range(2)]
            index = next(reader)
            for d in reader:
                d = np.array(d).astype(np.float32)
                self.time.append(d[index.index('time[ms]')])
                self.x1.append(d[index.index('x1')])
                self.y1.append(d[index.index('y1')])
                self.x2.append(d[index.index('x2')])
                self.y2.append(d[index.index('y2')])
                # print(np.array(d).astype(np.float32))

        self.time = np.array(self.time)

        self.shaped_x1 = self.make_cluster(self.time, self.x1)
        self.shaped_x2 = self.make_cluster(self.time, self.x2)

        self.shaped_x1 = self.normalize(self.shaped_x1)
        self.shaped_x2 = self.normalize(self.shaped_x2)

        print('shaped: ', len(self.shaped_x1))

        ic = probability.InfoContent()
        delayed_tau, mic = ic.get_tau(self.shaped_x1, max_tau=20)
        print('tau: ', delayed_tau)

        fig, (ax_mic, ax_delay, ax_traj) = plt.subplots(ncols=3, figsize=(18,6))
        ax_mic.plot(range(1,len(mic)+1), mic, c='black')
        ax_mic.set_title('Mutual Information Content(tau:{})'.format(delayed_tau))
        ax_mic.set_xticks(np.arange(0, 20+1, 1))
        ax_mic.grid()

        delayed_dim = 3
        delayed_out = []
        for i in reversed(range(delayed_dim)):
            delayed_out.append(np.roll(self.shaped_x1, -i*delayed_tau)[:len(self.shaped_x1)-delayed_tau])

        delayed_out = np.array(delayed_out).T
        # ax_delay = Axes3D(fig)
        ax_delay.set_title('delayed-out')
        ax_delay.plot(delayed_out[:,0], delayed_out[:,1], '.-', lw=0.1)

        ax_traj.plot(self.x1[::100], self.y1[::100], '.-', lw=0.1)
            

        re_plot = my.RecurrencePlot()

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))
        re_plot.plot(ax1, delayed_out[::2,:], eps=0.5)

        
        te_2to1, te_1to2 = [], []
        te_diff = []
        te_diff_rand = []
        N = 20

        r1 = np.random.rand(len(self.shaped_x1))
        r2 = np.random.rand(len(self.shaped_x1))
        for i in range(N, len(self.shaped_x1)):
            _te_2to1 = ic.np_get_TE(from_x=self.shaped_x2[i-N:i], to_x=self.shaped_x1[i-N:i])
            _te_1to2 = ic.np_get_TE(from_x=self.shaped_x1[i-N:i], to_x=self.shaped_x2[i-N:i])

            _te_2to1_rand = ic.np_get_TE(from_x=r2[i-N:i], to_x=r1[i-N:i])
            _te_1to2_rand = ic.np_get_TE(from_x=r1[i-N:i], to_x=r2[i-N:i])

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            te_diff.append(_te_2to1-_te_1to2)

            te_diff_rand.append(_te_2to1_rand-_te_1to2_rand)

        rate = sum(np.abs(te_diff)>max(abs(max(te_diff)),abs(min(te_diff)))/2)/len(te_diff)
        rate_rand = sum(np.abs(te_diff_rand)>max(abs(max(te_diff_rand)),abs(min(te_diff_rand)))/2)/len(te_diff_rand)
        # print('rate={:.1f}[%]\nrandom rate={:.1f}[%]'.format(rate*100, rate_rand*100))
        print('var={:.3f}\nrandom var={:.3f}'.format(np.var(te_diff), np.var(te_diff_rand)))

        '''
        for te1,te2 in zip(np.array_split(self.shaped_x1,N), np.array_split(self.shaped_x2,N)):
            _te_2to1 = ic.np_get_TE(to_x=te1, from_x=te2)
            _te_1to2 = ic.np_get_TE(to_x=te2, from_x=te1)

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            te_diff.append(_te_2to1-_te_1to2)
        '''

        # print(te_2to1, te_1to2)
        # ax2.plot(te_2to1, c='r', lw=0.7)
        # ax2.plot(te_1to2, c='b', lw=0.7)
        ax2.plot(te_diff, c='black', lw=0.7)
        ax2.plot(te_diff_rand, c='green', lw=0.7)

        # re_plot.plot(ax2, delayed_out[300:,:], eps=0.7)

        plt.show()
        
        
        # print(self.lyapunov(self.time, [self.shaped_x1, self.shaped_y1]))

    def make_cluster(self, time, data):
        shaped_data = []

        _time = (time/500).astype(np.int)
        # print('_time: ', _time)

        _time_idx = np.where(np.diff(_time)>0)[0]+1
        _time_idx = np.insert(_time_idx,0,0)
        # print('_time_idx: ', _time_idx)

        for i in range(len(_time_idx)-1):
            shaped_data.append(data[_time_idx[i+1]]-data[_time_idx[i]])

        if _time_idx[-1] != len(self.time)-1:
            shaped_data.append(data[-1]-data[_time_idx[-1]])

        return shaped_data



    def lyapunov(self, time, seqs):
        plt.figure()
        
        # print((seqs))
        # print(self.normalize(seqs))

        interval = 600
        dt = 1/interval
        for seq in seqs:
            lyapunov = []
            for i in range(math.ceil(len(seq)/interval)):
                _seq = seq[i*interval:(i+1)*interval]
                diff = np.abs(np.diff(_seq))
                lyapunov.append(np.mean(np.log(diff/dt+1e-17)-np.log(2.0)))
            plt.plot(lyapunov)

        plt.show()

        return lyapunov

    def normalize(self, seqs):
        seqs = np.array(seqs).T

        # To avoid a warning about zero-divide
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Normalize [-0.5, 0.5]
            norm = (seqs-np.min(seqs,axis=0))/(np.max(seqs,axis=0)-np.min(seqs,axis=0))-0.5

        return np.where(np.isnan(norm), 0.0, norm)



if __name__ == '__main__':
    a = Analysis('../preliminary_data/log_maeda_araki.txt')




