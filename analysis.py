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
        self.shaped_y1 = self.make_cluster(self.time, self.y1)
        self.shaped_y2 = self.make_cluster(self.time, self.y2)

        self.shaped_x1 = self.normalize(self.shaped_x1)
        self.shaped_x2 = self.normalize(self.shaped_x2)
        self.shaped_y1 = self.normalize(self.shaped_y1)
        self.shaped_y2 = self.normalize(self.shaped_y2)

        self.shaped_d1, self.shaped_d2 = [],[] 
        for ((x1,y1),(x2,y2)) in zip(zip(self.shaped_x1,self.shaped_y1), zip(self.shaped_x2,self.shaped_y2)):
            self.shaped_d1.append(np.linalg.norm([x1,y1],2))
            self.shaped_d2.append(np.linalg.norm([x2,y2],2))

        print('shaped: ', len(self.shaped_x1))

        ic = probability.InfoContent()
        delayed_tau1, mic1 = ic.get_tau(self.shaped_d1, max_tau=20)
        delayed_tau2, mic2 = ic.get_tau(self.shaped_d2, max_tau=20)
        print('tau1: ', delayed_tau1)
        print('tau2: ', delayed_tau2)

        fig, (ax_mic, ax_delay, ax_traj) = plt.subplots(ncols=3, figsize=(18,6))
        ax_mic.plot(range(1,len(mic1)+1), mic1, c='black')
        ax_mic.set_title('Mutual Information Content(tau:{})'.format(delayed_tau1))
        ax_mic.set_xticks(np.arange(0, 20+1, 1))
        ax_mic.grid()

        delayed_dim = 3
        delayed_out1, delayed_out2 = [], []
        for i in reversed(range(delayed_dim)):
            delayed_out1.append(np.roll(self.shaped_y1, -i*delayed_tau1)[:len(self.shaped_y1)-delayed_tau1])
            delayed_out2.append(np.roll(self.shaped_y2, -i*delayed_tau2)[:len(self.shaped_y2)-delayed_tau2])

        delayed_out1 = np.array(delayed_out1).T
        delayed_out2 = np.array(delayed_out2).T
        # ax_delay = Axes3D(fig)
        ax_delay.set_title('delayed-out')
        ax_delay.plot(delayed_out1[:,0], delayed_out1[:,1], '.-', lw=0.1)

        ax_traj.plot(self.x1[::100], self.y1[::100], '.-', lw=0.1)
            

        rp_plot = my.RecurrencePlot()

        fig, (ax_rp1,ax_rp2) = plt.subplots(ncols=2, figsize=(24,12))
        rp_plot.plot(ax_rp1, delayed_out1[::2,:], eps=0.5)
        rp_plot.plot(ax_rp2, delayed_out2[::2,:], eps=0.5)

        
        te_2to1, te_1to2 = [], []
        te_diff = []
        te_diff_rand = []
        N = 60

        r1 = np.random.rand(len(self.shaped_x1))
        r2 = np.random.rand(len(self.shaped_x1))
        for i in range(N, len(self.shaped_x1)):
            _te_2to1 = ic.np_get_TE(from_x=self.shaped_d2[i-N:i], to_x=self.shaped_d1[i-N:i])
            _te_1to2 = ic.np_get_TE(from_x=self.shaped_d1[i-N:i], to_x=self.shaped_d2[i-N:i])

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            te_diff.append(np.sign(_te_2to1-_te_1to2))

            _te_2to1_rand = ic.np_get_TE(from_x=r2[i-N:i], to_x=r1[i-N:i])
            _te_1to2_rand = ic.np_get_TE(from_x=r1[i-N:i], to_x=r2[i-N:i])
            te_diff_rand.append(_te_2to1_rand-_te_1to2_rand)


        rate = sum(np.abs(te_diff)>max(abs(max(te_diff)),abs(min(te_diff)))/2)/len(te_diff)
        rate_rand = sum(np.abs(te_diff_rand)>max(abs(max(te_diff_rand)),abs(min(te_diff_rand)))/2)/len(te_diff_rand)
        print('rate={:.1f}[%]\nrandom rate={:.1f}[%]'.format(rate*100, rate_rand*100))
        print('var={:.3f}\nrandom var={:.3f}'.format(np.var(te_diff), np.var(te_diff_rand)))

        '''
        for te1,te2 in zip(np.array_split(self.shaped_x1,N), np.array_split(self.shaped_x2,N)):
            _te_2to1 = ic.np_get_TE(to_x=te1, from_x=te2)
            _te_1to2 = ic.np_get_TE(to_x=te2, from_x=te1)

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            te_diff.append(_te_2to1-_te_1to2)
        '''

        fig3, (ax_te, ax_te_diff) = plt.subplots(ncols=2, figsize=(12,6))
        # print(te_2to1, te_1to2)
        ax_te.plot(te_2to1, c='r', lw=0.5)
        ax_te.plot(te_1to2, c='b', lw=0.5)
        ax_te_diff.plot(te_diff, c='black', lw=0.7)
        ax_te_diff.grid()
        # ax2.plot(te_diff_rand, c='b', lw=0.7)


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
    a = Analysis('../preliminary_data/log_hamada_oka.txt')




