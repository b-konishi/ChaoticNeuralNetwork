import sys
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
        self.time, self.mode = [],[]
        self.x1, self.y1, self.x2, self.y2 = [],[],[],[]
        self.relx, self.rely = [],[]
        self.prev_ver = False

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            [next(reader) for _ in range(2)]
            # Read an index
            index = next(reader)
            if len(index) == 5:
                self.prev_ver = True
            for d in reader:
                d = np.array(d).astype(np.float32)
                self.time.append(d[index.index('time[ms]')])
                self.x1.append(d[index.index('x1')])
                self.y1.append(d[index.index('y1')])
                self.x2.append(d[index.index('x2')])
                self.y2.append(d[index.index('y2')])
                if not self.prev_ver:
                    self.mode.append(d[index.index('mode')])
                    self.relx.append(d[index.index('x1-x2')])
                    self.rely.append(d[index.index('y1-y2')])

        self.time = np.array(self.time)


        self.shaped_x1 = self.make_cluster(self.time, self.x1)
        self.shaped_x2 = self.make_cluster(self.time, self.x2)
        self.shaped_y1 = self.make_cluster(self.time, self.y1)
        self.shaped_y2 = self.make_cluster(self.time, self.y2)

        if not self.prev_ver:
            self.shaped_mode = self.make_cluster(self.time, self.mode)

            self.shaped_relx = self.make_cluster(self.time, self.relx)
            self.shaped_rely = self.make_cluster(self.time, self.rely)

        # 差分データを取得
        self.shaped_dx1 = np.diff(self.shaped_x1)
        self.shaped_dx2 = np.diff(self.shaped_x2)
        self.shaped_dy1 = np.diff(self.shaped_y1)
        self.shaped_dy2 = np.diff(self.shaped_y2)

        self.shaped_dx1 = self.normalize(self.shaped_dx1)
        self.shaped_dx2 = self.normalize(self.shaped_dx2)
        self.shaped_dy1 = self.normalize(self.shaped_dy1)
        self.shaped_dy2 = self.normalize(self.shaped_dy2)

        # self.shaped_dx1 = [i for i in self.shaped_dx1 if not i==0]

        self.shaped_d1, self.shaped_d2 = [],[] 
        for ((x1,y1),(x2,y2)) in zip(zip(self.shaped_dx1,self.shaped_dy1), zip(self.shaped_dx2,self.shaped_dy2)):
            self.shaped_d1.append(np.linalg.norm([x1,y1],2))
            self.shaped_d2.append(np.linalg.norm([x2,y2],2))

        if not self.prev_ver:
            self.shaped_rel = []
            for (x,y) in zip(self.shaped_relx, self.shaped_rely):
                self.shaped_rel.append(np.linalg.norm([x,y],2))

        ##### Delayed OUT DATA #####
        delayed_out1, delayed_out2 = self.delay_coord_analysis(self.shaped_dx1, self.shaped_dx2)

        ##### Trajectory Plot #####
        fig, (ax_traj1,ax_traj2) = plt.subplots(ncols=2, figsize=(12,6))
        # ax_traj1.set_title('Trajectory(red:Human,Green:System)')
        # ax_traj.plot(self.x1[::100], self.y1[::100], '.-', lw=0.1)
        ax_traj1.set_title('Trajectory for Human')
        ax_traj1.plot(self.shaped_x1[::1], self.shaped_y1[::1], c='red', lw=1)
        ax_traj2.set_title('Trajectory for System')
        ax_traj2.plot(self.shaped_x2[::1], self.shaped_y2[::1], c='green', lw=1)
            

        ##### Recurrence Plot #####
        rp_plot = my.RecurrencePlot()

        fig, (ax_rp1,ax_rp2) = plt.subplots(ncols=2, figsize=(24,12))
        ax_rp1.set_title('RP for Human')
        rp_plot.plot(ax_rp1, delayed_out1[::1,:], eps=0.5)
        ax_rp2.set_title('RP for System')
        rp_plot.plot(ax_rp2, delayed_out2[::1,:], eps=0.5)

        
        te_2to1, te_1to2 = [], []
        te_diff = []
        te_diff_rand = []
        N = 60

        ic = probability.InfoContent()
        r1 = np.random.rand(len(self.shaped_dx1))
        r2 = np.random.rand(len(self.shaped_dx1))
        for i in range(N, len(self.shaped_dx1)):
            _te_2to1 = ic.np_get_TE(from_x=self.shaped_d2[i-N:i], to_x=self.shaped_d1[i-N:i])
            _te_1to2 = ic.np_get_TE(from_x=self.shaped_d1[i-N:i], to_x=self.shaped_d2[i-N:i])

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            # te_diff.append(np.sign(_te_2to1-_te_1to2))
            te_diff.append(_te_2to1-_te_1to2)

            _te_2to1_rand = ic.np_get_TE(from_x=r2[i-N:i], to_x=r1[i-N:i])
            _te_1to2_rand = ic.np_get_TE(from_x=r1[i-N:i], to_x=r2[i-N:i])
            te_diff_rand.append(_te_2to1_rand-_te_1to2_rand)


        rate = sum(np.abs(te_diff)>max(abs(max(te_diff)),abs(min(te_diff)))/2)/len(te_diff)
        rate_rand = sum(np.abs(te_diff_rand)>max(abs(max(te_diff_rand)),abs(min(te_diff_rand)))/2)/len(te_diff_rand)
        print('rate={:.1f}[%]\nrandom rate={:.1f}[%]'.format(rate*100, rate_rand*100))
        print('var={:.3f}\nrandom var={:.3f}'.format(np.var(te_diff), np.var(te_diff_rand)))

        '''
        for te1,te2 in zip(np.array_split(self.shaped_dx1,N), np.array_split(self.shaped_dx2,N)):
            _te_2to1 = ic.np_get_TE(to_x=te1, from_x=te2)
            _te_1to2 = ic.np_get_TE(to_x=te2, from_x=te1)

            te_2to1.append(_te_2to1)
            te_1to2.append(_te_1to2)
            te_diff.append(_te_2to1-_te_1to2)
        '''

        fig3, (ax_te, ax_te_diff) = plt.subplots(ncols=2, figsize=(12,6))
        # print(te_2to1, te_1to2)
        ax_te.set_title('TransferEntropy')
        ax_te.plot(te_2to1, c='r', lw=1, label='sys->human')
        ax_te.plot(te_1to2, c='b', lw=1, label='human->sys')
        # _mode = np.where(self.shaped_mode, max([max(te_2to1),max(te_1to2)]), min([min(te_2to1),min(te_1to2)]))
        # ax_te.plot(_mode, c='black', lw=1, label='mode')
        ax_te.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
        ax_te_diff.set_title('TransferEntropy Diff')
        ax_te_diff.plot(te_diff, c='black', lw=0.7)
        ax_te_diff.grid()
        # ax2.plot(te_diff_rand, c='b', lw=0.7)


        if not self.prev_ver:
            fig4, (ax_rel, ax_mode) = plt.subplots(ncols=2, figsize=(12,6))
            # print(te_2to1, te_1to2)
            ax_rel.set_title('Relation(Human-System)')
            ax_rel.plot(self.shaped_rel, c='r', lw=1)
            ax_mode.set_title('Mode')
            ax_mode.plot(self.shaped_mode, c='r', lw=1)



        plt.show()
        
        
        # print(self.lyapunov(self.time, [self.shaped_dx1, self.shaped_dy1]))

    def delay_coord_analysis(self, data1, data2):
        _max_tau = 50
        ic = probability.InfoContent()
        delayed_tau1, mic1 = ic.get_tau(data1[100:], max_tau=_max_tau)
        delayed_tau2, mic2 = ic.get_tau(data2[100:], max_tau=_max_tau)
        print('tau1: ', delayed_tau1)
        print('tau2: ', delayed_tau2)

        fig, (ax_mic, ax_delay) = plt.subplots(ncols=2, figsize=(12,6))
        ax_mic.set_title('Mutual Information Content(tau:{},{})'.format(delayed_tau1,delayed_tau2))
        ax_mic.plot(range(1,len(mic1)+1), mic1, c='red')
        ax_mic.plot(range(1,len(mic2)+1), mic2, c='green')
        ax_mic.set_xticks(np.arange(0, _max_tau+1, 1))
        ax_mic.grid()

        delayed_dim = 3
        delayed_out1, delayed_out2 = [], []
        for i in (range(delayed_dim)):
            delayed_out1.append(np.roll(data1, -i*delayed_tau1)[:len(data1)-delayed_tau1])
            delayed_out2.append(np.roll(data2, -i*delayed_tau2)[:len(data2)-delayed_tau2])

        delayed_out1 = np.array(delayed_out1).T
        delayed_out2 = np.array(delayed_out2).T
        # ax_delay = Axes3D(fig)
        ax_delay.set_title('delayed-out')
        ax_delay.plot(delayed_out1[:,0], delayed_out1[:,1], '.-', lw=0.1)

        return delayed_out1, delayed_out2

    def make_cluster(self, time, data):

        # 500ms 間隔に変換
        _time = (time/500).astype(np.int)
        # print('_time: ', _time)

        # 重複する時刻を除いたインデックスを取得 
        _time_idx = np.where(np.diff(_time)>0)[0]+1
        _time_idx = np.insert(_time_idx,0,0)
        # print('_time_idx: ', _time_idx)

        # サンプリングした時刻のデータを取得
        shaped_data = []
        for i in range(len(_time_idx)):
            shaped_data.append(data[_time_idx[i]])


        '''
        # 差分データを取得
        shaped_data = []
        for i in range(len(_time_idx)-1):
            shaped_data.append(data[_time_idx[i+1]]-data[_time_idx[i]])

        if _time_idx[-1] != len(self.time)-1:
            shaped_data.append(data[-1]-data[_time_idx[-1]])
        '''

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
    args = sys.argv
    a = Analysis(args[1])




