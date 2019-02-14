# -*- coding: utf-8 -*-

# my library
import my_library as my
import chaotic_nn_cell
import probability
import draw

# Standard
import sys
import math
import time
import csv
import numpy as np
import threading
import warnings
import datetime
from collections import deque
from scipy.interpolate import interp1d

# Tensorflow
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow_probability as tfp
tfd = tfp.distributions

# Drawing Graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tkinter

'''
# Baysian Optimization
import GPy
import GPyOpt
'''

class CNN_Simulator:
    MODEL_PATH = '../model/'
    SOUND_PATH = '../music/'

    RANDOM_BEHAVIOR = 'RANDOM'
    CHAOTIC_BEHAVIOR = 'CHAOS'

    IMITATION_MODE = True
    CREATIVE_MODE = False

    OPTIMIZE_MODE = 'OPT'
    PREDICT_MODE = 'PREDICT'
    TRAIN_MODE = 'TRAIN'


    def __init__(self, network_mode=TRAIN_MODE, behavior_mode=CHAOTIC_BEHAVIOR):
        self.testee = '' if len(sys.argv)==1 else sys.argv[1]

        if self.testee == '':
            self.act_logfile = '../log_act.txt'
            self.LOG_PATH = '../logdir'
        else:
            self.act_logfile = '../log_act_' + self.testee + '.txt'
            self.LOG_PATH = '../logdir_' + self.testee

        self.network_mode = network_mode

        self.behavior_mode = behavior_mode

        self.colors = ['r', 'b']
        self.markers = ['.', '*']
        self.LINE_WIDTH = 1.0
        self.MARKER_SIZE = 5

        # Save the model
        self.is_save = True
        self.is_save = False

        # Drawing graphs flag
        self.is_plot = False
        self.is_plot = True

        # sequence-length at once
        self.seq_len = 20

        # self.system_seq_len = int(self.seq_len*2)
        # self.random_seq_len = int(self.seq_len/5)

        self.epoch_size = 100

        self.input_units = 4
        self.inner_units = 10
        self.output_units = 2

        # The number of inner-layers
        self.inner_layers = 1

        self.Kf = 0.1
        self.Kr = 0.9
        self.Alpha = 10.0
        self.activation = tf.nn.tanh

        # Kf, Kr, Alpha = 0, 0, 0

        # time delayed value
        self.tau = int(self.seq_len/100)
        self.tau = 10

    def weight(self, shape = []):
        initial = tf.truncated_normal(shape, stddev = 0.1, dtype=tf.float32)
        # return tf.Variable(initial)
        return initial

    def set_innerlayers(self, inputs, layers_size):

        inner_output = inputs
        for i in range(layers_size):
            with tf.name_scope('Layer' + str(i+2)):
                cell = chaotic_nn_cell.ChaoticNNCell(num_units=self.inner_units, Kf=self.Kf, Kr=self.Kr, alpha=self.Alpha, activation=self.activation)
                outputs, state = tf.nn.static_rnn(cell=cell, inputs=[inner_output], dtype=tf.float32)
                inner_output = outputs[-1]

        return inner_output

    def tf_normalize(self, v):
        with tf.name_scope('Normalize'):
            # Normalize [-0.5, 0.5]
            norm = (v-tf.reduce_min(v,0))/(tf.reduce_max(v,0)-tf.reduce_min(v,0)) - 0.5
            # norm = (v-tf.reduce_min(v))/(tf.reduce_max(v)-tf.reduce_min(v)) - 0.5

            # sign = tf.nn.relu(tf.sign(v))
            # return tf.where(tf.is_nan(norm), sign, norm)

            return tf.where(tf.is_nan(norm), tf.constant(0.0,shape=v.get_shape()), norm)


    def np_normalize(self, v):

        # To avoid a warning about zero-divide
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Normalize [-0.5, 0.5]
            norm = (v-np.min(v,axis=0))/(np.max(v,axis=0)-np.min(v,axis=0)) - 0.5

        # sign = np.where(np.sign(v)==-1, 0, v)
        # return np.where(np.isnan(norm), sign, norm)

        return np.where(np.isnan(norm), 0.0, norm)


    def inference(self, length):
        params = {}

        with tf.name_scope('Input_Layer'):
            with tf.name_scope('data'):
                inputs = tf.placeholder(dtype = tf.float32, shape = [length, self.input_units], name='inputs')
            with tf.name_scope('Wi'):
                Wi = tf.Variable(self.weight(shape=[self.input_units, self.inner_units]), name='Wi')
                # tf.summary.histogram('Wi', Wi)
                params['Wi'] = Wi

            with tf.name_scope('bi'):
                bi = tf.Variable(self.weight(shape=[1,self.inner_units]), name='bi')
                # tf.summary.histogram('bi', bi)
                params['bi'] = bi

            # input: [None, input_units]
            # in_norm = inputs
            # in_norm = self.tf_normalize(inputs[:,0:2])
            # in_norm = tf.concat([in_norm, inputs[:,2:4]], 1)
            in_norm = self.tf_normalize(inputs)
            # in_norm = inputs
            fi = tf.matmul(in_norm, Wi) + bi
            sigm = tf.nn.sigmoid(fi)

        inner_output = self.set_innerlayers(sigm, self.inner_layers)

        with tf.name_scope('Output_Layer'):
            if self.inner_units % self.output_units != 0:
                print("Can't make the clusters")
                exit()
            cluster_size = int(self.inner_units / self.output_units)
            print('cluster: ', cluster_size)
            one = [1]*cluster_size
            one.extend([0]*(self.inner_units-cluster_size))
            ones = []
            for i in range(self.output_units):
                ones.append(np.roll(one, cluster_size*i))
            ones = np.reshape(ones, [self.output_units, self.inner_units]).T

            Io = tf.cast(ones, tf.float32)

            with tf.name_scope('Wo'):
                Wo = tf.Variable(self.weight(shape=[self.inner_units, self.output_units]), name='Wo')
                # tf.summary.histogram('Wo', Wo)
                params['Wo'] = Wo

            with tf.name_scope('bo'):
                bo = tf.Variable(self.weight(shape=[1, self.output_units]), name='bo')
                # tf.summary.histogram('bo', bo)
                params['bo'] = bo

            # fo = tf.matmul(inner_output, tf.multiply(Wo, Io))
            fo = tf.matmul(inner_output, Wo) + bo
            outputs = self.tf_normalize(fo)
            # outputs = fo

        return in_norm, inputs, outputs, params

    def tf_get_lyapunov(self, seq, length):
        '''
        本来リアプノフ指数はmean(log(f'))だが、f'<<0の場合に-Infとなってしまうため、
        mean(log(1+f'))とする。しかし、それだとこの式ではカオス性を持つかわからなくなってしまう。
        カオス性の境界log(f')=0のとき、f'=1
        log(1+f')+alpha=log(2)+alpha=0となるようにalphaを定めると
        alpha=-log(2)
        となるため、プログラム上のリアプノフ指数の定義は、
        mean(log(1+f')-log(2))とする（0以上ならばカオス性を持つという性質は変わらない）
        '''
        
        dt = 1/length
        diff = tf.abs(seq[1:]-seq[:-1])
        lyapunov = tf.reduce_mean(tf.log1p(diff/dt)-tf.log(tf.cast(2.0, tf.float32)))

        return lyapunov

    def x_input(self, x, y):
        ic = probability.InfoContent()
        te_term = 0
        for i in range(self.output_units):
            for j in range(self.output_units):
                _x, _y = x[:,i], y[:,j]
                # TE(y->x)
                _en, _pdf = ic.tf_get_TE(_x, _y, self.seq_len)
                te_term += _en
        return te_term

    def x_output(self, x, y):
        ic = probability.InfoContent()
        te_term = 0
        for i in range(self.output_units):
            for j in range(self.output_units):
                _x, _y = x[:,i], y[:,j]
                # TE(x->y)
                _en, _pdf = ic.tf_get_TE(_x, _y, self.seq_len)
                te_term += _en
        return te_term


    def loss(self, inputs, outputs, length, mode):
        # if mode==True: TE(y->x)

        # Laypunov-Exponent
        with tf.name_scope('Lyapunov'):
            lyapunov = []
            for i in range(self.output_units):
                lyapunov.append(self.tf_get_lyapunov(outputs[:,i], length))
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])

        with tf.name_scope('TE-Loss'):
            '''
            ic = probability.InfoContent()

            # The empty constant Tensor to get unit-num.
            output_units_ = tf.constant(0, shape=[self.output_units])
            input_units_ = tf.constant(0, shape=[self.input_units])

            x, x_units_, y, y_units_ = tf.cond(mode,
                    lambda:(outputs, output_units_, inputs, input_units_),
                    lambda:(inputs, input_units_, outputs, output_units_))
            '''
            '''
            x, x_units, y, y_units = tf.cond(mode,
                    lambda:(outputs, int(outputs.get_shape()[1]), inputs, input_units_),
                    lambda:(inputs, int(inputs.get_shape()[1]), outputs, output_units_))

            print('x_units: ', x_units)


            # x_units = int(x_units_.get_shape()[0])
            # y_units = int(y_units_.get_shape()[0])

            # x_units, y_units = 1,1

            entropy = []
            for i in range(x_units):
                _entropy = 0
                for j in range(y_units):
                    _x, _y = x[:,j], y[:,i]
                    # TE(y->x)
                    _en, _pdf = ic.tf_get_TE(_x, _y, self.seq_len)
                    _entropy += _en
                entropy.append(_entropy)
                tf.summary.scalar('entropy{}'.format(i), entropy[i])

            te_term = 0
            for i in range(x_units):
                for j in range(y_units):
                    _x, _y = x[:,j], y[:,i]
                    # TE(y->x)
                    _en, _pdf = ic.tf_get_TE(_x, _y, self.seq_len)
                    te_term += _en
            '''

            te_term = tf.cond(mode, lambda:self.x_output(outputs, inputs), lambda:self.x_input(inputs, outputs))
            tf.summary.scalar('entropy', te_term)


            # 出力値同士の差別化項
            # diff_term = tf.reduce_sum(tf.log(tf.abs(tf.abs(outputs[:,0])-tf.abs(outputs[:,1]))+1e-10))

            # 差別化するのは人間をモデル化する上でおかしい
            # 同じパターンがこないようにしたい=飽きと解釈できる
            # ただし、絶対座標にはそのままは適用できない
            # diff_term = tf.log(tf.var(outputs[:,0])+1e-10)+tf.log(tf.var(outputs[:,1])+1e-10)
            # _, var_x = tf.nn.moments(outputs[:,0], [0])
            # _, var_y = tf.nn.moments(outputs[:,1], [0])
            # diff_term = tf.log(var_x+1e-10)+tf.log(var_y+1e-10)

            # theta = tf.angle(tf.complex(outputs[:,1][1:]-outputs[:,1][:-1], outputs[:,0][1:]-outputs[:,0][:-1]))
            theta = tf.atan((outputs[:,1][1:]-outputs[:,1][:-1])/(outputs[:,0][1:]-outputs[:,0][:-1]+1e-10))
            theta2 = tf.atan((inputs[:,1][1:]-inputs[:,1][:-1])/(inputs[:,0][1:]-inputs[:,0][:-1]+1e-10))
            '''
            # theta = (theta-tf.reduce_min(theta)) / (tf.reduce_max(theta)-tf.reduce_min(theta))
            cor = []
            N = int((length-1)/2)
            for i in range(N):
                _cor = 0
                for j in range(N):
                    _cor += theta[j] * theta[j+i]
                cor.append(_cor)

            cor0 = cor[0]
            if cor[0] != 0:
                for i in range(N):
                    cor[i] = cor[i]/cor0

            # _a = list(np.array(cor[1:])-1 + np.pi/2 - 1e-10)
            # diff_term = tf.reduce_sum(tf.tan(tf.nn.relu(_a)))
            _a = list(-1 * np.array(cor[1:]) +1e-10)
            # diff_term = tf.reduce_mean(tf.log(_a+tf.reduce_max(cor)))
            # diff_term2 = (tf.reduce_max(cor))
            # diff_term = tf.reduce_sum(tf.tan(tf.nn.relu(cor)-tf.reduce_max(cor)+np.pi/2-1e-10))
            '''
            cor = tf.contrib.distributions.auto_correlation(theta)
            cor = tf.where(tf.is_nan(cor), tf.constant(0.0,shape=cor.get_shape()), cor)
            cor2 = tf.contrib.distributions.auto_correlation(theta2)
            cor2 = tf.where(tf.is_nan(cor2), tf.constant(0.0,shape=cor.get_shape()), cor2)
            
            '''
            _, var = tf.nn.moments(cor[1:]-cor[:-1], [0])
            var = tf.cond(tf.is_nan(var), lambda:0., lambda:var)
            # diff_term = tf.log(var**3+1e-100)
            # y=xの直線の時の分散は0.20
            var_th = 0.20
            logy0 = 1-var_th*2
            # diff_term = -tf.log(var+logy0) * tf.abs(te_term)/tf.log(var_th+logy0)
            diff_term = tf.log(1/var_th * tf.exp(te_term) * var + 1e-100)
            # diff_term = -var*tf.abs(te_term)/0.2
            # diff_term = tf.log(10/3 * )
            
            diff_term2 = tf.reduce_min(tf.contrib.distributions.auto_correlation(theta))
            '''
            # diff_term = tf.reduce_max(cor[1:]) * tf.abs(te_term)
            diff_term = (tf.pow(tf.reduce_max(cor[1:]),2)*10)
            diff_term2 = (tf.pow(tf.reduce_max(cor2[1:]),2)*10)
            # diff_term = tf.exp(tf.sign(tf.reduce_max(cor[1:]))*tf.pow(tf.reduce_max(cor[1:]),1))
            # diff_term = tf.reduce_max(cor[1:])
            # tf.summary.scalar('CCF: ', tf.reduce_max(ccf))
            tf.summary.scalar('error_CCF', diff_term)
            tf.summary.scalar('error_CCF2', diff_term2)

            # return -te_term, te_term, diff_term
            return -(te_term-diff_term-diff_term2), te_term, diff_term, theta

        '''
        with tf.name_scope('loss_lyapunov'):
            # リアプノフ指数が増加するように誤差関数を設定
            lyapunov = []
            loss = []
            for i in range(self.output_units):
                lyapunov.append(get_lyapunov(outputs[:,i]))
                # loss.append(1/(1+tf.exp(lyapunov[i])))
                # loss.append(tf.exp(-lyapunov[i]))
                # loss.append(-lyapunov[i])

            # リアプノフ指数を取得(評価の際は最大リアプノフ指数をとる)
            # return tf.reduce_sum(loss), lyapunov
            return tf.reduce_max(loss), lyapunov
            # return tf.reduce_min(loss), lyapunov
            # return loss, lyapunov
        '''


    def train(self, error, update_params):
        # return tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error)
        with tf.name_scope('training'):
            # opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            opt = tf.train.AdamOptimizer(0.001)
            _grads = opt.compute_gradients(error, var_list=update_params)
            # _grads = [(grad1,var1),(grad2,var2),...]

            # If the system cannot update weights(grad:Nan), it occurs an error.
            # So Nan-value must replace to zeros.
            grads = []
            for g in _grads:
                _g = tf.where(tf.is_nan(g[0]), tf.zeros_like(g[0]), g[0])
                grads.append((_g, g[1]))

            # training = opt.minimize(grads, var_list=update_params)
            training = opt.apply_gradients(grads)

        return training, grads


    def np_get_lyapunov(self, seq):
        # print('Measuring lyapunov...')
        dt = 1/len(seq)
        diff = np.abs(np.diff(seq))
        lyapunov = np.mean(np.log1p(diff/dt)-np.log(2.0))

        return lyapunov

    def network_bg(self, event, proctime, num):
        for i in range(num):
            time.sleep(proctime/num)
            event.set_network_interval()
            print('Add background')
            

    def human_agent_interaction(self):
        sess = tf.InteractiveSession()

        with tf.name_scope('Mode'):
            Mode = tf.placeholder(dtype = tf.bool, name='Mode')
            tf.summary.scalar('Mode', tf.cast(Mode, tf.int32))

        norm_in, inputs, outputs, params = self.inference(self.seq_len)
        Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

        error, te_term, diff_term, param = self.loss(norm_in, outputs, self.seq_len, Mode)
        tf.summary.scalar('error', error)
        # tf.summary.scalar('te_term', te_term)
        # tf.summary.scalar('diff_term', diff_term)
        train_step, grad = self.train(error, [Wi, bi, Wo, bo])

        merged = tf.summary.merge_all()


        if tf.gfile.Exists(self.LOG_PATH):
            tf.gfile.DeleteRecursively(self.LOG_PATH)
        writerA = tf.summary.FileWriter(self.LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())

        event = draw.Event(draw.Event.USER_MODE, testee=self.testee)

        # True: Following, False: Creative
        modeA = self.IMITATION_MODE
        modeA = self.CREATIVE_MODE
        event.set_system_mode(modeA)


        trajectoryA = []
        trajectoryB = []

        # Wait Init process
        print("SYSTEM_WAIT")
        while not event.get_startup_signal():
            time.sleep(0.1)
        diff, diff_pos, is_drawing = event.get_diff()

        # Using 3-dim spline function
        N = self.seq_len
        r = np.random.rand(N, 2)-0.5
        r[0],r[2] = [0]*2, [0]*2
        # r = np.insert(r, 0, outB[-1,0:2], axis=0)
        x = np.linspace(0, N-1, num=N)
        xnew = np.linspace(0, 2, num=self.seq_len)
        outB = np.zeros([self.seq_len, self.input_units])
        outB[:,:] = diff+diff_pos
        for i in range(2):
            outB[:,i] = interp1d(x, r[:,i], kind='cubic')(xnew)
        outB = outB.tolist()
        print(outB)


        # is_drawing = True
        is_changemode = True
        _premodeA = modeA
        mode_switch = []
        outA_all, outB_all = [], []
        epoch = 0
        proctime = 1
        tmp_error = deque([])
        switch_prob = []
        error_slope = []
        
        f = open(self.act_logfile, mode='w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['seq_len:{}'.format(self.seq_len)])
        writer.writerow(['epoch','time[ms]','loss','TE_term','DIFF_term'])
        starttime = event.get_starttime()


        # for epoch in range(self.epoch_size):
        while not event.get_systemstop_signal():
            print('### epoch:{}, mode:{} ###'.format(epoch, modeA))

            network_proctime_st = datetime.datetime.now()

            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:

                # RANDOM INPUT
                outB = np.array(outB)
                var = sum(np.var(outB[:,0:2], axis=0))
                print('var: ', var)
                if var == 0:
                    print('INPUT=RANDOM')
                    # outB[:,0:2] = np.random.rand(self.seq_len, 2)-0.5
                    N = self.seq_len
                    r = np.random.rand(N, 2)-0.5
                    r[0],r[2] = [0]*2, [0]*2
                    # r = np.insert(r, 0, outB[-1,0:2], axis=0)
                    x = np.linspace(0, N-1, num=N)
                    xnew = np.linspace(0, 2, num=self.seq_len)
                    for i in range(2):
                        outB[:,i] = interp1d(x, r[:,i], kind='cubic')(xnew)
                        # outB.append(interp1d(range(N), r[:,i], kind='cubic')(xnew))

                print('outB: ', outB)
                outB = outB.tolist()

                feed_dictA = {inputs:outB, Mode:modeA}
                outA = sess.run(outputs, feed_dict=feed_dictA)
                print('outA: ', outA)

                # Using 3-dim spline function
                x = np.linspace(0, self.seq_len-1, num=self.seq_len)
                xnew = np.linspace(0, 1, num=10)
                spline_outA = []
                for i in range(self.output_units):
                    spline_outA.append(interp1d(x, np.array(outA)[:,i], kind='cubic')(xnew))
                spline_outA = np.array(spline_outA).T.tolist()



                if epoch % self.seq_len == 0:
                    # print('outA: ', outA)
                    summaryA, _t, gradientsA = sess.run([merged, train_step, grad], feed_dictA)
                    _error, _te_term, _diff_term, _param = sess.run([error, te_term, diff_term, param], feed_dictA)
                    t = int((datetime.datetime.now()-starttime).total_seconds()*1000)
                    writer.writerow([epoch, t, _error, _te_term, _diff_term])
                    writerA.add_summary(summaryA, epoch)

                    for (g, v) in gradientsA:
                        print('gradA: ', g[0][0:5])
                    print('te_term: ', _te_term)
                    print('diff_term: ', _diff_term)
                    print('theta: ', np.mean(_param)*180/np.pi)

                    # Boredom
                    # A: Error-Value is not change
                    # B: Error-Value increases 
                    bored_len = 10
                    if is_changemode and self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                        tmp_error.append(_error)
                        if len(tmp_error) > bored_len:
                            tmp_error.popleft()

                        if len(tmp_error) == bored_len:
                            a, b = np.polyfit(range(bored_len), tmp_error, 1)
                            error_slope.append(a)
                            print('slope: ', a)
                            
                            # Degree of the slope(S): -10deg<S<10deg or S>80deg
                            if abs(a) < np.tan(10/180*np.pi):
                                print('[Change Mode A]: ', a)
                                tmp_error = deque([])
                                modeA = not modeA
                                mode_switch.append(epoch)
                                event.set_system_mode(modeA)
                            elif a > np.tan(80/180*np.pi):
                                print('[Change Mode B]: ', a)
                                tmp_error = deque([])
                                modeA = not modeA
                                mode_switch.append(epoch)
                                event.set_system_mode(modeA)


            if self.behavior_mode == self.RANDOM_BEHAVIOR:

                # Using 3-dim spline function
                N = 4
                if epoch != 0:
                    _r1 = r[1]
                    _r2 = r[2]
                r = np.random.rand(N, self.output_units)-0.5
                if epoch != 0:
                    r[0] = _r1
                    r[1] = _r2
                # if epoch != 0:
                # r[0,:] = past_outA[-1]
                print('r: ', r)
                x = np.linspace(0, N-1, num=N)
                xnew = np.linspace(0, 1, num=10)
                spline_outA = []
                for i in range(self.output_units):
                    spline_outA.append(interp1d(x, r[:,i], kind='cubic')(xnew))
                spline_outA = np.array(spline_outA).T.tolist()
                print('spline: ', spline_outA)


            mag = 5
            for out in spline_outA:
                event.set_movement(out, mag)
                '''
                while event.get_is_output() and not event.get_systemstop_signal():
                    print('Execute...')
                    # time.sleep(0.001)
                    pass
                '''
                time.sleep(0.01)

            diff, diff_pos, is_drawing = event.get_diff()
            outB.pop(0)
            outB.append(diff+diff_pos)

            '''
            # outA = np.random.rand(self.seq_len, self.output_units)-0.5
            outA = []
            for i in range(int(np.ceil(self.seq_len/3))):
                r = (np.random.rand(1,self.output_units)-0.5).tolist()
                if i == np.ceil(self.seq_len/3)-1 and self.seq_len%3 != 0:
                    outA.extend(r*(self.seq_len%3))
                else:
                    outA.extend(r*3)
            outA = np.array(outA)
            '''
            '''
            # ネットワークの学習処理時間を埋めるための処理
            networkbg_thread = threading.Thread(target=self.network_bg, args=[event, proctime, 3])
            networkbg_thread.daemon = True
            networkbg_thread.start()
            '''

            network_proctime_en = datetime.datetime.now()
            proctime = (network_proctime_en-network_proctime_st).total_seconds()
            print('proctime:{}s '.format(proctime))

            if not is_drawing:
                break

            # outB = np.array(outB)/mag

            # outA_all.extend(list(np.array(outA)[:,0]))
            # outB_all.extend(list(np.array(outB)[:,0]))

            # outA_all.extend(outA)
            # outB_all.extend(outB)

            '''
            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:

                # Mesuring the amount of activity
                d1 = np.mean(abs(np.diff(outB[:,0])))
                d2 = np.mean(abs(np.diff(outB[:,1])))
                self.activity = np.mean([d1,d2])
                print(self.activity)
                f.write(str(self.activity)+'\n')

                if is_changemode and self.activity < 0.08:
                    print('[Change Mode]', self.activity)
                    modeA = not modeA
                    mode_switch.append(epoch)
                    event.set_system_mode(modeA)
            '''


            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                if _premodeA != modeA:
                    mode_switch.append(epoch)
                    _premodeA = modeA

                past_outA = outA
                past_outB = outB

            epoch += 1

        # print('outA: ', np.array(trajectoryA))
        # print('outB: ', np.array(trajectoryB))

        writer.writerow(['mode_switch'] + np.unique(mode_switch).astype(int).tolist())
        writer.writerow(['slope'] + error_slope)
        f.close()
        mode_switch = np.unique(mode_switch) * self.seq_len
        print('mode_switch: ', mode_switch)
        print('slope: ', error_slope)

        '''
        print('switch_prob: ', switch_prob)
        plt.figure(figsize=(10,6))
        plt.plot(switch_prob, marker='.')
        plt.show()

        fig4, (ax_vec1) = plt.subplots(ncols=1, figsize=(6,6))
        # ax_vec1.quiver(outB_all[:,0], np.zeros(len(outB_all)), outA_all[:,0], np.zeros(len(outB_all)), angles='xy')
        ax_vec1.quiver(outB_all[self.seq_len*1:self.seq_len*3][:,0], outB_all[self.seq_len*1:self.seq_len*3][:,1], outA_all[self.seq_len*1:self.seq_len*3][:,0], outA_all[self.seq_len*1:self.seq_len*3][:,1], angles='xy', scale_units='xy')
        # ax_vec1.set_title('system-vector')

        ax_vec2.quiver(outB_all[self.seq_len*3:self.seq_len*5][:,0], outB_all[self.seq_len*3:self.seq_len*5][:,1], outA_all[self.seq_len*3:self.seq_len*5][:,0], outA_all[self.seq_len*3:self.seq_len*5][:,1], angles='xy', scale_units='xy')
        # ax_vec2.set_title('user-vector')

        ax_vec3.quiver(outB_all[self.seq_len*5:self.seq_len*7][:,0], outB_all[self.seq_len*5:self.seq_len*7][:,1], outA_all[self.seq_len*5:self.seq_len*7][:,0], outA_all[self.seq_len*5:self.seq_len*7][:,1], angles='xy', scale_units='xy')
        # ax_vec.quiver(outB_all[:,0], outB_all[:,1], outB_all[:,0], outB_all[:,1], angles='xy')

        '''

        print('Finish')


if __name__ == "__main__":
    simulator = CNN_Simulator(network_mode=CNN_Simulator.TRAIN_MODE, behavior_mode=CNN_Simulator.CHAOTIC_BEHAVIOR)
    # simulator = CNN_Simulator(network_mode=CNN_Simulator.TRAIN_MODE, behavior_mode=CNN_Simulator.RANDOM_BEHAVIOR)
    # simulator.learning1()
    # simulator.robot_robot_interaction()
    simulator.human_agent_interaction()

    '''
    calc_thread = threading.Thread(target=test)
    # calc_thread.daemon = True
    calc_thread.start()
    '''

