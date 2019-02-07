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

        self.system_seq_len = int(self.seq_len*2)
        self.random_seq_len = int(self.seq_len/5)

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
            in_norm = self.tf_normalize(inputs)
            in_norm = inputs
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
            # theta = tf.atan((inputs[:,1][1:]-inputs[:,1][:-1])/(inputs[:,0][1:]-inputs[:,0][:-1]+1e-10))
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
            diff_term = tf.reduce_max(cor[1:]) * tf.abs(te_term)
            # tf.summary.scalar('CCF: ', tf.reduce_max(ccf))
            tf.summary.scalar('error_CCF', diff_term)
            tf.summary.scalar('error_CCF2', tf.reduce_mean(theta)*180/np.pi)

            # return -te_term, te_term, diff_term
            return -(te_term-diff_term), te_term, diff_term, theta

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

        # fig, ax = plt.subplots(1, 1)
        # fig = plt.figure(figsize=(10,6))
        # fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,6))

        event = draw.Event(draw.Event.USER_MODE, testee=self.testee)
        re_plot = my.RecurrencePlot()

        # True: Following, False: Creative
        modeA = self.IMITATION_MODE
        modeA = self.CREATIVE_MODE
        event.set_system_mode(modeA)

        trajectoryA = []
        trajectoryB = []
        init_pos1, init_pos2 = event.get_init_pos()

        '''
        outB = []
        for i in range(int(np.ceil(self.seq_len/3))):
            r = np.random.rand(1,self.input_units).tolist()
            if i == np.ceil(self.seq_len/3)-1 and self.seq_len%3 != 0:
                outB.extend(r*(self.seq_len%3))
            else:
                outB.extend(r*3)
        outB = np.array(outB)
        '''

        # Using 3-dim spline function
        N = self.random_seq_len
        r = np.random.rand(N, self.input_units)-0.5
        x = np.linspace(-0.5,0.5,num=N)
        xnew = np.linspace(-0.5,0.5,num=self.seq_len)
        f_cs = []
        outB = []
        for i in range(self.input_units):
            outB.append(interp1d(x, r[:,i], kind='cubic')(xnew))
        print('outB: ', outB)
        outB = np.array(outB).T


        is_drawing = True
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
        writer.writerow(['loss','TE_term','DIFF_term'])

        print("SYSTEM_WAIT")
        while not event.get_startup_signal():
            time.sleep(0.1)
            pass
        # for epoch in range(self.epoch_size):
        while not event.get_systemstop_signal():
            print('### epoch:{}, mode:{} ###'.format(epoch, modeA))

            network_proctime_st = datetime.datetime.now()

            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                # event.set_network_interval(proctime, 1)
                # ネットワークの学習処理時間を埋めるための処理
                networkbg_thread = threading.Thread(target=self.network_bg, args=[event, proctime, 3])
                networkbg_thread.daemon = True
                networkbg_thread.start()

                print(outB)

                feed_dictA = {inputs:outB, Mode:modeA}
                _outA, _error, _te_term, _diff_term, gradientsA = sess.run([outputs, error, te_term, diff_term, grad], feed_dict=feed_dictA)
                writer.writerow([_error, _te_term, _diff_term])
                p = sess.run(param, feed_dict=feed_dictA)
                print('param: ', np.mean(p)*180/np.pi)

                if epoch % 1 == 0:
                    for (g, v) in gradientsA:
                        print('gradA: ', g[0][0:5])
                    print('te_term: ', _te_term)
                    print('diff_term: ', _diff_term)
                    # print('outA: ', outA)

                summaryA, _ = sess.run([merged, train_step], feed_dictA)
                writerA.add_summary(summaryA, epoch)

                # Using 3-dim spline function
                # if epoch != 0:
                # outA[0,:] = past_outA[-1]
                print('outA: ', _outA)
                x = np.linspace(-0.5,0.5,num=self.seq_len)
                xnew = np.linspace(-0.5,0.5,num=self.system_seq_len)
                f_cs = []
                outA = []
                for i in range(self.output_units):
                    outA.append(interp1d(x, _outA[:,i], kind='cubic')(xnew))
                outA = np.array(outA).T
                print('outA: ', outA)

                size = 4
                if epoch != 0:
                    outA = np.insert(outA, 0, past_outA[-(size-1):], axis=0)
                for i in range(len(outA)-(size-1)):
                    _out = 0
                    for j in range(size):
                        _out += outA[i+j,:]
                    outA[i+(size-1),:] = _out/size

                if epoch != 0:
                    outA = outA[size-1:]
                print('len(outA): ',len(outA))


            if self.behavior_mode == self.RANDOM_BEHAVIOR:
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

                # Using 3-dim spline function
                N = self.random_seq_len
                r = np.random.rand(N, self.output_units)-0.5
                # if epoch != 0:
                # r[0,:] = past_outA[-1]
                x = np.linspace(-0.5,0.5,num=N)
                xnew = np.linspace(-0.5,0.5,num=self.seq_len)
                f_cs = []
                outA = []
                for i in range(self.output_units):
                    outA.append(interp1d(x, r[:,i], kind='cubic')(xnew))
                outA = np.array(outA).T
                print('r: ', r)
                print(outA)


                '''
                # 前回の出力からの移動平均
                past_size = 4
                if epoch != 0:
                    _out = outA[0,:]
                    for i in range(past_size-1):
                        _out += past_outA[-(i+1),:]
                    outA[0,:] = _out/past_size
                '''
            '''
            for i in range(len(outA)-(size-1)):
                _out = 0
                for j in range(size):
                    _out += outA[i+j,:]
                outA[i+(size-1),:] = _out/size
            '''


            network_proctime_en = datetime.datetime.now()
            proctime = (network_proctime_en-network_proctime_st).total_seconds()
            print('proctime:{}s '.format(proctime))

            outB = []
            mag = 30
            for i in range(self.seq_len):
                event.set_movement(np.array(outA[i]), mag)

                diff, diff_pos, is_drawing = event.get_diff()
                outB.append(diff+diff_pos)

                # ADJUST
                time.sleep(0.05)
            outB = np.array(outB)

            # print('outB[:,0]: ', np.array(outB)[:,0])
            
            print('var', sum(np.var(outB[:,0:2],0)))
            # RANDOM INPUT
            if sum(np.var(outB[:,0:2],0)) < 10:
                print('INPUT=RANDOM')
                '''
                outB = []
                for i in range(int(np.ceil(self.seq_len/3))):
                    r = np.random.rand(1,self.input_units).tolist()
                    if i == np.ceil(self.seq_len/3)-1 and self.seq_len%3 != 0:
                        outB.extend(r*(self.seq_len%3))
                    else:
                        outB.extend(r*3)
                outB = np.array(outB)
                '''

                # Using 3-dim spline function
                N = self.random_seq_len
                r = np.random.rand(N, 2)-0.5
                # if epoch != 0:
                #    r[0,:] = past_outB[-1]
                x = np.linspace(-0.5,0.5,num=N)
                xnew = np.linspace(-0.5,0.5,num=self.seq_len)
                f_cs = []
                # outB = []
                for i in range(2):
                    outB[:,i] = interp1d(x, r[:,i], kind='cubic')(xnew)
                    # outB.append(interp1d(range(N), r[:,i], kind='cubic')(xnew))
                outB = np.array(outB)


            if not is_drawing:
                break

            # outB = np.array(outB)/mag

            # outA_all.extend(list(np.array(outA)[:,0]))
            # outB_all.extend(list(np.array(outB)[:,0]))

            outA_all.extend(outA)
            outB_all.extend(outB)
            

            # Boredom
            # A: Error-Value is not change
            # B: Error-Value increases 
            bored_len = 20
            if is_changemode and self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                tmp_error.append(_error)
                if len(tmp_error) > bored_len:
                    tmp_error.popleft()

                if len(tmp_error) == bored_len:
                    a, b = np.polyfit(range(bored_len), tmp_error, 1)
                    error_slope.append(a)
                    print('slope: ', a)
                    
                    if abs(a) < 0.5:
                        print('[Change Mode A]: ', a)
                        tmp_error = deque([])
                        modeA = not modeA
                        mode_switch.append(epoch)
                        event.set_system_mode(modeA)
                    elif a > 5:
                        print('[Change Mode B]: ', a)
                        tmp_error = deque([])
                        modeA = not modeA
                        mode_switch.append(epoch)
                        event.set_system_mode(modeA)

                '''
                _diff = np.diff(tmp_error)


                _s = np.mean(np.diff(tmp_error))
                switch_prob.append(np.sign(_s) * 1/(abs(_s)+1))
                print('TE-diff: ', np.diff(tmp_error))
                print('Pr(switch) = ', switch_prob[-1])


                if switch_prob[-1]>0.9 or (np.sign(_s)==-1 and switch_prob[-1]<-0.1):
                    print('[Change Mode]: ', switch_prob[-1])
                    tmp_error = deque([])
                    modeA = not modeA
                    mode_switch.append(epoch)
                    event.set_system_mode(modeA)
                '''
                

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

            '''
            # A: SYSTEM, B: USER
            for i in range(self.seq_len):
                trajectoryA.extend([list(np.array(trajectoryA[-1] if len(trajectoryA) != 0 else init_pos2) + np.array(outA[i]))])
                trajectoryB.extend([list(np.array(trajectoryB[-1] if len(trajectoryB) != 0 else init_pos1) + np.array(outB[i]))])

            # Show characters on data points
            axL.annotate(str(epoch), trajectoryA[-1])
            axR.annotate(str(epoch), trajectoryB[-1])
            '''

            if _premodeA != modeA:
                mode_switch.append(epoch)
                _premodeA = modeA

            # print('[A] value={}'.format(outA))
            # print('[B] value={}'.format(outB))

            past_outA = outA
            past_outB = outB
            epoch += 1

        # print('outA: ', np.array(trajectoryA))
        # print('outB: ', np.array(trajectoryB))

        f.close()
        mode_switch = np.unique(mode_switch) * self.seq_len
        print('mode_switch: ', mode_switch)
        print('slope: ', error_slope)

        '''
        print('switch_prob: ', switch_prob)
        plt.figure(figsize=(10,6))
        plt.plot(switch_prob, marker='.')
        plt.show()
        '''


        '''
        # Drawing a start point
        axL.plot(trajectoryA[0][0],trajectoryA[0][1],'s'+self.colors[0], markersize=self.MARKER_SIZE*1.5)
        axR.plot(trajectoryB[0][0],trajectoryB[0][1],'s'+self.colors[1], markersize=self.MARKER_SIZE*1.5)

        for i in range(len(mode_switch)):
            _i = 0 if i == 0 else mode_switch[i-1]

            axL.plot(np.array(trajectoryA)[_i:mode_switch[i]+1,0], np.array(trajectoryA)[_i:mode_switch[i]+1,1], self.markers[i%2]+'-'+self.colors[0], lw=self.LINE_WIDTH, markersize=self.MARKER_SIZE, label='A')

            axR.plot(np.array(trajectoryB)[_i:mode_switch[i]+1,0], np.array(trajectoryB)[_i:mode_switch[i]+1,1], self.markers[i%2]+'-'+self.colors[1], lw=self.LINE_WIDTH, markersize=self.MARKER_SIZE, label='B')

        
        axL.set_title(self.behavior_mode)
        axR.set_title(event.get_mode())

        # outA: system, outB: user
        print('outA_all:', len(outA_all))
        outA_all = np.array(outA_all)
        outB_all = np.array(outB_all)

        fig5, ax_sig = plt.subplots(ncols=1, figsize=(6,6))
        ax_sig.plot(outA_all[:,0], c='red')
        ax_sig.plot(outB_all[:,0], c='blue')

        fig4, (ax_vec1) = plt.subplots(ncols=1, figsize=(6,6))
        # ax_vec1.quiver(outB_all[:,0], np.zeros(len(outB_all)), outA_all[:,0], np.zeros(len(outB_all)), angles='xy')
        ax_vec1.quiver(outB_all[self.seq_len*1:self.seq_len*3][:,0], outB_all[self.seq_len*1:self.seq_len*3][:,1], outA_all[self.seq_len*1:self.seq_len*3][:,0], outA_all[self.seq_len*1:self.seq_len*3][:,1], angles='xy', scale_units='xy')
        # ax_vec1.set_title('system-vector')

        ax_vec2.quiver(outB_all[self.seq_len*3:self.seq_len*5][:,0], outB_all[self.seq_len*3:self.seq_len*5][:,1], outA_all[self.seq_len*3:self.seq_len*5][:,0], outA_all[self.seq_len*3:self.seq_len*5][:,1], angles='xy', scale_units='xy')
        # ax_vec2.set_title('user-vector')

        ax_vec3.quiver(outB_all[self.seq_len*5:self.seq_len*7][:,0], outB_all[self.seq_len*5:self.seq_len*7][:,1], outA_all[self.seq_len*5:self.seq_len*7][:,0], outA_all[self.seq_len*5:self.seq_len*7][:,1], angles='xy', scale_units='xy')
        # ax_vec.quiver(outB_all[:,0], outB_all[:,1], outB_all[:,0], outB_all[:,1], angles='xy')


        _o = []
        for i in range(int(len(outA_all)/self.seq_len)):
            _o.append(outA_all[self.seq_len*i:self.seq_len*(i+1)][:,0])
        print('variance: ', np.mean(np.var(_o, 0)))


        fig3, (ax_mic, ax_delay) = plt.subplots(ncols=2, figsize=(12,6))
        ic = probability.InfoContent()
        delayed_tau, mic = ic.get_tau(outA_all[:,0][self.seq_len:], max_tau=20)

        print('tau: ', delayed_tau)
        ax_mic.plot(range(1,len(mic)+1), mic, c='black')
        ax_mic.set_title('Mutual Information Content(tau:{})'.format(delayed_tau))
        ax_mic.set_xticks(np.arange(0, 20+1, 1))
        ax_mic.grid()

        delayed_dim = 3
        delayed_out = []
        for i in reversed(range(delayed_dim)):
            delayed_out.append(np.roll(outA_all[:,0], -i*delayed_tau)[:len(outA_all)-delayed_tau])

        delayed_out = np.array(delayed_out).T

        ax_delay.set_title('delayed-out')
        ax_delay.plot(delayed_out[:,0], delayed_out[:,1], '.-')

        # Recurrence Plot
        fig2, (ax_sys, ax_sin, ax_rand) = plt.subplots(ncols=3, figsize=(18,6))

        _r = delayed_out[0:100]
        re_plot.plot(ax_sys, _r)
        ax_sys.set_title('System')

        # re_plot.plot(ax_sin, _r, eps=0.3)
        re_plot.plot(ax_sin, np.sin(2*np.pi*5*np.linspace(0,1,len(_r))))
        ax_sin.set_title('Sin')

        # re_plot.plot(ax_rand, _r, eps=0.4)
        re_plot.plot(ax_rand, np.random.rand(len(_r)))
        ax_rand.set_title('Random')
        '''

        print('Finish')
        # plt.show()


        # tf.app.run()


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

