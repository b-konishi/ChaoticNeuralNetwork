# -*- coding: utf-8 -*-

# my library
import my_library as my
import chaotic_nn_cell
import probability
import draw

# Standard
import math
import time
import numpy as np
import threading
import warnings
import datetime

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
    LOG_PATH = '../logdir'
    SOUND_PATH = '../music/'

    act_logfile = '../log_act.txt'

    RANDOM_BEHAVIOR = 'RANDOM'
    CHAOTIC_BEHAVIOR = 'CHAOS'

    IMITATION_MODE = True
    CREATIVE_MODE = False

    OPTIMIZE_MODE = 'OPT'
    PREDICT_MODE = 'PREDICT'
    TRAIN_MODE = 'TRAIN'


    def __init__(self, network_mode=TRAIN_MODE, behavior_mode=CHAOTIC_BEHAVIOR):
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
        self.seq_len = 25
        self.epoch_size = 100

        self.input_units = 2
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
                tf.summary.histogram('Wi', Wi)
                params['Wi'] = Wi

            with tf.name_scope('bi'):
                bi = tf.Variable(self.weight(shape=[1,self.inner_units]), name='bi')
                tf.summary.histogram('bi', bi)
                params['bi'] = bi

            # input: [None, input_units]
            in_norm = self.tf_normalize(inputs)
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
                tf.summary.histogram('Wo', Wo)
                params['Wo'] = Wo

            with tf.name_scope('bo'):
                bo = tf.Variable(self.weight(shape=[1, self.output_units]), name='bo')
                tf.summary.histogram('bo', bo)
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


    def loss(self, inputs, outputs, length, mode):
        # if mode==True: TE(y->x)

        # Laypunov-Exponent
        with tf.name_scope('Lyapunov'):
            lyapunov = []
            for i in range(self.output_units):
                lyapunov.append(self.tf_get_lyapunov(outputs[:,i], length))
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])

        with tf.name_scope('TE-Loss'):
            ic = probability.InfoContent()

            # The empty constant Tensor to get unit-num.
            output_units_ = tf.constant(0, shape=[self.output_units])
            input_units_ = tf.constant(0, shape=[self.input_units])

            x, x_units_, y, y_units_ = tf.cond(mode,
                    lambda:(outputs, output_units_, inputs, input_units_),
                    lambda:(inputs, input_units_, outputs, output_units_))

            x_units = int(x_units_.get_shape()[0])
            y_units = int(y_units_.get_shape()[0])
            print('x_units: {}, y_units: {}'.format(x_units, y_units))

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

            return -tf.reduce_mean(entropy), _pdf

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
            



    def human_agent_interaction(self):
        sess = tf.InteractiveSession()

        with tf.name_scope('Mode'):
            Mode = tf.placeholder(dtype = tf.bool, name='Mode')
            tf.summary.scalar('Mode', tf.cast(Mode, tf.int32))

        norm_in, inputs, outputs, params = self.inference(self.seq_len)
        Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

        error, pdf = self.loss(norm_in, outputs, self.seq_len, Mode)
        tf.summary.scalar('error', error)
        train_step, grad = self.train(error, [Wi, bi, Wo, bo])

        merged = tf.summary.merge_all()

        # Tensorboard logfile
        self.LOG_PATHA = '../logdir'

        if tf.gfile.Exists(self.LOG_PATHA):
            tf.gfile.DeleteRecursively(self.LOG_PATHA)
        writerA = tf.summary.FileWriter(self.LOG_PATHA, sess.graph)

        sess.run(tf.global_variables_initializer())

        # fig, ax = plt.subplots(1, 1)
        # fig = plt.figure(figsize=(10,6))
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,6))

        event = draw.Event(draw.Event.USER_MODE)
        re_plot = my.RecurrencePlot()

        # True: Following, False: Creative
        modeA = self.IMITATION_MODE
        modeA = self.CREATIVE_MODE
        event.set_system_mode(modeA)

        trajectoryA = []
        trajectoryB = []
        init_pos1, init_pos2 = event.get_init_pos()

        outB = np.random.rand(self.seq_len, 2)
        is_drawing = True
        is_changemode = True
        _premodeA = modeA
        mode_switch = []
        outA_all, outB_all = [], []
        epoch = 0
        proctime = 1
        
        f = open(self.act_logfile, mode='w')

        print("SYSTEM_WAIT")
        while not event.get_startup_signal():
            time.sleep(0.1)
            pass
        time.sleep(0.5)
        # for epoch in range(self.epoch_size):
        while not event.get_systemstop_signal():
            print('epoch:{}, mode:{}'.format(epoch, modeA))

            network_proctime_st = datetime.datetime.now()

            # event.set_network_interval(proctime, 3)
            networkbg_thread = threading.Thread(target=self.network_bg, args=[event, proctime, 3])
            networkbg_thread.daemon = True
            networkbg_thread.start()
            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                feed_dictA = {inputs:outB, Mode:modeA}
                outA, gradientsA = sess.run([outputs, grad], feed_dict=feed_dictA)

                if epoch % 1 == 0:
                    for (g, v) in gradientsA:
                        print('gradA: ', g[0][0:5])

                summaryA, _ = sess.run([merged, train_step], feed_dictA)
                writerA.add_summary(summaryA, epoch)

            if self.behavior_mode == self.RANDOM_BEHAVIOR:
                outA = np.random.rand(self.seq_len, self.output_units)-0.5

            network_proctime_en = datetime.datetime.now()
            proctime = (network_proctime_en-network_proctime_st).total_seconds()
            print('proctime:{}s '.format(proctime))

            outB = []
            mag = 20
            for i in range(self.seq_len):
                event.set_movement(np.array(outA[i]), mag)

                diff, is_drawing = event.get_pos()
                outB.append(diff)
                print('DIFF: ', diff)

                # ADJUST
                time.sleep(0.05)


            print('outB[:,0]: ', np.array(outB)[:,0])
            
            print('var', sum(np.var(outB,0)))
            # ADJUST
            if sum(np.var(outB,0)) < 20:
                print('INPUT=RANDOM')
                outB = []
                for i in range(int(np.ceil(self.seq_len/3))):
                    r = np.random.rand(1,2).tolist()
                    if i == np.ceil(self.seq_len/3)-1 and self.seq_len%3 != 0:
                        outB.extend(r*(self.seq_len%3))
                    else:
                        outB.extend(r*3)
                outB = np.array(outB)


                '''
                r = np.random.rand(int((self.seq_len/3)), 2)
                outB = np.zeros([self.seq_len,2])
                outB[0:self.seq_len:3] = r
                outB[1:self.seq_len:3] = r
                outB[2:self.seq_len:3] = r
                # outB = np.random.rand(self.seq_len, 2)
                '''


            if not is_drawing:
                break

            outB = np.array(outB)/mag

            # outA_all.extend(list(np.array(outA)[:,0]))
            # outB_all.extend(list(np.array(outB)[:,0]))

            outA_all.extend(outA)
            outB_all.extend(outB)


            # Mesuring the amount of activity
            d1 = np.mean(abs(np.diff(outB[:,0])))
            d2 = np.mean(abs(np.diff(outB[:,1])))
            self.activity = np.mean([d1,d2])
            print(self.activity)
            f.write(str(self.activity)+'\n')

            if is_changemode and self.activity < 0.2:
                print('[Change Mode]', self.activity)
                modeA = not modeA
                mode_switch.append(epoch)
                event.set_system_mode(modeA)

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

            epoch += 1

        # print('outA: ', np.array(trajectoryA))
        # print('outB: ', np.array(trajectoryB))

        f.close()
        mode_switch = np.unique(mode_switch) * self.seq_len
        print('mode_switch: ', mode_switch)


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
    # simulator.learning1()
    # simulator.robot_robot_interaction()
    simulator.human_agent_interaction()

    '''
    calc_thread = threading.Thread(target=test)
    # calc_thread.daemon = True
    calc_thread.start()
    '''

