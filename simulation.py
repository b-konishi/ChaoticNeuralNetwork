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
        self.seq_len = 30
        self.epoch_size = 100

        self.input_units = 2
        self.inner_units = 20
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

    def make_data(self, length, loop=0):
        # print('making data...')

        sound1 = my.Sound.load_sound(self.SOUND_PATH + 'ifudoudou.wav')
        sound2 = my.Sound.load_sound(self.SOUND_PATH + 'jinglebells.wav')
        sound1 = my.Sound.load_sound(self.SOUND_PATH + 'jinglebells.wav')
        sound2 = my.Sound.load_sound(self.SOUND_PATH + 'ifudoudou.wav')

        sound1 = sound1[30000:]
        sound2 = sound2[30000:]

        loop1 = loop % int(len(sound1)/length)
        loop2 = loop % int(len(sound2)/length)
        # print(loop1*length, len(sound1))
        # print(loop2*length, len(sound2))

        sound1 = sound1[loop1*length:(loop1+1)*length].reshape(length,1)
        sound2 = sound2[loop2*length:(loop2+1)*length].reshape(length,1)

        x1 = np.linspace(start=0, stop=length, num=length).reshape(length,1)
        y1 = np.sin(2*np.pi*x1) + 0.1*np.random.rand(length,1)
        # y1 = np.sin(x1/2)

        x2 = np.linspace(start=0, stop=length, num=length).reshape(length,1)
        y2 = np.sin(4*2*np.pi*x2) + 0.1*np.random.rand(length,1)
        # y2 = np.sin(x2)    

        # y2 = np.random.rand(length, 1)
        # y2 = 2*y1[1:] + 0.01*np.random.rand()
        # y1 = y1[:-1]

        '''
        y1 = np.linspace(0, 100, length)
        y2 = np.linspace(0, 100, length)
        '''

        data = np.resize(np.transpose([sound1,sound2]),(length, self.input_units))
        # data = np.resize(np.transpose([y1, y2]),(length, self.input_units))
        # data = np.resize(np.transpose([y1,sound1]),(length, self.input_units))

        data = data.astype(np.float32)

        return data

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


    def predict(self):
        compare = True
        pself.seq_len = 1000
        dt = 1/pself.seq_len
        psess = tf.InteractiveSession()

        saver = tf.train.import_meta_graph(self.MODEL_PATH + 'model.ckpt.meta')
        saver.restore(psess, tf.train.latest_checkpoint(self.MODEL_PATH))

        graph = tf.get_default_graph()
        '''
        for op in graph.get_operations():
            if op.name.find('Wi') > -1:
                print(op.name)
        '''

        Wi = graph.get_tensor_by_name("Wi/Wi:0")
        Wo = graph.get_tensor_by_name("Wo/Wo:0")

        print('predict')

        data = self.make_data(pself.seq_len, loop=1)*100000
        inputs = tf.placeholder(dtype = tf.float32, shape = [None, self.input_units], name='inputs')

        feed_dict={inputs:data}
        # inputs = self.make_data(pself.seq_len)
        output = self.inference(inputs, Wi, Wo)
        l = []
        out = psess.run(output, feed_dict=feed_dict)
        for i in range(self.output_units):
            o = out[:,i]
            o = (o-np.mean(o))/np.std(o)
            l.append(np.mean(np.log1p(abs(np.diff(o)/dt))-np.log(2.0)))

            # print('output[:,i]: {}'.format(out[:,i]))
            # l.append(get_lyapunov(seq=out[:,i]))
            # print('predict-lyapunov:{}'.format(psess.run([l[i]], feed_dict=feed_dict)))
            print('predict-lyapunov:{}'.format(l[i]))

        out = np.array(out)
        print('predictor-output:\n{}'.format(out))

        print('num: {}, unique: {}'.format(len(out[:,0]), len(set(out[:,0]))))
        print('mean: {}'.format(np.mean(out)))

        x = out[:,0]
        y = out[:,1]
        print('CC: ', np.corrcoef(x, y))

        if self.is_plot:

            '''
            plt.figure()
            plt.scatter(range(pself.seq_len), out[:,0], c='b', s=1)
            plt.figure()
            plt.scatter(range(pself.seq_len), out[:,1], c='r', s=1)
            '''

            plt.figure()
            plt.plot(x[0:pself.seq_len-1-self.tau], x[self.tau:pself.seq_len-1], c='r', lw=1)
            plt.figure()
            plt.plot(y[0:pself.seq_len-1-self.tau], y[self.tau:pself.seq_len-1], c='r', lw=1)



        '''
        out = out * 1000
        print('predictor-output:\n{}'.format(out))
        sampling = 44100
        my.Sound.save_sound(out[:,0], self.SOUND_PATH + 'chaos.wav', sampling)
        my.Sound.save_sound(out[:,1], self.SOUND_PATH + 'chaos2.wav', sampling)
        '''

        # random_sound = np.random.rand(pself.seq_len)*1000
        # save_sound(random_sound.astype(np.int), self.SOUND_PATH + 'random.wav', sampling)

        # In case of No Learning
        if compare:
            Wi = tf.Variable(self.weight(shape=[self.input_units, self.inner_units]), name='Wi')
            Wo = tf.Variable(self.weight(shape=[self.inner_units, self.output_units]), name='Wo')

            init_op = tf.global_variables_initializer()
            psess.run(init_op)

            data_nolearn = self.make_data(pself.seq_len, loop=1)
            output = self.inference(inputs, Wi, Wo)
            feed_dict={inputs:data_nolearn}
            l = []
            out = psess.run(output, feed_dict=feed_dict)
            for i in range(self.output_units):
                o = out[:,i]
                # l.append(get_lyapunov(seq=output[:,i]))
                o = (o-np.mean(o))/np.std(o)
                l.append(np.mean(np.log1p(abs(np.diff(o)/dt))-np.log(2.0)))
                # print('no-learning-lyapunov::{}'.format(psess.run(l[i], feed_dict=feed_dict)))
                print('no-learning-lyapunov::{}'.format(l[i]))

            print('num: {}, unique: {}'.format(len(out[:,0]), len(set(out[:,0]))))
            print('mean: {}'.format(np.mean(out)))

            x = out[:,0]
            y = out[:,1]
            print('CC: ', np.corrcoef(x, y))

            if self.is_plot:

                '''
                plt.figure()
                plt.scatter(range(pself.seq_len), out[:,0], c='b', s=1)
                plt.figure()
                plt.scatter(range(pself.seq_len), out[:,1], c='r', s=1)
                '''
                plt.figure()
                plt.plot(x[0:pself.seq_len-1-self.tau], x[self.tau:pself.seq_len-1], c='b', lw=1)
                plt.figure()
                plt.plot(y[0:pself.seq_len-1-self.tau], y[self.tau:pself.seq_len-1], c='b', lw=1)

            '''
            out_nolearn = psess.run(output, feed_dict=feed_dict)
            print('no-learning-output:\n{}'.format(out_nolearn))

            out_nolearn = out_nolearn * float(10000)
            print('no-learning-output:\n{}'.format(out_nolearn))
            my.Sound.save_sound(out_nolearn[:,0], self.SOUND_PATH + 'chaos_no.wav', sampling)
            my.Sound.save_sound(out_nolearn[:,1], self.SOUND_PATH + 'chaos_no2.wav', sampling)
            '''

        if self.is_plot:
            plt.show()


    def opt(self, x):
        kf = x[:,0]
        kr = x[:,1]
        alpha = x[:,2]

        # return np.sin(kf*kr+alpha)

        oself.seq_len = 5
        osess = tf.InteractiveSession()

        saver = tf.train.import_meta_graph(self.MODEL_PATH + 'model.ckpt.meta')

        saver.restore(osess, self.MODEL_PATH + 'model.ckpt')

        graph = tf.get_default_graph()
        Wi = graph.get_tensor_by_name("Wi/Wi:0")
        Wo = graph.get_tensor_by_name("Wo/Wo:0")

        print('optimize')
        inputs = self.make_data(oself.seq_len)
        output = self.inference(inputs, Wi, Wo)
        error, lyapunov = self.loss(output)

        return osess.run(error)

    def np_get_lyapunov(self, seq):
        # print('Measuring lyapunov...')
        dt = 1/len(seq)
        diff = np.abs(np.diff(seq))
        lyapunov = np.mean(np.log1p(diff/dt)-np.log(2.0))

        return lyapunov


    def learning1(self):

        if self.network_mode == self.TRAIN_MODE:
            sess = tf.InteractiveSession()

            with tf.name_scope('Mode'):
                Mode = tf.placeholder(dtype = tf.bool, name='Mode')
                mode = self.CREATIVE_MODE

            norm_in, inputs, outputs, params = self.inference(self.seq_len)
            Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

            error, pdf = self.loss(norm_in, outputs, self.seq_len, Mode)

            # Read PDFs
            dm, dmx, dmxx, dmxy = pdf['dm'], pdf['dmx'], pdf['dmxx'], pdf['dmxy']
            sampling = 10000
            bin_tau = 1/10
            # x = np.reshape(np.linspace(-0.5,1.5,10000), [10000,1])
            ix = np.reshape(np.linspace(0,1,sampling), [sampling,1])
            iy = np.reshape(np.linspace(0,1,sampling), [sampling,1])
            ixy = np.concatenate((ix+bin_tau,ix,iy), axis=1)
            dmxo = dmx.prob(ixy[:,1])
            dmxyo = dmxy.prob(ixy[:,1:])

            tf.summary.scalar('error', error)
            train_step, grad = self.train(error, [Wi, bi, Wo])

            '''
            with tf.name_scope('lyapunov'):
                for i in range(self.output_units):
                    tf.summary.scalar('lyapunov'+str(i), lyapunov[i])
            '''

            # Tensorboard logfile
            if tf.gfile.Exists(self.LOG_PATH):
                tf.gfile.DeleteRecursively(self.LOG_PATH)
            writer = tf.summary.FileWriter(self.LOG_PATH, sess.graph)

            run_options = tf.RunOptions(output_partition_graphs=True)
            run_metadata = tf.RunMetadata()

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # For Debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            merged = tf.summary.merge_all()

            
            '''
            # confirm
            data = self.make_data(self.seq_len, loop=0)
            feed_dict = {inputs:data}
            d, out = sess.run([dmxo, outputs], feed_dict=feed_dict)
            # print(out)
            # print((out[:,0]-min(out[:,0]))/(max(out[:,0])-min(out[:,0])))
            plt.figure()
            # plt.scatter(d[:,0], d[:,1], s=1)
            plt.plot(x, d, lw=1)
            if self.is_plot:
                plt.show()
            '''
            
            ic = probability.InfoContent() 
            
            lcluster = 500
            epoch_cluster = math.ceil(lcluster/self.seq_len)
            print('epoch_cluster: ', epoch_cluster)
            out_cluster, lyapunov = [], []
            dt = 1/lcluster

            l_list, te_list, time_list = [], [], []
            out_sound1, out_sound2 = [], []
            in_color, out_color = 'r', 'b'
            allout = []

            for epoch in range(self.epoch_size):
                print('\n[epoch:{}-times]'.format(epoch))
                data = self.make_data(self.seq_len, loop=epoch)
                feed_dict = {inputs:data, Mode:mode}

                start = time.time()
                wi, wo, _in, out, error_val = sess.run([Wi, Wo, norm_in, outputs, error], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

                # run for pdf
                dx, dxy = sess.run([dmxo, dmxyo], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

                summary, t, gradients = sess.run([merged, train_step, grad], feed_dict)
                end = time.time()

                allout.extend(out[:,0])

                indata, outdata = [_in[:,0], out[:,0]]
                out_sound1.extend(out[:,0])
                out_sound2.extend(out[:,1])
                print('wi: ', wi[:,0:10])
                print('wo: ', wo[0:10,:])
                print('in: ', indata[0:10])
                print('out: ', outdata[0:10])
                # print('grad: ', gradients[0][0:10])
                for (g, v) in gradients:
                    print('grad: ', g[0][0:5])

                # Lyapunov
                out_cluster.extend(outdata)
                print('len(out_cluster)', len(out_cluster))
                print('len(outdata)', len(outdata))
                if epoch != 0 and epoch % epoch_cluster == 0:
                    lyapunov.append(self.np_get_lyapunov(out_cluster))
                    out_cluster = []
                    

                # l_list.append(l[0])
                te_list.append(ic.np_get_TE(outdata, indata))

                elapsed_time = end-start
                time_list.append(elapsed_time)

                if epoch%1 == 0:
                    total_time = np.mean(time_list)*self.epoch_size
                    cumulative_time = sum(time_list)
                    remineded_time = total_time - cumulative_time

                    print("elapsed_time: {}sec/epoch".format(int(elapsed_time)))
                    print("Estimated-reminded-time: {}sec({}sec/{}sec)".format(int(remineded_time), int(cumulative_time), int(total_time)))
                    print("error:{}".format(error_val))
                    print("Transfer-Entropy: ", ic.np_get_TE(outdata, indata))
                    # print(d)
                    # print(out)

                if self.is_plot and epoch%(self.epoch_size-1) == 0:

                    in1, in2, out1, out2 = _in[:,0], _in[:,1], out[:,0], out[:,1]

                    if True:
                        plt.figure()
                        plt.title('time-input-graph(epoch:{})'.format(epoch))
                        plt.plot(range(len(in1)), in1, c='r', lw=1, label='input1')
                        plt.plot(range(len(in2)), in2, c='b', lw=1, label='input2')
                        plt.legend(loc=2)

                        plt.figure()
                        plt.title('time-output-graph(epoch:{})'.format(epoch))
                        plt.plot(range(len(out1)), out1, c='r', lw=1, label='output1')
                        plt.plot(range(len(out2)), out2, c='b', lw=1, label='output2')
                        plt.legend(loc=2)

                    if True:
                        plt.figure()
                        plt.title('delayed-input-2Dgraph(epoch:{})'.format(epoch))
                        plt.plot(in1[:-self.tau], in1[self.tau:], c='r', lw=1, label='input')
                        plt.plot(in2[:-self.tau], in2[self.tau:], c='b', lw=1, label='input')
                        plt.legend(loc=2)

                        plt.figure()
                        plt.title('delayed-out-2Dgraph(epoch:{})'.format(epoch))
                        plt.plot(out1[:-self.tau], out1[self.tau:], c='r', lw=1, label='output')
                        plt.plot(out2[:-self.tau], out2[self.tau:], c='b', lw=1, label='output')
                        plt.legend(loc=2)

                    if True:
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_title('delayed-out-3Dgraph(epoch:{})'.format(epoch))
                        ax.scatter3D(in1[:-2*self.tau],in1[self.tau:-self.tau],in1[2*self.tau:], c=in_color, label='input')
                        ax.scatter3D(out1[:-2*self.tau],out1[self.tau:-self.tau],out1[2*self.tau:], c=out_color, label='output')
                        plt.legend(loc=2)

                writer.add_summary(summary, epoch)


            re_plot = my.RecurrencePlot()

            # Recurrence Plot
            fig2, (ax_sys, ax_sin, ax_rand) = plt.subplots(ncols=3, figsize=(18,6))
            _r = allout[-100:]
            re_plot.plot(ax_sys, _r)
            ax_sys.set_title('System')
            re_plot.plot(ax_sin, np.sin(2*np.pi*5*np.linspace(0,1,len(_r))))
            ax_sin.set_title('Sin')
            re_plot.plot(ax_rand, np.random.rand(len(_r)))
            ax_rand.set_title('Random')



            sampling_freq = 14700
            if False:
                my.Sound.save_sound((np.array(out_sound1))*40000, self.SOUND_PATH + 'chaos1.wav', sampling_freq)
                my.Sound.save_sound((np.array(out_sound2))*40000, self.SOUND_PATH + 'chaos2.wav', sampling_freq)

            if False:
                plt.figure()
                plt.title('Transfer Entropy')
                plt.plot(range(len(te_list)), te_list)

            if False:
                plt.figure()
                plt.title('Disposal Time')
                plt.plot(range(len(time_list)-1), time_list[1:])

            if False:
                plt.figure()
                plt.title('Lyapunov Exponent')
                plt.plot(range(len(lyapunov)), lyapunov)
                print('Mean-Lyapunov-Value: ', np.mean(lyapunov))

                lcluster = 500
                lyapunov_sin = (self.np_get_lyapunov(np.sin(np.linspace(0, 1, lcluster))))
                lyapunov_random = (self.np_get_lyapunov(np.random.rand(lcluster)))
                print('Lyapunov-sin: ', lyapunov_sin)
                print('Lyapunov-random: ', lyapunov_random)

            # 混合ベイズ分布ができているか確認
            if False:
                plt.figure()
                plt.title('pdf: p(x)')
                # plt.scatter(dx[:,0], dx[:,1], s=1)
                plt.plot(ixy[:,1], dx, lw=2)

                print(np.shape(dxy))
                print(np.shape(ixy))
                '''
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                ax.set_title('pdf: p(x,y)')
                ax.scatter3D(ixy[:,1], ixy[:,2], dxy)
                '''

            if self.is_save:
                '''
                # 特定の変数だけ保存するときに使用
                train_vars = tf.trainable_variables()
                '''
                saver = tf.train.Saver()
                saver.save(sess, self.MODEL_PATH + 'model.ckpt')

                '''
                saver.restore(sess, tf.train.latest_checkpoint(self.MODEL_PATH))
                print(sess.run(Wi))
                '''
            # print("output:{}".format(out))
            out = np.array(out)

            if self.is_plot:
                plt.show()

            # sess.close()

        elif self.network_mode == self.PREDICT_MODE:
            self.predict()

        elif self.network_mode == self.OPTIMIZE_MODE:
            bounds = [{'name': 'kf',    'type': 'continuous',  'domain': (0.0, 100.0)},
                      {'name': 'kr',    'type': 'continuous',  'domain': (0.0, 100.0)},
                      {'name': 'alpha', 'type': 'continuous',  'domain': (0.0, 100.0)}]

            # Do Presearch
            opt_mnist = GPyOpt.methods.BayesianOptimization(f=self.opt, domain=bounds)

            # Search Optimized Parameter
            opt_mnist.run_optimization(max_iter=10)
            print("optimized parameters: {0}".format(opt_mnist.x_opt))
            print("optimized loss: {0}".format(opt_mnist.fx_opt))

    # if __name__ == "__main__":
    def robot_robot_interaction(self):
        sessA = tf.InteractiveSession()
        sessB = tf.InteractiveSession()

        with tf.name_scope('Mode'):
            Mode = tf.placeholder(dtype = tf.bool, name='Mode')
            tf.summary.scalar('Mode', tf.cast(Mode, tf.int32))

        norm_in, inputs, outputs, params = self.inference(self.seq_len)
        Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

        error, pdf = self.loss(norm_in, outputs, self.seq_len, Mode)
        tf.summary.scalar('error', error)
        train_step, grad = self.train(error, [Wi, bi, Wo])

        merged = tf.summary.merge_all()

        # Tensorboard logfile
        self.LOG_PATHA = '../Alogdir'
        self.LOG_PATHB = '../Blogdir'

        if tf.gfile.Exists(self.LOG_PATHA):
            tf.gfile.DeleteRecursively(self.LOG_PATHA)
        writerA = tf.summary.FileWriter(self.LOG_PATHA, sessA.graph)

        if tf.gfile.Exists(self.LOG_PATHB):
            tf.gfile.DeleteRecursively(self.LOG_PATHB)
        writerB = tf.summary.FileWriter(self.LOG_PATHB, sessB.graph)


        sessA.run(tf.global_variables_initializer())
        sessB.run(tf.global_variables_initializer())

        # fig, ax = plt.subplots(1, 1)
        fig = plt.figure(figsize=(10,6))

        # True: Following, False: Creative
        modeA = self.IMITATION_MODE
        modeB = self.CREATIVE_MODE

        online_update = False
        online_update = True

        trajectoryA = []
        trajectoryB = []

        outB = np.random.rand(self.seq_len, 2)
        for epoch in range(self.epoch_size):
            print('epoch: ', epoch)

            if epoch%10 == 0:
                modeA, modeB = modeB, modeA
                pass

            feed_dictA = {inputs:outB, Mode:modeA}

            outA, gradientsA = sessA.run([outputs, grad], feed_dict=feed_dictA)

            
            feed_dictB = {inputs:outA, Mode:modeB}
            outB, gradientsB = sessB.run([outputs, grad], feed_dict=feed_dictB)
            if epoch % 10:
                for (g, v) in gradientsA:
                    print('gradA: ', g[0][0:5])
                for (g, v) in gradientsB:
                    print('gradB: ', g[0][0:5])

            summaryA, _ = sessA.run([merged, train_step], feed_dictA)
            summaryB, _ = sessB.run([merged, train_step], feed_dictB)

            writerA.add_summary(summaryA, epoch)
            writerB.add_summary(summaryB, epoch)

            print('[A] mode={}, value={}'.format(modeA, np.array(outA[0])-0.5))
            print('[B] mode={}, value={}'.format(modeB, np.array(outB[0])-0.5))

            if not online_update:
                trajectoryA.extend((trajectoryA[-1] if len(trajectoryA) != 0 else np.zeros(len(outA[0]))) + np.cumsum(np.array(outA)-0.5, axis=0))
                trajectoryB.extend((trajectoryB[-1] if len(trajectoryB) != 0 else np.zeros(len(outB[0]))) + np.cumsum(np.array(outB)-0.5, axis=0))

            if online_update:
                for i in range(self.seq_len):
                    trajectoryA.extend([np.array(trajectoryA[-1] if len(trajectoryA) != 0 else [0,0]) + np.array(outA[i])-0.5])
                    trajectoryB.extend([np.array(trajectoryB[-1] if len(trajectoryB) != 0 else [0,0]) + np.array(outB[i])-0.5])

                    plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+self.colors[0], lw=0.1, label='A')
                    plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+self.colors[1], lw=0.1, label='B')
                    plt.pause(0.01)

            
        if not online_update:
            plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+self.colors[0], lw=0.1, label='A')
            plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+self.colors[1], lw=0.1, label='B')


        print('Finish')
        plt.show()


    # if __name__ == "__main__":
    def human_agent_interaction(self):
        sessA = tf.InteractiveSession()

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
        self.LOG_PATHA = '../Alogdir'

        if tf.gfile.Exists(self.LOG_PATHA):
            tf.gfile.DeleteRecursively(self.LOG_PATHA)
        writerA = tf.summary.FileWriter(self.LOG_PATHA, sessA.graph)

        sessA.run(tf.global_variables_initializer())

        # fig, ax = plt.subplots(1, 1)
        # fig = plt.figure(figsize=(10,6))
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,6))

        event = draw.Event(draw.Event.USER_MODE)
        re_plot = my.RecurrencePlot()

        # True: Following, False: Creative
        modeA = self.IMITATION_MODE
        modeA = self.CREATIVE_MODE

        trajectoryA = []
        trajectoryB = []

        outB = np.random.rand(self.seq_len, 2)
        is_drawing = True
        is_changemode = False
        _premodeA = modeA
        mode_switch = [self.epoch_size]
        outA_all, outB_all = [], []
        for epoch in range(self.epoch_size):
            print('epoch:{}, mode:{}'.format(epoch, modeA))

            if self.behavior_mode == self.CHAOTIC_BEHAVIOR:
                feed_dictA = {inputs:outB, Mode:modeA}
                outA, gradientsA = sessA.run([outputs, grad], feed_dict=feed_dictA)

                if epoch % 1 == 0:
                    for (g, v) in gradientsA:
                        print('gradA: ', g[0][0:5])

                summaryA, _ = sessA.run([merged, train_step], feed_dictA)
                writerA.add_summary(summaryA, epoch)

            if self.behavior_mode == self.RANDOM_BEHAVIOR:
                outA = np.random.rand(self.seq_len, self.output_units)-0.5

            outB = []
            mag = 100
            for i in range(self.seq_len):
                event.set_movement(np.array(outA[i]), mag)

                diff, is_drawing = event.get_pos()
                outB.append(diff)

                time.sleep(0.1)

            if not is_drawing:
                break


            outB = np.array(outB)/mag

            outA_all.extend(list(np.array(outA)[:,0]))
            outB_all.extend(list(np.array(outB)[:,0]))


            # Mesuring the amount of activity
            d1 = np.mean(abs(np.diff(abs(np.diff(outB[:,0])))))
            d2 = np.mean(abs(np.diff(abs(np.diff(outB[:,1])))))
            print(np.mean([d1,d2]))

            if is_changemode and np.mean([d1,d2]) < 0.07:
                print('[Change Mode]', np.mean([d1,d2]))
                modeA = not modeA
                mode_switch.append(epoch)
                event.set_system_mode(modeA)



            # A: SYSTEM, B: USER
            for i in range(self.seq_len):
                trajectoryA.extend([list(np.array(trajectoryA[-1] if len(trajectoryA) != 0 else draw.Event.INIT_POS2) + np.array(outA[i]))])
                trajectoryB.extend([list(np.array(trajectoryB[-1] if len(trajectoryB) != 0 else draw.Event.INIT_POS1) + np.array(outB[i]))])

            # Show characters on data points
            axL.annotate(str(epoch), trajectoryA[-1])
            axR.annotate(str(epoch), trajectoryB[-1])

            if _premodeA != modeA:
                mode_switch.append(epoch)
                _premodeA = modeA

            # print('[A] value={}'.format(outA))
            # print('[B] value={}'.format(outB))


        mode_switch = np.unique(mode_switch) * self.seq_len
        print('mode_switch: ', mode_switch)

        print('outA_all:', len(outA_all))

        # Drawing a start point
        axL.plot(trajectoryA[0][0],trajectoryA[0][1],'s'+self.colors[0], markersize=self.MARKER_SIZE*1.5)
        axR.plot(trajectoryB[0][0],trajectoryB[0][1],'s'+self.colors[1], markersize=self.MARKER_SIZE*1.5)
        for i in range(len(mode_switch)):
            _i = 0 if i == 0 else mode_switch[i-1]

            axL.plot(np.array(trajectoryA)[_i:mode_switch[i]+1,0], np.array(trajectoryA)[_i:mode_switch[i]+1,1], self.markers[i%2]+'-'+self.colors[0], lw=self.LINE_WIDTH, markersize=self.MARKER_SIZE, label='A')

            axR.plot(np.array(trajectoryB)[_i:mode_switch[i]+1,0], np.array(trajectoryB)[_i:mode_switch[i]+1,1], self.markers[i%2]+'-'+self.colors[1], lw=self.LINE_WIDTH, markersize=self.MARKER_SIZE, label='B')

        
        axL.set_title(self.behavior_mode)
        axR.set_title(event.get_mode())


        fig3, (ax_mic, ax_delay) = plt.subplots(ncols=2, figsize=(12,6))
        ic = probability.InfoContent()
        delayed_tau, mic = ic.get_tau(outA_all[self.seq_len:], max_tau=20)

        print('tau: ', delayed_tau)
        ax_mic.plot(range(1,len(mic)+1), mic, c='black')
        ax_mic.set_title('Mutual Information Content(tau:{})'.format(delayed_tau))
        ax_mic.set_xticks(np.arange(0, 20+1, 1))
        ax_mic.grid()

        delayed_dim = 3
        delayed_out = []
        for i in reversed(range(delayed_dim)):
            delayed_out.append(np.roll(outA_all, -i*delayed_tau)[:-delayed_tau])

        delayed_out = np.array(delayed_out).T

        ax_delay.set_title('delayed-out')
        ax_delay.plot(delayed_out[:,0], delayed_out[:,1], '.-')

        # Recurrence Plot
        fig2, (ax_sys, ax_sin, ax_rand) = plt.subplots(ncols=3, figsize=(18,6))

        _r = delayed_out[100:300]
        re_plot.plot(ax_sys, _r)
        ax_sys.set_title('System')

        # re_plot.plot(ax_sin, _r, eps=0.3)
        re_plot.plot(ax_sin, np.sin(2*np.pi*5*np.linspace(0,1,len(_r))))
        ax_sin.set_title('Sin')

        # re_plot.plot(ax_rand, _r, eps=0.4)
        re_plot.plot(ax_rand, np.random.rand(len(_r)))
        ax_rand.set_title('Random')

        print('Finish')
        plt.show()


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

