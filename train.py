# -*- coding: utf-8 -*-

'''
[OK?(重みがブロック行列になっていればよいのでは？)]重みにあとから単位ブロック行列を掛けるのではなく、初めからそうしたい（勾配行列を変こえる？）

リアルタイム入出力
'''

# my library
import my_library as my
import chaotic_nn_cell
import info_content

# Standard
import math
import time
import numpy as np

# Tensorflow
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow_probability as tfp
tfd = tfp.distributions

# Drawing Graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyxhook

'''
# Baysian Optimization
import GPy
import GPyOpt
'''

model_path = '../model/'
log_path = '../logdir'

MODE = 'opt'
MODE = 'predict'
MODE = 'train'

# Save the model
is_save = True
is_save = False

# Drawing graphs flag
is_plot = False
is_plot = True

activation = tf.nn.tanh

seq_len = 100

epoch_size = 100
input_units = 2
inner_units = 100
output_units = 2

# 中間層層数
inner_layers = 1

Kf = 0.1
Kr = 0.9
Alpha = 10.0

# Kf, Kr, Alpha = 0, 0, 0

# time delayed value
tau = int(seq_len/100)
tau = 10

def make_data(length, loop=0):
    # print('making data...')

    sound1 = my.Sound.load_sound('../music/ifudoudou.wav')
    sound2 = my.Sound.load_sound('../music/jinglebells.wav')
    sound1 = my.Sound.load_sound('../music/jinglebells.wav')
    sound2 = my.Sound.load_sound('../music/ifudoudou.wav')

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

    data = np.resize(np.transpose([sound1,sound2]),(length, input_units))
    # data = np.resize(np.transpose([y1, y2]),(length, input_units))
    # data = np.resize(np.transpose([y1,sound1]),(length, input_units))

    data = data.astype(np.float32)
    '''
    # Normalization
    for i in range(input_units):
        d = data[:,i]
        if np.max(d) != np.min(d):
            norm = (d-np.min(d))/(np.max(d)-np.min(d))
        else:
            norm = np.clip(np.sign(d), 0., 1.)
        data[:,i] = norm
    '''

    return data

def weight(shape = []):
    initial = tf.truncated_normal(shape, stddev = 0.1, dtype=tf.float32)
    # return tf.Variable(initial)
    return initial

def set_innerlayers(inputs, layers_size):

    inner_output = inputs
    for i in range(layers_size):
        with tf.name_scope('Layer' + str(i+2)):
            cell = chaotic_nn_cell.ChaoticNNCell(num_units=inner_units, Kf=Kf, Kr=Kr, alpha=Alpha, activation=activation)
            outputs, state = tf.nn.static_rnn(cell=cell, inputs=[inner_output], dtype=tf.float32)
            inner_output = outputs[-1]

    return inner_output

def normalize(v):
    with tf.name_scope('Normalize'):
        norm = (v-tf.reduce_min(v,0))/(tf.reduce_max(v,0)-tf.reduce_min(v,0))
        sign = tf.nn.relu(tf.sign(v))
        var = tf.where(tf.is_nan(norm), sign, norm)

    return var


def inference(length):
    params = {}

    with tf.name_scope('Input_Layer'):
        with tf.name_scope('data'):
            inputs = tf.placeholder(dtype = tf.float32, shape = [length, input_units], name='inputs')
        with tf.name_scope('Wi'):
            Wi = tf.Variable(weight(shape=[input_units, inner_units]), name='Wi')
            tf.summary.histogram('Wi', Wi)
            params['Wi'] = Wi

        with tf.name_scope('bi'):
            bi = tf.Variable(weight(shape=[1,inner_units]), name='bi')
            tf.summary.histogram('bi', bi)
            params['bi'] = bi

        # input: [None, input_units]
        in_norm = normalize(inputs)
        fi = tf.matmul(in_norm, Wi) + bi
        sigm = tf.nn.sigmoid(fi)

    inner_output = set_innerlayers(sigm, inner_layers)

    with tf.name_scope('Output_Layer'):
        if inner_units % output_units != 0:
            print("Can't make the clusters")
            exit()
        cluster_size = int(inner_units / output_units)
        print('cluster: ', cluster_size)
        one = [1]*cluster_size
        one.extend([0]*(inner_units-cluster_size))
        ones = []
        for i in range(output_units):
            ones.append(np.roll(one, cluster_size*i))
        ones = np.reshape(ones, [output_units, inner_units]).T

        Io = tf.cast(ones, tf.float32)

        with tf.name_scope('Wo'):
            Wo = tf.Variable(weight(shape=[inner_units, output_units]), name='Wo')
            tf.summary.histogram('Wo', Wo)
            params['Wo'] = Wo

        with tf.name_scope('bo'):
            bo = tf.Variable(weight(shape=[1, output_units]), name='bo')
            tf.summary.histogram('bo', bo)
            params['bo'] = bo

        fo = tf.matmul(inner_output, tf.multiply(Wo, Io))
        outputs = normalize(fo)

    return in_norm, inputs, outputs, params

def tf_get_lyapunov(seq, length):
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


def loss(inputs, outputs, length, mode):
    # if mode==True: TE(y->x)

    # Laypunov-Exponent
    with tf.name_scope('Lyapunov'):
        lyapunov = []
        for i in range(output_units):
            lyapunov.append(tf_get_lyapunov(outputs[:,i], length))
            tf.summary.scalar('lyapunov'+str(i), lyapunov[i])

    with tf.name_scope('TE-Loss'):
        ic = info_content.info_content()

        output_units_ = tf.constant(0, shape=[output_units])
        input_units_ = tf.constant(0, shape=[input_units])

        x, x_units_, y, y_units_ = tf.cond(mode,
                lambda:(outputs, output_units_, inputs, input_units_),
                lambda:(inputs, input_units_, outputs, output_units_))

        x_units = int(x_units_.get_shape()[0])
        y_units = int(y_units_.get_shape()[0])
        print('x_units: {}, y_units: {}'.format(x_units, y_units))

        entropy = []
        for i in range(x_units):
            entropy_ = 0
            for j in range(y_units):
                x_, y_ = x[:,j], y[:,i]
                # TE(y->x)
                en, pdf = ic.get_TE_for_tf4(x_, y_, seq_len)
                entropy_ += en
            entropy.append(entropy_)
            tf.summary.scalar('entropy{}'.format(i), entropy[i])

        return -tf.reduce_mean(entropy), pdf

    '''
    with tf.name_scope('loss_lyapunov'):
        # リアプノフ指数が増加するように誤差関数を設定
        lyapunov = []
        loss = []
        for i in range(output_units):
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


def train(error, update_params):
    # return tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error)
    with tf.name_scope('training'):
        # opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        opt = tf.train.AdamOptimizer(0.001)
        grads_ = opt.compute_gradients(error, var_list=update_params)
        # grads_ = [(grad1,var1),(grad2,var2),...]

        grads = []
        for g in grads_:
            g_ = tf.where(tf.is_nan(g[0]), tf.zeros_like(g[0]), g[0])
            grads.append((g_, g[1]))

        # training = opt.minimize(grads, var_list=update_params)
        training = opt.apply_gradients(grads)

    return training, grads


def predict():
    compare = True
    pseq_len = 1000
    dt = 1/pseq_len
    psess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')
    saver.restore(psess, tf.train.latest_checkpoint(model_path))

    graph = tf.get_default_graph()
    '''
    for op in graph.get_operations():
        if op.name.find('Wi') > -1:
            print(op.name)
    '''

    Wi = graph.get_tensor_by_name("Wi/Wi:0")
    Wo = graph.get_tensor_by_name("Wo/Wo:0")

    print('predict')

    data = make_data(pseq_len, loop=1)*100000
    inputs = tf.placeholder(dtype = tf.float32, shape = [None, input_units], name='inputs')

    feed_dict={inputs:data}
    # inputs = make_data(pseq_len)
    output = inference(inputs, Wi, Wo)
    l = []
    out = psess.run(output, feed_dict=feed_dict)
    for i in range(output_units):
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

    if is_plot:

        '''
        plt.figure()
        plt.scatter(range(pseq_len), out[:,0], c='b', s=1)
        plt.figure()
        plt.scatter(range(pseq_len), out[:,1], c='r', s=1)
        '''

        plt.figure()
        plt.plot(x[0:pseq_len-1-tau], x[tau:pseq_len-1], c='r', lw=1)
        plt.figure()
        plt.plot(y[0:pseq_len-1-tau], y[tau:pseq_len-1], c='r', lw=1)



    '''
    out = out * 1000
    print('predictor-output:\n{}'.format(out))
    sampling = 44100
    my.Sound.save_sound(out[:,0], '../music/chaos.wav', sampling)
    my.Sound.save_sound(out[:,1], '../music/chaos2.wav', sampling)
    '''

    # random_sound = np.random.rand(pseq_len)*1000
    # save_sound(random_sound.astype(np.int), '../music/random.wav', sampling)

    # In case of No Learning
    if compare:
        Wi = tf.Variable(weight(shape=[input_units, inner_units]), name='Wi')
        Wo = tf.Variable(weight(shape=[inner_units, output_units]), name='Wo')

        init_op = tf.global_variables_initializer()
        psess.run(init_op)

        data_nolearn = make_data(pseq_len, loop=1)
        output = inference(inputs, Wi, Wo)
        feed_dict={inputs:data_nolearn}
        l = []
        out = psess.run(output, feed_dict=feed_dict)
        for i in range(output_units):
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

        if is_plot:

            '''
            plt.figure()
            plt.scatter(range(pseq_len), out[:,0], c='b', s=1)
            plt.figure()
            plt.scatter(range(pseq_len), out[:,1], c='r', s=1)
            '''
            plt.figure()
            plt.plot(x[0:pseq_len-1-tau], x[tau:pseq_len-1], c='b', lw=1)
            plt.figure()
            plt.plot(y[0:pseq_len-1-tau], y[tau:pseq_len-1], c='b', lw=1)

        '''
        out_nolearn = psess.run(output, feed_dict=feed_dict)
        print('no-learning-output:\n{}'.format(out_nolearn))

        out_nolearn = out_nolearn * float(10000)
        print('no-learning-output:\n{}'.format(out_nolearn))
        my.Sound.save_sound(out_nolearn[:,0], '../music/chaos_no.wav', sampling)
        my.Sound.save_sound(out_nolearn[:,1], '../music/chaos_no2.wav', sampling)
        '''

    if is_plot:
        plt.show()


def opt(x):
    kf = x[:,0]
    kr = x[:,1]
    alpha = x[:,2]

    # return np.sin(kf*kr+alpha)

    oseq_len = 5
    osess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')

    saver.restore(osess, model_path + 'model.ckpt')

    graph = tf.get_default_graph()
    Wi = graph.get_tensor_by_name("Wi/Wi:0")
    Wo = graph.get_tensor_by_name("Wo/Wo:0")

    print('optimize')
    inputs = make_data(oseq_len)
    output = inference(inputs, Wi, Wo)
    error, lyapunov = loss(output)

    return osess.run(error)

def np_get_lyapunov(seq):
    # print('Measuring lyapunov...')
    dt = 1/len(seq)
    diff = np.abs(np.diff(seq))
    lyapunov = np.mean(np.log1p(diff/dt)-np.log(2.0))

    return lyapunov

def learning_():

    if MODE == 'train':
        sess = tf.InteractiveSession()

        with tf.name_scope('Mode'):
            Mode = tf.placeholder(dtype = tf.bool, name='Mode')
            mode = True

        norm_in, inputs, outputs, params = inference(seq_len)
        Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

        error, pdf = loss(norm_in, outputs, seq_len, Mode)

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
        train_step, grad = train(error, [Wi, bi, Wo])

        '''
        with tf.name_scope('lyapunov'):
            for i in range(output_units):
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])
        '''

        # Tensorboard logfile
        if tf.gfile.Exists(log_path):
            tf.gfile.DeleteRecursively(log_path)
        writer = tf.summary.FileWriter(log_path, sess.graph)

        run_options = tf.RunOptions(output_partition_graphs=True)
        run_metadata = tf.RunMetadata()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # For Debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        merged = tf.summary.merge_all()

        
        '''
        # confirm
        data = make_data(seq_len, loop=0)
        feed_dict = {inputs:data}
        d, out = sess.run([dmxo, outputs], feed_dict=feed_dict)
        # print(out)
        # print((out[:,0]-min(out[:,0]))/(max(out[:,0])-min(out[:,0])))
        plt.figure()
        # plt.scatter(d[:,0], d[:,1], s=1)
        plt.plot(x, d, lw=1)
        if is_plot:
            plt.show()
        '''
        
        ic = info_content.info_content() 
        
        lcluster = 500
        epoch_cluster = math.ceil(lcluster/seq_len)
        print('epoch_cluster: ', epoch_cluster)
        out_cluster, lyapunov = [], []
        dt = 1/lcluster

        l_list, te_list, time_list = [], [], []
        out_sound1, out_sound2 = [], []
        in_color, out_color = 'r', 'b'

        for epoch in range(epoch_size):
            print('\n[epoch:{}-times]'.format(epoch))
            data = make_data(seq_len, loop=epoch)
            feed_dict = {inputs:data, Mode:mode}

            start = time.time()
            wi, wo, in_, out, error_val = sess.run([Wi, Wo, norm_in, outputs, error], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

            # run for pdf
            dx, dxy = sess.run([dmxo, dmxyo], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

            summary, t, gradients = sess.run([merged, train_step, grad], feed_dict)
            end = time.time()

            indata, outdata = [in_[:,0], out[:,0]]
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
                lyapunov.append(np_get_lyapunov(out_cluster))
                out_cluster = []
                

            # l_list.append(l[0])
            te_list.append(ic.get_TE2(outdata, indata))

            elapsed_time = end-start
            time_list.append(elapsed_time)

            if epoch%1 == 0:
                total_time = np.mean(time_list)*epoch_size
                cumulative_time = sum(time_list)
                remineded_time = total_time - cumulative_time

                print("elapsed_time: {}sec/epoch".format(int(elapsed_time)))
                print("Estimated-reminded-time: {}sec({}sec/{}sec)".format(int(remineded_time), int(cumulative_time), int(total_time)))
                print("error:{}".format(error_val))
                print("Transfer-Entropy: ", ic.get_TE2(outdata, indata))
                # print(d)
                # print(out)

            if is_plot and epoch%(epoch_size-1) == 0:

                in1, in2, out1, out2 = in_[:,0], in_[:,1], out[:,0], out[:,1]

                if False:
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

                if False:
                    plt.figure()
                    plt.title('delayed-input-2Dgraph(epoch:{})'.format(epoch))
                    plt.plot(in1[:-tau], in1[tau:], c='r', lw=1, label='input')
                    plt.plot(in2[:-tau], in2[tau:], c='b', lw=1, label='input')
                    plt.legend(loc=2)

                    plt.figure()
                    plt.title('delayed-out-2Dgraph(epoch:{})'.format(epoch))
                    plt.plot(out1[:-tau], out1[tau:], c='r', lw=1, label='output')
                    plt.plot(out2[:-tau], out2[tau:], c='b', lw=1, label='output')
                    plt.legend(loc=2)

                if False:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_title('delayed-out-3Dgraph(epoch:{})'.format(epoch))
                    ax.scatter3D(in1[:-2*tau],in1[tau:-tau],in1[2*tau:], c=in_color, label='input')
                    ax.scatter3D(out1[:-2*tau],out1[tau:-tau],out1[2*tau:], c=out_color, label='output')
                    plt.legend(loc=2)

            writer.add_summary(summary, epoch)

        # plt.plot(range(epoch_size), (l_list), c='b', lw=1)

        sampling_freq = 14700
        my.Sound.save_sound((np.array(out_sound1)-0.5)*40000, '../music/chaos1.wav', sampling_freq)
        my.Sound.save_sound((np.array(out_sound2)-0.5)*40000, '../music/chaos2.wav', sampling_freq)

        if False:
            plt.figure()
            plt.title('Transfer Entropy')
            plt.plot(range(len(te_list)), te_list)

        if False:
            plt.figure()
            plt.title('Disposal Time')
            plt.plot(range(len(time_list)-1), time_list[1:])

        if True:
            plt.figure()
            plt.title('Lyapunov Exponent')
            plt.plot(range(len(lyapunov)), lyapunov)
            print('Mean-Lyapunov-Value: ', np.mean(lyapunov))

            lcluster = 500
            lyapunov_sin = (np_get_lyapunov(np.sin(np.linspace(0, 1, lcluster))))
            lyapunov_random = (np_get_lyapunov(np.random.rand(lcluster)))
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

        if is_save:
            '''
            # 特定の変数だけ保存するときに使用
            train_vars = tf.trainable_variables()
            '''
            saver = tf.train.Saver()
            saver.save(sess, model_path + 'model.ckpt')

            '''
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print(sess.run(Wi))
            '''
        # print("output:{}".format(out))
        out = np.array(out)

        if is_plot:
            plt.show()

        # sess.close()

    elif MODE == 'predict':
        predict()

    elif MODE == 'opt':
        bounds = [{'name': 'kf',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'kr',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'alpha', 'type': 'continuous',  'domain': (0.0, 100.0)}]

        # Do Presearch
        opt_mnist = GPyOpt.methods.BayesianOptimization(f=opt, domain=bounds)

        # Search Optimized Parameter
        opt_mnist.run_optimization(max_iter=10)
        print("optimized parameters: {0}".format(opt_mnist.x_opt))
        print("optimized loss: {0}".format(opt_mnist.fx_opt))

def learning():

    if MODE == 'train':
        sess = tf.InteractiveSession()

        with tf.name_scope('Mode'):
            Mode = tf.placeholder(dtype = tf.bool, name='Mode')
            mode = True

        norm_in, inputs, outputs, params = inference(seq_len)
        Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

        error, pdf = loss(norm_in, outputs, seq_len, Mode)

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
        train_step, grad = train(error, [Wi, bi, Wo])

        '''
        with tf.name_scope('lyapunov'):
            for i in range(output_units):
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])
        '''

        # Tensorboard logfile
        if tf.gfile.Exists(log_path):
            tf.gfile.DeleteRecursively(log_path)
        writer = tf.summary.FileWriter(log_path, sess.graph)

        run_options = tf.RunOptions(output_partition_graphs=True)
        run_metadata = tf.RunMetadata()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # For Debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        merged = tf.summary.merge_all()

        
        '''
        # confirm
        data = make_data(seq_len, loop=0)
        feed_dict = {inputs:data}
        d, out = sess.run([dmxo, outputs], feed_dict=feed_dict)
        # print(out)
        # print((out[:,0]-min(out[:,0]))/(max(out[:,0])-min(out[:,0])))
        plt.figure()
        # plt.scatter(d[:,0], d[:,1], s=1)
        plt.plot(x, d, lw=1)
        if is_plot:
            plt.show()
        '''
        
        ic = info_content.info_content() 
        
        lcluster = 500
        epoch_cluster = math.ceil(lcluster/seq_len)
        print('epoch_cluster: ', epoch_cluster)
        out_cluster, lyapunov = [], []
        dt = 1/lcluster

        l_list, te_list, time_list = [], [], []
        out_sound1, out_sound2 = [], []
        in_color, out_color = 'r', 'b'

        for epoch in range(epoch_size):
            print('\n[epoch:{}-times]'.format(epoch))
            data = make_data(seq_len, loop=epoch)
            feed_dict = {inputs:data, Mode:mode}

            start = time.time()
            wi, wo, in_, out, error_val = sess.run([Wi, Wo, norm_in, outputs, error], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

            # run for pdf
            dx, dxy = sess.run([dmxo, dmxyo], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)

            summary, t, gradients = sess.run([merged, train_step, grad], feed_dict)
            end = time.time()

            indata, outdata = [in_[:,0], out[:,0]]
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
                lyapunov.append(np_get_lyapunov(out_cluster))
                out_cluster = []
                

            # l_list.append(l[0])
            te_list.append(ic.get_TE2(outdata, indata))

            elapsed_time = end-start
            time_list.append(elapsed_time)

            if epoch%1 == 0:
                total_time = np.mean(time_list)*epoch_size
                cumulative_time = sum(time_list)
                remineded_time = total_time - cumulative_time

                print("elapsed_time: {}sec/epoch".format(int(elapsed_time)))
                print("Estimated-reminded-time: {}sec({}sec/{}sec)".format(int(remineded_time), int(cumulative_time), int(total_time)))
                print("error:{}".format(error_val))
                print("Transfer-Entropy: ", ic.get_TE2(outdata, indata))
                # print(d)
                # print(out)

            if is_plot and epoch%(epoch_size-1) == 0:

                in1, in2, out1, out2 = in_[:,0], in_[:,1], out[:,0], out[:,1]

                if False:
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

                if False:
                    plt.figure()
                    plt.title('delayed-input-2Dgraph(epoch:{})'.format(epoch))
                    plt.plot(in1[:-tau], in1[tau:], c='r', lw=1, label='input')
                    plt.plot(in2[:-tau], in2[tau:], c='b', lw=1, label='input')
                    plt.legend(loc=2)

                    plt.figure()
                    plt.title('delayed-out-2Dgraph(epoch:{})'.format(epoch))
                    plt.plot(out1[:-tau], out1[tau:], c='r', lw=1, label='output')
                    plt.plot(out2[:-tau], out2[tau:], c='b', lw=1, label='output')
                    plt.legend(loc=2)

                if False:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_title('delayed-out-3Dgraph(epoch:{})'.format(epoch))
                    ax.scatter3D(in1[:-2*tau],in1[tau:-tau],in1[2*tau:], c=in_color, label='input')
                    ax.scatter3D(out1[:-2*tau],out1[tau:-tau],out1[2*tau:], c=out_color, label='output')
                    plt.legend(loc=2)

            writer.add_summary(summary, epoch)

        # plt.plot(range(epoch_size), (l_list), c='b', lw=1)

        sampling_freq = 14700
        my.Sound.save_sound((np.array(out_sound1)-0.5)*40000, '../music/chaos1.wav', sampling_freq)
        my.Sound.save_sound((np.array(out_sound2)-0.5)*40000, '../music/chaos2.wav', sampling_freq)

        if False:
            plt.figure()
            plt.title('Transfer Entropy')
            plt.plot(range(len(te_list)), te_list)

        if False:
            plt.figure()
            plt.title('Disposal Time')
            plt.plot(range(len(time_list)-1), time_list[1:])

        if True:
            plt.figure()
            plt.title('Lyapunov Exponent')
            plt.plot(range(len(lyapunov)), lyapunov)
            print('Mean-Lyapunov-Value: ', np.mean(lyapunov))

            lcluster = 500
            lyapunov_sin = (np_get_lyapunov(np.sin(np.linspace(0, 1, lcluster))))
            lyapunov_random = (np_get_lyapunov(np.random.rand(lcluster)))
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

        if is_save:
            '''
            # 特定の変数だけ保存するときに使用
            train_vars = tf.trainable_variables()
            '''
            saver = tf.train.Saver()
            saver.save(sess, model_path + 'model.ckpt')

            '''
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print(sess.run(Wi))
            '''
        # print("output:{}".format(out))
        out = np.array(out)

        if is_plot:
            plt.show()

        # sess.close()

    elif MODE == 'predict':
        predict()

    elif MODE == 'opt':
        bounds = [{'name': 'kf',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'kr',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'alpha', 'type': 'continuous',  'domain': (0.0, 100.0)}]

        # Do Presearch
        opt_mnist = GPyOpt.methods.BayesianOptimization(f=opt, domain=bounds)

        # Search Optimized Parameter
        opt_mnist.run_optimization(max_iter=10)
        print("optimized parameters: {0}".format(opt_mnist.x_opt))
        print("optimized loss: {0}".format(opt_mnist.fx_opt))

if __name__ == "__main__":
def training2():
    sessA = tf.InteractiveSession()
    sessB = tf.InteractiveSession()

    with tf.name_scope('Mode'):
        Mode = tf.placeholder(dtype = tf.bool, name='Mode')

    norm_in, inputs, outputs, params = inference(seq_len)
    Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

    error, pdf = loss(norm_in, outputs, seq_len, Mode)
    tf.summary.scalar('error', error)
    train_step, grad = train(error, [Wi, bi, Wo])

    merged = tf.summary.merge_all()

    # Tensorboard logfile
    log_pathA = '../Alogdir'
    log_pathB = '../Blogdir'

    if tf.gfile.Exists(log_pathA):
        tf.gfile.DeleteRecursively(log_pathA)
    writerA = tf.summary.FileWriter(log_pathA, sessA.graph)

    if tf.gfile.Exists(log_pathB):
        tf.gfile.DeleteRecursively(log_pathB)
    writerB = tf.summary.FileWriter(log_pathB, sessB.graph)


    sessA.run(tf.global_variables_initializer())
    sessB.run(tf.global_variables_initializer())

    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(10,6))

    # True: Following, False: Creative
    modeA = True
    modeB = False

    online_update = True
    online_update = False

    trajectoryA = []
    trajectoryB = []

    outB = make_data(seq_len)
    outB = np.random.rand(seq_len, 2)
    for epoch in range(epoch_size):
        print('epoch: ', epoch)

        if epoch%1 == 0:
            # modeA, modeB = modeB, modeA
            pass

        # colorA, colorB = ('r','b') if modeA else ('g','m')
        colorA, colorB = 'r', 'b'

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
            for i in range(seq_len):
                trajectoryA.extend([np.array(trajectoryA[-1] if len(trajectoryA) != 0 else [0,0]) + np.array(outA[i])-0.5])
                trajectoryB.extend([np.array(trajectoryB[-1] if len(trajectoryB) != 0 else [0,0]) + np.array(outB[i])-0.5])

                plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+colorA, lw=0.1, label='A')
                plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+colorB, lw=0.1, label='B')
                plt.pause(0.01)

        
    if not online_update:
        plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+colorA, lw=0.1, label='A')
        plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+colorB, lw=0.1, label='B')


    print('Finish')
    plt.show()


def kbevent(event):
    print(event.Key)
    pos = 0

    if event.Ascii == 32:
        print('::space')
    if event.Key == 'Up':
        print('::UP')
        pos = pos + 1
        log = open('../key.log', 'a')
        log.write(str(pos))



# def test():
if __name__ == "__main__":
    sessA = tf.InteractiveSession()

    with tf.name_scope('Mode'):
        Mode = tf.placeholder(dtype = tf.bool, name='Mode')

    norm_in, inputs, outputs, params = inference(seq_len)
    Wi, bi, Wo, bo = params['Wi'], params['bi'], params['Wo'], params['bo']

    error, pdf = loss(norm_in, outputs, seq_len, Mode)
    tf.summary.scalar('error', error)
    train_step, grad = train(error, [Wi, bi, Wo])

    merged = tf.summary.merge_all()

    # Tensorboard logfile
    log_pathA = '../Alogdir'

    if tf.gfile.Exists(log_pathA):
        tf.gfile.DeleteRecursively(log_pathA)
    writerA = tf.summary.FileWriter(log_pathA, sessA.graph)

    sessA.run(tf.global_variables_initializer())

    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure(figsize=(10,6))

    '''
    hookman = pyxhook.HookManager()
    hookman.KeyDown = kbevent
    hookman.HookKeyboard()
    hookman.start()
    '''

    # True: Following, False: Creative
    modeA = True

    online_update = False
    online_update = True

    trajectoryA = []
    trajectoryB = []

    outB = np.random.rand(seq_len, 2)
    colorA, colorB = 'r', 'b'
    for epoch in range(epoch_size):
        print('epoch: ', epoch)

        # colorA, colorB = ('r','b') if modeA else ('g','m')

        feed_dictA = {inputs:outB, Mode:modeA}

        outA, gradientsA = sessA.run([outputs, grad], feed_dict=feed_dictA)

        
        if epoch % 10:
            for (g, v) in gradientsA:
                print('gradA: ', g[0][0:5])

        summaryA, _ = sessA.run([merged, train_step], feed_dictA)

        writerA.add_summary(summaryA, epoch)

        print('[A] mode={}, value={}'.format(modeA, np.array(outA[0])-0.5))

        if not online_update:
            trajectoryA.extend((trajectoryA[-1] if len(trajectoryA) != 0 else np.zeros(len(outA[0]))) + np.cumsum(np.array(outA)-0.5, axis=0))
            trajectoryB.extend((trajectoryB[-1] if len(trajectoryB) != 0 else np.zeros(len(outB[0]))) + np.cumsum(np.array(outB)-0.5, axis=0))

        if online_update:
            for i in range(seq_len):
                trajectoryA.extend([np.array(trajectoryA[-1] if len(trajectoryA) != 0 else [0,0]) + np.array(outA[i])-0.5])
                trajectoryB.extend([np.array(trajectoryB[-1] if len(trajectoryB) != 0 else [0,0]) + np.array(outB[i])-0.5])

                plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+colorA, lw=0.1, label='A')
                plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+colorB, lw=0.1, label='B')
                plt.pause(0.01)

        
    if not online_update:
        plt.plot([x[0] for x in trajectoryA], [x[1] for x in trajectoryA], '.-'+colorA, lw=0.1, label='A')
        plt.plot([x[0] for x in trajectoryB], [x[1] for x in trajectoryB], '.-'+colorB, lw=0.1, label='B')


    print('Finish')
    plt.show()

    # tf.app.run()


