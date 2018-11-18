# -*- coding: utf-8 -*-

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

# Baysian Optimization
import GPy
import GPyOpt


model_path = '../model/'
log_path = '../logdir'

MODE = 'opt'
MODE = 'predict'
MODE = 'train'

# Save the model
is_save = False
is_save = True

# Drawing graphs flag
is_plot = False
is_plot = True

activation = tf.nn.tanh

# 1秒で取れるデータ数に設定(1秒おきにリアプノフ指数計測)
seq_len = 100

epoch_size = 100
input_units = 2
inner_units = 10
output_units = 2

# 中間層層数
inner_layers = 1

Kf = 0.1
Kr = 0.9
Alpha = 10.0

# Kf, Kr, Alpha = 0, 0, 0

# time delayed value
tau = int(seq_len/100)
tau = 1

def make_data(length, loop=0):
    # print('making data...')

    sound1 = my.Sound.load_sound('../music/jinglebells.wav')
    sound2 = my.Sound.load_sound('../music/ifudoudou.wav')
    sound1 = my.Sound.load_sound('../music/ifudoudou.wav')
    sound2 = my.Sound.load_sound('../music/jinglebells.wav')

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
    # Normalization
    for i in range(input_units):
        d = data[:,i]
        if np.max(d) != np.min(d):
            norm = (d-np.min(d))/(np.max(d)-np.min(d))
        else:
            norm = np.clip(np.sign(d), 0., 1.)
        data[:,i] = norm

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
    norm = (v-tf.reduce_min(v,0))/(tf.reduce_max(v,0)-tf.reduce_min(v,0))
    sign = tf.nn.relu(tf.sign(v))
    var = tf.where(tf.is_nan(norm), sign, norm)

    return var


def inference(inputs, Wi, bi, Wo, bo):
    with tf.name_scope('Input_Layer'):
        # input: [None, input_units]
        i = normalize(inputs)
        fi = tf.matmul(i, Wi) + bi
        sigm = tf.nn.sigmoid(fi)

    inner_output = set_innerlayers(sigm, inner_layers)

    with tf.name_scope('Output_Layer'):
        fo = tf.matmul(inner_output, Wo) + bo
        o = normalize(fo)

    return o

def get_lyapunov(seq, dt=1/seq_len):
    with tf.name_scope('Normalization'):
        seq_max = tf.reduce_max(seq)
        seq_min = tf.reduce_min(seq)

        seq = (seq-seq_min)/(seq_max-seq_min)
        
    '''
        moment = tf.nn.moments(seq, [0])
        m = moment[0]
        v = moment[1]
        seq = (seq-m)/tf.sqrt(v)
    '''
    
    diff = tf.abs(seq[1:]-seq[:-1])

    '''
    本来リアプノフ指数はmean(log(f'))だが、f'<<0の場合に-Infとなってしまうため、
    mean(log(1+f'))とする。しかし、それだとこの式ではカオス性を持つかわからなくなってしまう。
    カオス性の境界log(f')=0のとき、f'=1
    log(1+f')+alpha=log(2)+alpha=0となるようにalphaを定めると
    alpha=-log(2)
    となるため、プログラム上のリアプノフ指数の定義は、
    mean(log(1+f')-log(2))とする（0以上ならばカオス性を持つという性質は変わらない）
    '''
    lyapunov = tf.reduce_mean(tf.log1p(diff/dt)-tf.log(tf.cast(2.0, tf.float32)))
    # lyapunov = tf.reduce_mean(tf.log1p(diff))

    return lyapunov


def loss(inputs, outputs, length):
    print('input::{}'.format(inputs[:,0]))
    print('output::{}'.format(outputs[:,0]))

    # Laypunov-Exponent
    lyapunov = []
    for i in range(output_units):
        lyapunov.append(get_lyapunov(outputs[:,i]))

    with tf.name_scope('loss_tf'):
        in_data, out_data = inputs[:,0], outputs[:,0]

        ic = info_content.info_content()
        # TF(y->x): input->output
        entropy, pdf = ic.get_TE_for_tf4(out_data, in_data, seq_len)
        print('entropy_shape: ', entropy)

    return -entropy, lyapunov, pdf

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
        opt = tf.train.AdamOptimizer(1.0)
        grads = opt.compute_gradients(error, var_list=update_params)
        grad = grads[0]
        g = (tf.where(tf.is_nan(grad[0]), tf.zeros_like(grad[0]), grad[0]),)
        g += grad[1:]
        grads = [g]
        
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
    my.Sound.save_sound(sampling, out[:,0], '../music/chaos.wav')
    my.Sound.save_sound(sampling, out[:,1], '../music/chaos2.wav')
    '''

    # random_sound = np.random.rand(pseq_len)*1000
    # save_sound(sampling, random_sound.astype(np.int), '../music/random.wav')

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
        my.Sound.save_sound(sampling, out_nolearn[:,0], '../music/chaos_no.wav')
        my.Sound.save_sound(sampling, out_nolearn[:,1], '../music/chaos_no2.wav')
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

def main(_):

    if MODE == 'train':
        sess = tf.InteractiveSession()

        with tf.name_scope('data'):
            inputs = tf.placeholder(dtype = tf.float32, shape = [seq_len, input_units], name='inputs')
        with tf.name_scope('Wi'):
            Wi = tf.Variable(weight(shape=[input_units, inner_units]), name='Wi')
            tf.summary.histogram('Wi', Wi)

        with tf.name_scope('bi'):
            bi = tf.Variable(weight(shape=[inner_units]), name='bi')
            tf.summary.histogram('bi', bi)

        with tf.name_scope('Wo'):
            Wo = tf.Variable(weight(shape=[inner_units, output_units]), name='Wo')
            tf.summary.histogram('Wo', Wo)

        with tf.name_scope('bo'):
            bo = tf.Variable(weight(shape=[output_units]), name='bo')
            tf.summary.histogram('bo', bo)

        outputs = inference(inputs, Wi, bi, Wo, bo)

        error, lyapunov, pdf = loss(inputs, outputs, seq_len)
        dm, dmx, dmxx, dmxy = pdf['dm'], pdf['dmx'], pdf['dmxx'], pdf['dmxy']
        x = np.reshape(np.linspace(-0.5,1.5,10000), [10000,1])
        dmxo = dmx.prob(x)

        tf.summary.scalar('error', error)
        train_step, grad = train(error, [Wo])

        with tf.name_scope('lyapunov'):
            for i in range(output_units):
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])

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
        
        l_list = []
        # data = []
        for epoch in range(epoch_size):
            print('\n[epoch:{}-times]'.format(epoch))
            data = make_data(seq_len, loop=epoch)
            feed_dict = {inputs:data}

            start = time.time()
            # summary, out, error_val, l, d = sess.run([merged, outputs, error, lyapunov, dmxo], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)
            wi, out, error_val, l, d = sess.run([Wi, outputs, error, lyapunov, dmxo], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)
            # sess.run([ops], run_metadata=run_metadata, options=run_options)
            summary, t, gradients = sess.run([merged, train_step, grad], feed_dict)
            end = time.time()
            indata, outdata = [data[:,0], out[:,0]]
            print('wi: ', wi)
            print('in: ', indata[0:100])
            print('out: ', outdata[0:100])
            print('grad: ', gradients)

            l_list.append(l[0])

            '''
            data1 = (data[:,0]-min(data[:,0]))/(max(data[:,0])-min(data[:,0]))
            out1 = (out[:,0]-min(out[:,0]))/(max(out[:,0])-min(out[:,0]))
            '''
            
            if epoch%1 == 0:
                print("elapsed_time: ", int((end-start)*1000), '[sec]')
                print("error:{}".format(error_val))
                print("Transfer-Entropy: ", ic.get_TE2(outdata, indata))
                # print(d)
                # print(out)

            if is_plot and epoch%(epoch_size-1) == 0:

                '''
                data2 = data1[int(seq_len/2):int(seq_len/2)+100]
                out2 = out1[int(seq_len/2):int(seq_len/2)+100]
                '''

                data1, out1 = data[:,0], out[:,0]
                data2 = (data1-min(data1))/(max(data1)-min(data1))
                out2 = (out1-min(out1))/(max(out1)-min(out1))

                plt.figure()
                plt.title('time-data-graph(epoch:{})'.format(epoch))
                plt.plot(range(len(data2)), data2, c='b', lw=1, label='input')
                plt.plot(range(len(data2)), -data2+1, c='b', lw=1, label='-input')
                plt.plot(range(len(out2)), out2, c='r', lw=1, label='output')
                plt.legend(loc=2)

                '''
                plt.figure()
                plt.title('delayed-data-graph(epoch:{})'.format(epoch))
                plt.plot(data1[:-tau], data1[tau:], c='r', lw=0.1)

                fig = plt.figure()
                plt.title('delayed-out-graph(epoch:{})'.format(epoch))
                plt.plot(out1[:-tau], out1[tau:], c='r', lw=0.1)

                ax = fig.add_subplot(111,projection='3d')
                ax.set_title('delayed-out-graph(epoch:{})'.format(epoch))
                ax.scatter3D(out1[:-2*tau],out1[tau:-tau],out1[2*tau:])
                '''
            

            # print("output:{}".format(out))
            # print("lyapunov:{}".format(sess.run(tf.reduce_max(error_val))))

            '''
            graph = run_metadata.partition_graphs[0]
            writer = tf.summary.FileWriter(logdir=log_path, graph=graph)
            writer.flush()
            '''
            writer.add_summary(summary, epoch)

        # plt.plot(range(epoch_size), (l_list), c='b', lw=1)

        # 混合ベイズ分布ができているか確認
        plt.figure()
        # plt.scatter(d[:,0], d[:,1], s=1)
        plt.plot(x, d, lw=2)

        

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
            pass
            '''
            plt.figure()
            data = (data[:,0]-min(data[:,0]))/(max(data[:,0])-min(data[:,0]))
            out = (out[:,0]-min(out[:,0]))/(max(out[:,0])-min(out[:,0]))

            plt.plot(range(seq_len), data, c='b', lw=1)
            plt.plot(range(seq_len), out, c='r', lw=1)

            plt.figure()
            plt.scatter(range(seq_len), out[:,1], c='r', s=1)
            random = np.random.rand(seq_len,1)

            plt.figure()
            plt.title('time-random-graph(epoch:{})'.format(epoch))
            plt.plot(range(100), random[int(seq_len/2):int(seq_len/2+100)], c='b', lw=1)

            plt.figure()
            plt.title('delayed-random-graph(epoch:{})'.format(epoch))
            plt.plot(random[:-tau], random[tau:], c='r', lw=0.1)
            plt.show()
            '''
        if is_plot:
            plt.show()

        sess.close()

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
    tf.app.run()


