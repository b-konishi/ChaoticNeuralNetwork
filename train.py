# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import chaotic_nn_cell
import my_library as my
import matplotlib.pyplot as plt
import math

# ベイズ最適化
import GPy
import GPyOpt

model_path = '../model/'
log_path = '../logdir'

MODE = 'opt'
MODE = 'train'
MODE = 'predict'

# モデルの保存
is_save = False
is_save = True

# グラフ描画
is_plot = True
is_plot = False

activation = tf.nn.tanh

# 1秒で取れるデータ数に設定(1秒おきにリアプノフ指数計測)
seq_len = 44100

epoch_size = 1000
input_units = 2
inner_units = 4
output_units = 2

Kf = 26.654
Kr = 16.553
Alpha = 86.721

'''
Kf = 26.279
Kr = 84.793
Alpha = 46.165
'''

def make_data(length, loop=0):
    print('making data...')

    sound1 = my.Sound.load_sound('../music/ifudoudou.wav')
    sound2 = my.Sound.load_sound('../music/jinglebells.wav')
    sound1 = my.Sound.load_sound('../music/jinglebells.wav')
    sound2 = my.Sound.load_sound('../music/ifudoudou.wav')

    if ((loop+1)*length > len(sound1) or (loop+1)*length > len(sound2)):
        loop = 0

    sound1 = sound1[loop*length:(loop+1)*length].reshape(length,1) * 10000
    sound2 = sound2[loop*length:(loop+1)*length].reshape(length,1) * 10000


    x1 = np.linspace(start=0, stop=length, num=length)
    y1 = np.sin(x1)

    x2 = np.linspace(start=0, stop=length, num=length)
    y2 = np.sin(2*x2)

    data = np.resize(np.transpose([sound1,sound2]),(length, input_units))
    data = np.resize(np.transpose([y1,y2]),(length, input_units))

    '''
    plt.figure()
    plt.plot(range(length), data[:,0], c='b', lw=1)
    plt.figure()
    plt.plot(range(length), data[:,1], c='b', lw=1)
    '''
    
    return data

def weight(shape = []):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    # return tf.Variable(initial)
    return initial

def inference(inputs, Wi, Wo):
    with tf.name_scope('Layer1'):
        # input: [None, input_units]
        fi = tf.matmul(inputs, Wi)
        sigm = tf.nn.sigmoid(fi)

    with tf.name_scope('Layer2'):
        cell = chaotic_nn_cell.ChaoticNNCell(num_units=inner_units, Kf=Kf, Kr=Kr, alpha=Alpha, activation=activation)
        outputs, state = tf.nn.static_rnn(cell=cell, inputs=[sigm], dtype=tf.float32)
        inner_output = outputs[-1]

    with tf.name_scope('Layer3'):
        fo = tf.matmul(inner_output, Wo)
        # tf.summary.histogram('fo', fo)

    return fo

def get_lyapunov(seq, dt=1/seq_len):
   seq_shift = tf.manip.roll(seq, shift=1, axis=0)
   diff = tf.abs(seq - seq_shift)

   '''
   本来リアプノフ指数はmean(log(f'))だが、f'<<0の場合に-Infとなってしまうため、
   mean(log(1+f'))とする。しかし、それだとこの式ではカオス性を持つかわからなくなってしまう。
   カオス性の境界log(f')=0のとき、f'=1
   log(1+f')+alpha=log(2)+alpha=0となるようにalphaを定めると
   alpha=-log(2)
   となるため、プログラム上のリアプノフ指数の定義は、
   mean(log(1+f')-log(2))とする（0以上ならばカオス性を持つという性質は変わらない）
   '''
   lyapunov = tf.reduce_mean(tf.log1p(diff/dt)-tf.log(2.0))

   return lyapunov

def loss(output):
    with tf.name_scope('loss'):
        # リアプノフ指数が増加するように誤差関数を設定
        print('output::{}'.format(output))
        lyapunov = []
        loss = []
        for i in range(output_units):
            lyapunov.append(get_lyapunov(output[:,i]))
            loss.append(1/(1+tf.exp(lyapunov[i])))
            # loss.append(tf.exp(-lyapunov[i]))
            # loss.append((-lyapunov[i]))


        # リアプノフ指数を取得(評価の際は最大リアプノフ指数をとる)
        # return tf.reduce_sum(loss), lyapunov
        return tf.reduce_max(loss), lyapunov
        # return tf.reduce_min(loss), lyapunov
        # return loss, lyapunov

def train(error):
    # return tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error)
    with tf.name_scope('training'):
        training = tf.train.AdamOptimizer().minimize(error)

    return training



def predict():
    compare = True
    pseq_len = 44100 * 20
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
        # print('output[:,i]: {}'.format(out[:,i]))
        l.append(get_lyapunov(seq=out[:,i]))
        print('predict-lyapunov:{}'.format(psess.run([l[i]], feed_dict=feed_dict)))

    out = np.array(out)
    print('predictor-output:\n{}'.format(out))

    print('num: {}, unique: {}'.format(len(out[:,0]), len(set(out[:,0]))))
    print('mean: {}'.format(np.mean(out)))

    if is_plot:
        plt.figure()
        plt.scatter(range(pseq_len), out[:,0], c='b', s=1)
        plt.figure()
        plt.scatter(range(pseq_len), out[:,1], c='r', s=1)
        # plt.figure()
        # plt.plot(out[:,0], out[:,1], c='r', lw=0.3)

        plt.show()


    out = out * 1000
    print('predictor-output:\n{}'.format(out))
    sampling = 44100
    my.Sound.save_sound(sampling, out[:,0], '../music/chaos.wav')
    my.Sound.save_sound(sampling, out[:,1], '../music/chaos2.wav')

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
        for i in range(output_units):
            l.append(get_lyapunov(seq=output[:,i]))
            print('no-learning-lyapunov::{}'.format(psess.run(l[i], feed_dict=feed_dict)))

        out_nolearn = psess.run(output, feed_dict=feed_dict)
        print('no-learning-output:\n{}'.format(out_nolearn))

        out_nolearn = out_nolearn * float(10000)
        print('no-learning-output:\n{}'.format(out_nolearn))
        my.Sound.save_sound(sampling, out_nolearn[:,0], '../music/chaos_no.wav')
        my.Sound.save_sound(sampling, out_nolearn[:,1], '../music/chaos_no2.wav')


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
            inputs = tf.placeholder(dtype = tf.float32, shape = [None, input_units], name='inputs')



        with tf.name_scope('Wi'):
            Wi = tf.Variable(weight(shape=[input_units, inner_units]), name='Wi')
            tf.summary.histogram('Wi', Wi)

        with tf.name_scope('Wo'):
            Wo = tf.Variable(weight(shape=[inner_units, output_units]), name='Wo')
            tf.summary.histogram('Wo', Wo)

        output = inference(inputs, Wi, Wo)
        error, lyapunov = loss(output)

        with tf.name_scope('lyapunov'):
            for i in range(output_units):
                tf.summary.scalar('lyapunov'+str(i), lyapunov[i])


        tf.summary.scalar('error', error)
        train_step = train(error)


        # Tensorboard logfile
        if tf.gfile.Exists(log_path):
            tf.gfile.DeleteRecursively(log_path)
        writer = tf.summary.FileWriter(log_path, sess.graph)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        merged = tf.summary.merge_all()
        
        for epoch in range(epoch_size):
            data = make_data(seq_len, loop=epoch)
            feed_dict = {inputs:data}

            if epoch%100 == 0:
                print('epoch:{}-times'.format(epoch))

            # print(sess.run(inputs))

            t = sess.run(train_step, feed_dict=feed_dict)
            summary, out, error_val = sess.run([merged, output, error], feed_dict=feed_dict)
            # print("output:{}".format(out))
            # print("error:{}".format(error_val))
            # print("lyapunov:{}".format(sess.run(tf.reduce_max(error_val))))
            writer.add_summary(summary, epoch)

        if is_save:
            # 特定の変数だけ保存するときに使用
            # train_vars = tf.trainable_variables()
            saver = tf.train.Saver()
            saver.save(sess, model_path + 'model.ckpt')

            '''
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print(sess.run(Wi))
            '''
        print("output:{}".format(out))
        out = np.array(out)

        if is_plot:
            plt.figure()
            plt.scatter(range(seq_len), out[:,0], c='b', s=1)
            plt.figure()
            plt.scatter(range(seq_len), out[:,1], c='r', s=1)
            plt.figure()
            plt.plot(out[:,0], out[:,1], c='r', lw=1)
            plt.show()

        sess.close()

    elif MODE == 'predict':
        predict()

    elif MODE == 'opt':
        bounds = [{'name': 'kf',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'kr',    'type': 'continuous',  'domain': (0.0, 100.0)},
                  {'name': 'alpha', 'type': 'continuous',  'domain': (0.0, 100.0)}]
        # 事前探索を行います。
        opt_mnist = GPyOpt.methods.BayesianOptimization(f=opt, domain=bounds)

        # 最適なパラメータを探索します。
        opt_mnist.run_optimization(max_iter=10)
        print("optimized parameters: {0}".format(opt_mnist.x_opt))
        print("optimized loss: {0}".format(opt_mnist.fx_opt))
    

            
if __name__ == "__main__":
    tf.app.run()


