# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import chaotic_nn_cell
import matplotlib.pyplot as plt
import math

''' ベイズ最適化
'''
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

activation = tf.nn.tanh

# 1秒で取れるデータ数に設定(1秒おきにリアプノフ指数計測)
seq_len = 100

epoch_size = 10000
input_units = 2
inner_units = 200
output_units = input_units

Kf = 26.654
Kr = 16.553
Alpha = 86.721

Kf = 49.471
Kr = 24.736
Alpha = 32.889

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

def make_data(length):
    with tf.name_scope('data'):
        r = tf.random_uniform(shape=[input_units*3])*10
        # r = tf.zeros([input_units*3])+1

        line = tf.lin_space(0.0, 2*math.pi, num=length)

        values = []

        if input_units == 2:
            values.append(tf.random_uniform(shape=[length])*10)
            values.append(tf.sin(line)+0.1*tf.random_uniform(shape=[length]))
            # values.append(tf.sin(line)+0.1*tf.random_uniform(shape=[length]))
        else:
            for i in range(input_units):
                values.append(r[3*i] * tf.sin(r[3*i+1]*line+r[3*i+2]))

        inputs = tf.transpose(tf.reshape(values, shape=[input_units, length]))

        # random-input
        # inputs = tf.random_uniform(shape=[length, input_units])

    return inputs

def predict():
    compare = False
    pseq_len = 10000
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
    inputs = make_data(pseq_len)
    output = inference(inputs, Wi, Wo)
    l = []
    out = psess.run(output)
    for i in range(output_units):
        # print('output[:,i]: {}'.format(out[:,i]))
        l.append(get_lyapunov(seq=out[:,i]))
        print('predict-lyapunov:{}'.format(psess.run([l[i]])))

    out = np.array(out)
    print('predictor-output:\n{}'.format(out))

    plt.scatter(range(pseq_len), out[:,0], c='b', s=1)
    plt.figure()
    plt.scatter(range(pseq_len), out[:,1], c='r', s=1)
    plt.figure()
    plt.scatter(out[:,0], out[:,1], c='r', s=1)
    plt.show()


    # In case of No Learning
    if compare:
        Wi = tf.Variable(weight(shape=[input_units, inner_units]), name='Wi')
        Wo = tf.Variable(weight(shape=[inner_units, output_units]), name='Wo')

        init_op = tf.global_variables_initializer()
        psess.run(init_op)

        inputs = make_data(pseq_len)
        output = inference(inputs, Wi, Wo)
        l = []
        for i in range(output_units):
            l.append(get_lyapunov(seq=output[:,i]))
            print('no-learning-lyapunov::{}'.format(psess.run([l[i]])))

        print('no-learning-output:\n{}'.format(psess.run([output])))

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
        
        inputs = make_data(seq_len)

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

        feed_dict = {}
        merged = tf.summary.merge_all()
        
        for epoch in range(epoch_size):
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
        plt.scatter(out[:,0], out[:,1], c='r', s=1)
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


