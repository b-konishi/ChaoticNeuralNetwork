import tensorflow as tf
import numpy as np


class ChaoticNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, Kf, Kr, alpha, activation):
        
        self._num_units = num_units
        self._output_size = num_units
        self._Kf = Kf
        self._Kr = Kr
        self._alpha = alpha
        self._activation = activation

        self._state_size = tuple([self._num_units] * 3)


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        print('inputs::{}'.format(inputs))

        (e_prev, z_prev, o_prev) = state

        '''
        Kf = tf.Variable(self._Kf, name='Kf', dtype=tf.float32)
        Kr = tf.Variable(self._Kr, name='Kr', dtype=tf.float32)
        alpha = tf.Variable(self._alpha, name='alpha', dtype=tf.float32)
        '''
        
        '''
        tf.summary.scalar('Kf', Kf)
        tf.summary.scalar('Kr', Kr)
        tf.summary.scalar('alpha', alpha)
        '''

        '''
        print('e_prev::{}'.format(e_prev))
        print('z_prev::{}'.format(z_prev))
        print('o_prev::{}'.format(o_prev))
        '''

        # センサ値入力を省略中
        e = self._Kf * e_prev + inputs
        z = self._Kr * z_prev - self._alpha * o_prev
        o = self._activation(e + z)

        new_state = tuple([e, z, o])

        return o, new_state

        


