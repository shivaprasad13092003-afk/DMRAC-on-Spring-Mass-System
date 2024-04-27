import numpy as np
import tensorflow.compat.v1 as tf
from replay_buffer import ReplayBuffer
import scipy.linalg as sp


class MRAC(object):

    def __init__(self, sess,A, B, state_dim, action_dim, MRAC_gain = 0.1,lr = 0.001, lr_flag=1, buffer_size = 5000):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.timeStep = 0.05
        self.gain = MRAC_gain
        self.lr = lr
        self.lr_decay = 1
        self.lr_status = lr_flag
        self.buffer_size = buffer_size
        self.batch_size = 50
        self.A = np.array([[0,1,0,0],
                    [-25,-10,0,0],
                    [0,0,0,1],
                    [0,0,-25,-10]])
        Q = 1*np.eye(4)
        self.B = np.array([[0,0],
                           [25,0],
                           [0,0],
                           [0,25]])
        self.P = sp.solve_lyapunov(self.A.transpose(),Q)

        self.kx = (np.linalg.lstsq(B,A-self.A))[0]
        self.kr = (np.linalg.lstsq(B,self.B))[0]

        #Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Network Definition
        self.n_hidden_layer1 = 100
        self.n_hidden_layer2 = 200
        self.n_hidden_layer3 = 10
        self.netWeights = np.zeros((self.n_hidden_layer3,2), dtype = float)
        self.phi = np.zeros((self.n_hidden_layer3,1), dtype = float)
        self._placeholders()
        self.basis, self.out = self._adapNet()
        self.network_params = tf.trainable_variables()
        self.regualizer = 0
        self.grad_fun = tf.gradients(self.basis[0][self.gradIndex], self.state_ph)
        for i in range(len(self.network_params)):
            self.regualizer = self.regualizer + tf.nn.l2_loss(self.network_params[i])
        self.loss = tf.reduce_mean(tf.square(self.delta_ph-self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss + 0.001*self.regualizer)

        #Data RecordingGradientDescent
        self.TOTAL_CNTRL_REC = []
        self.ADAP_CNTRL_REC = []
        self.DADAP_CNTRL_REC = []
        self.NET_PARAM = []

    def _placeholders(self):
        self.state_ph = tf.placeholder(tf.float32, (None, 4), name='state')
        self.delta_ph = tf.placeholder(tf.float32, (None, self.action_dim), name='true_uncertainty')
        self.gradIndex = tf.placeholder(tf.int32)

    def _adapNet(self):
        w1 = self._weight_init([self.state_dim, self.n_hidden_layer1], 'w1')
        b1 = self._base_init([self.n_hidden_layer1], 'b1')

        w2 = self._weight_init([self.n_hidden_layer1,self.n_hidden_layer2], 'w2')
        b2 = self._base_init([self.n_hidden_layer2], 'b2')

        w3 = self._weight_init([self.n_hidden_layer2,self.n_hidden_layer3], 'w3')
        b3 = self._base_init([self.n_hidden_layer3], 'b3')

        w4 = self._weight_init([self.n_hidden_layer3,self.action_dim], 'w4')
        b4 = self._base_init([self.action_dim], 'b4')

        h1 = tf.nn.relu(tf.matmul(self.state_ph, w1)+b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)

        basis = tf.nn.tanh(tf.matmul(h2, w3)+b3)
        
        out = tf.matmul(basis, w4)+b4

        return basis, out

    def evalCNTRL(self, state):
        self.sess.run(self.out, feed_dict={self.state_ph: np.reshape(state, [1, self.state_ph])})

    def updateBasis(self, state):
        basis = self.sess.run(self.basis, feed_dict={self.state_ph:np.reshape(state, [1, self.state_dim])})
        return basis

    def _weight_init(self, shape, var_name):
        initial_value = tf.random.truncated_normal(shape)
        return tf.Variable(initial_value, name=var_name)

    def _base_init(self, shape, var_name):
        initial_value = tf.constant(0.0, shape=shape)
        return tf.Variable(initial_value, name=var_name)  

    def _get_feature_gradients(self, x):
        grads = []
        for i in range(self.n_hidden_layer3):
            grads.append(self.sess.run(self.grad_fun, feed_dict={self.gradIndex:i, self.state_ph:x}))

        return np.reshape(grads, (self.n_hidden_layer3,4))
        
    def total_Cntrl(self, state, ref_state, ref_signal):
        lin_cntrl = self.linear_Cntrl(state, ref_signal)
        adap_cntrl, dcntrl = self.mrac_Cntrl(state, ref_state)
        total_cntrl = lin_cntrl + adap_cntrl
        self.TOTAL_CNTRL_REC.append(total_cntrl)
        self.ADAP_CNTRL_REC.append(-adap_cntrl.T)
        self.DADAP_CNTRL_REC.append(-dcntrl)
        return total_cntrl

    def linear_Cntrl(self, state, ref_signal):
        fb = -self.kx @ state
        ff = np.reshape(self.kr @ ref_signal,(self.action_dim,1))
        cntrl = fb+ff
        return cntrl.T

    def mrac_Cntrl(self, state, ref_state):
        # Update the feature
        self.phi = self.updateBasis(state)
        #Get feature graidents
        self.dphi = self._get_feature_gradients(np.reshape(state, (1,self.state_dim))) 

        if self.lr_status:
            self.updateNetWeights(state, ref_state)
            self.NET_PARAM.append(np.reshape(self.netWeights, (2,self.n_hidden_layer3)))
            cntrl = np.dot(self.phi, self.netWeights)
            dcntrl = np.dot(self.dphi.transpose(), self.netWeights)
        else:
            cntrl = np.dot(self.phi, self.netWeights)
            dcntrl = np.dot(self.dphi.transpose(), self.netWeights)
        
        self.replay_buffer.add(state, cntrl)       
        # Update the Features
        if self.replay_buffer.size() % 100 == 0 and self.replay_buffer.size() > 0:
            self.updateDMRAC_NET()
        
        return cntrl, dcntrl
        
    def updateNetWeights(self, state, ref_state):
        error = state-ref_state
        temp = np.dot(np.dot(np.dot(self.phi.transpose(), np.transpose(error)), self.P), self.B)
        total_gain = np.linalg.inv(self.gain*np.eye(self.n_hidden_layer3) + 0.01*np.dot(self.dphi, self.dphi.transpose()))
        self.netWeights = self.netWeights + self.timeStep*(np.dot(total_gain,temp))

    def updateDMRAC_NET(self):
        N = self.batch_size
        for steps  in range(30):
            state, deltas = self.replay_buffer.sample_batch(N)
            self.sess.run(self.optimize, feed_dict={self.state_ph:np.reshape(state,[N, self.state_dim]), self.delta_ph:np.reshape(deltas,[N, self.action_dim])})

    def add_to_buffer(self, state, delta):
        self.replay_buffer.add(np.reshape(state,(self.state_dim,)), np.reshape(delta, (self.action_dim,)))
        