import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def gaussian_likelihood(x, mu, log_std, mode = 'SAC'):
    
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 
    if mode == 'SAC:  
        pre_sum += 2*log_std + np.log(2*np.pi))
    elif mode == 'TQC':
        pre_sum -= 2 * np.log(2) + tf.math.log_sigmoid(2 * pi) + tf.math.log_sigmoid(-2 * pi)
    
    return tf.reduce_sum(pre_sum, axis=1)


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def mlp_gaussian_policy(x, act_dim, hidden, layers, mode='SAC'):
    net = nn(x, [hidden] * (layers+1))
    mu = tf.layers.dense(net, act_dim, activation=None)

    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    #if training in TQC
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std #r sample, pretanh
    if mode == 'TQC':
        logp_pi = gaussian_likelihood(pi, mu, log_std, mode = 'TQC')
    else: 
        logp_pi = gaussian_likelihood(pi, mu, log_std) # log prob + the sum line 
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, sac, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.z_tf = inputs_tf['z']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        z = self.z_tf

        input_pi = tf.concat(axis=1, values=[o, z, g])  # for actor

        # policy net
        if sac:
            with tf.variable_scope('pi'):
                mu, pi, logp_pi = mlp_gaussian_policy(input_pi, self.dimu, self.hidden, self.layers)
                mu, pi, self.logp_pi_tf = apply_squashing_func(mu, pi, logp_pi)
                # make sure actions are in correct range
                self.mu_tf = mu * self.max_u
                self.pi_tf = pi * self.max_u
                self.neg_logp_pi_tf = - self.logp_pi_tf
        if sac == 'TQC': 
            with tf.variable_scope('pi'):
                mu, pretanh, logp_pi = mlp_gaussian_policy(input_pi, self.dimu, self.hidden, self.layers, mode = 'TQC')
                mu, pi, self.logp_pi_tf = apply_squashing_func(mu, pretanh, logp_pi)
                # make sure actions are in correct range
                self.mu_tf = mu * self.max_u
                self.pi_tf = pi * self.max_u
                self.neg_logp_pi_tf = - self.logp_pi_tf
            
        elif sac == 'DDPG' or sac == 'TD3':
        #else: # ddpg
            with tf.variable_scope('pi'):
                self.pi_tf = self.max_u * tf.tanh(nn(
                    input_pi, [self.hidden] * self.layers + [self.dimu]))

        # Q value net
        if sac == 'TQC': 
             with tf.variable_scope('Q'): 
                # for policy training
                input_Q = tf.concat(axis=1, values=[o, z, g, self.pi_tf / self.max_u])
                nets = [nn(input_Q, [self.hidden] * self.layers + [1], name = str(i)) for i in range(n_nets)]
                quantiles = tf.stack(tuple(nets), axis=1)
                self.Q_pi_tf = quantiles
                
                input_Q = tf.concat(axis=1, values=[o, z, g, self.u_tf / self.max_u])
                nets = [nn(input_Q, [self.hidden] * self.layers + [1], reuse=True, name = str(i)) for i in range(n_nets)]
                quantiles = tf.stack(tuple(nets), axis=1)
                self.Q_tf = quantiles
        if sac == 'TD3':
            input_Q = tf.concat(axis=1, values=[o, z, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], name= str(1))
            self.Q_pi_tf_1 = nn(input_Q, [self.hidden] * self.layers + [1], name= str(2))
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, z, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True, name=str(2))
            self.Q_tf_1 = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True, name=str(2))
        else: 
            with tf.variable_scope('Q'):
                # for policy training
                input_Q = tf.concat(axis=1, values=[o, z, g, self.pi_tf / self.max_u])
                self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
                # for critic training
                input_Q = tf.concat(axis=1, values=[o, z, g, self.u_tf / self.max_u])
                self._input_Q = input_Q  # exposed for tests
                self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
