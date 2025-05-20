
import tensorflow as tf

class double:
    @tf.function
    def propagation(self, input):
        prob = (tf.tanh(input) + 1.) / 2.
        rand = tf.random.uniform(prob.shape, dtype=prob.dtype)
        sample = tf.where(rand < prob, tf.ones_like(prob), tf.zeros_like(prob))
        return sample * 2. - 1.
    
    def meanfield_propagation(self, input):
        return tf.tanh(input)
    
    @tf.function
    def single_marginalize(self, x):
        return tf.tanh(x)

    @tf.function
    def first_smci_marginalize(self, x, y, z):
        return tf.reduce_mean(tf.math.tanh( tf.math.atanh( tf.math.tanh(x)*tf.math.tanh(y) ) + z), axis=0)
    
    @tf.function
    def second_smci_marginalize(self, signal, weight, axis=-1):
        return tf.reduce_sum( tf.math.atanh( tf.math.tanh(signal) * tf.math.tanh(weight) ), axis=axis)