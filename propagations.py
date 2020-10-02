
import tensorflow as tf

class double:
    @tf.function
    def propagation(self, input):
        prob = ( tf.tanh(input) + 1. ) / 2.
        return (tf.keras.backend.random_binomial(prob.shape, prob, dtype=input.dtype)*2.)-1.
    
    def meanfield_propagation(self, input):
        return tf.tanh(input)