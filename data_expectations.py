
import tensorflow as tf
import numpy as np
from sampling import Sampler, MeanfieldSampler

class exact:
    def __init__(self, dbm):
        self.dbm = dbm

    def expectation(self, data, batch_idx):
        mdata = tf.gather(data, batch_idx)
        expectations = [None for i in self.dbm.weights]
        bits = self.dbm.get_bit()
        probability = self.dbm.probability(mdata)
        prob_shape_diag = tf.linalg.diag(probability.shape)
        prob_shape_diag = tf.where(prob_shape_diag==0, 1, prob_shape_diag) 
        probability = tf.reshape(probability, probability.shape + (1,1) )

        for i,_ in enumerate(expectations):
            layer_diag = tf.linalg.diag(self.dbm.layers_matrix_sizes[i])
            layer_diag = tf.where(layer_diag==0, 1, layer_diag)
            shape_a = np.hstack((prob_shape_diag[i], layer_diag[0]))
            shape_b = np.hstack((prob_shape_diag[i+1], layer_diag[1]))
            sum_axis = list(range(len(self.dbm.layers)))

            if i==0:
                a = tf.reshape(mdata, shape_a)
            else:
                a = tf.reshape(bits[i], shape_a)
            
            b = tf.reshape(bits[i+1], shape_b)

            expectations[i] = tf.reduce_sum( probability * a * b, axis=sum_axis ) / len(batch_idx)
        
        return expectations

class meanfield:
    def __init__(self, dbm, initial_update=1000, update_time=1):
        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time

    def expectation(self, data, batch_idx):
        datasize = len(data)

        if self.sampler is None:
            self.sampler = MeanfieldSampler(self.dbm, datasize, self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        batched_values = [None for i in values]

        for i,_ in enumerate(values):
            batched_values[i] = tf.gather(values[i], batch_idx)

        weight = self.dbm.weight_matrix(batched_values)

        return weight

class first_smci:
    def __init__(self, dbm, sample_size=500, initial_update=1000, update_time=1):
        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time
        self.sample_size = sample_size
        self.mariginalize = self.dbm.propagation.first_smci_marginalize
    
    def expectation(self, data, batch_idx):
        if self.sampler is None:
            self.sampler = Sampler(self.dbm, self.sample_size, self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)

        signals = [None for i in self.dbm.layers]
        for i in range(1, len(self.dbm.layers)-1):
            signals[i] = self.dbm.signal(values[i-1], 1) + self.dbm.signal(values[i+1], -(i+1))
        signals[-1] = self.dbm.signal(values[-2], len(self.dbm.weights))

        multiply_up = [None for i in self.dbm.weights]
        multiply_down = [None for i in self.dbm.weights]
        for i in range(1, len(self.dbm.weights)):
            multiply_up[i] = values[i][:, :, tf.newaxis] * self.dbm.weights[i]
            multiply_down[i] = values[i+1][:, tf.newaxis, :] * self.dbm.weights[i]
        
        expectations = [None for i in self.dbm.weights]
        for i,_ in enumerate(self.dbm.weights):
            if i==0:
                expectations[i] = tf.reduce_mean( values[i][:, :, tf.newaxis] * tf.math.tanh(signals[i+1])[:, tf.newaxis, :], axis=0 )
            else:
                expectations[i] = tf.reduce_mean(self.mariginalize( signals[i][:, :, tf.newaxis]-multiply_down[i], signals[i+1][:, tf.newaxis, :]-multiply_up[i], self.dbm.weights[i]), axis=0)

        return expectations