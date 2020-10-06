
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

class three_layer_1smci:
    def __init__(self, dbm, sample_size=500, initial_update=1000, update_time=1):
        if len(dbm.layers) != 3:
            raise ValueError("this DBM is not 3-layer.")

        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time
        self.sample_size = sample_size
    
    def expectation(self, data, batch_idx):
        if self.sampler is None:
            self.sampler = Sampler(self.dbm, self.sample_size, self.initial_update, self.update_time)
        
        tantan = lambda x,y,z: tf.math.tanh(tf.math.atanh(tf.math.tanh(x)*tf.math.tanh(y))+z)
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)
        expectations = [None for i in self.dbm.weights]

        lower_signal = self.dbm.signal(values[1], -1)
        middle_signal = self.dbm.signal(values[0], 1) + self.dbm.signal(values[2], -2)
        upper_signal = self.dbm.signal(values[1], 2)

        mid_to_up  = values[1][:, :, tf.newaxis] * self.dbm.weights[1]
        up_to_mid  = values[2][:, tf.newaxis, :] * self.dbm.weights[1]

        expectations[0] = tf.reduce_mean( values[0][:, :, tf.newaxis] * tf.math.tanh(middle_signal)[:, tf.newaxis, :], axis=0 )
        expectations[1] = tf.reduce_mean(tantan( middle_signal[:, :, tf.newaxis] - up_to_mid, upper_signal[:, tf.newaxis, :] - mid_to_up, self.dbm.weights[1] ), axis=0)

        return expectations

class four_layer_1smci:
    def __init__(self, dbm, sample_size=500, initial_update=1000, update_time=1):
        if len(dbm.layers) != 4:
            raise ValueError("this DBM is not 4-layer.")

        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time
        self.sample_size = sample_size
    
    def expectation(self, data, batch_idx):
        if self.sampler is None:
            self.sampler = Sampler(self.dbm, self.sample_size, self.initial_update, self.update_time)
        
        tantan = lambda x,y,z: tf.math.tanh(tf.math.atanh(tf.math.tanh(x)*tf.math.tanh(y))+z)
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)
        expectations = [None for i in self.dbm.weights]

        lower_signal = self.dbm.signal(values[1], -1)
        middle1_signal = self.dbm.signal(values[0], 1) + self.dbm.signal(values[2], -2)
        middle2_signal = self.dbm.signal(values[1], 2) + self.dbm.signal(values[3], -3)
        upper_signal = self.dbm.signal(values[2], 3)

        mid1_to_mid2 = values[1][:, :, tf.newaxis] * self.dbm.weights[1]
        mid2_to_up   = values[2][:, :, tf.newaxis] * self.dbm.weights[2]

        mid2_to_mid1 = values[2][:, tf.newaxis, :] * self.dbm.weights[1]
        up_to_mid2   = values[3][:, tf.newaxis, :] * self.dbm.weights[2]

        expectations[0] = tf.reduce_mean( values[0][:, :, tf.newaxis] * tf.math.tanh(middle1_signal)[:, tf.newaxis, :], axis=0 )
        expectations[1] = tf.reduce_mean(tantan( middle1_signal[:, :, tf.newaxis] - mid2_to_mid1, middle2_signal[:, tf.newaxis, :] - mid1_to_mid2, self.dbm.weights[1]), axis=0)
        expectations[2] = tf.reduce_mean(tantan( middle2_signal[:, :, tf.newaxis] - up_to_mid2,     upper_signal[:, tf.newaxis, :] -   mid2_to_up, self.dbm.weights[2]), axis=0)

        return expectations