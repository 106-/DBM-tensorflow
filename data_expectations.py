
import tensorflow as tf
import numpy as np
from sampling import Sampler, MeanfieldSampler

class exact:
    def __init__(self, dbm):
        pass

    def expectation(self, dbm, data, batch_idx):
        mdata = tf.gather(data, batch_idx)
        expectations = [None for i in dbm.weights]
        bits = dbm.get_bit()
        probability = dbm.probability(mdata)
        prob_shape_diag = tf.linalg.diag(probability.shape)
        prob_shape_diag = tf.where(prob_shape_diag==0, 1, prob_shape_diag) 
        probability = tf.reshape(probability, probability.shape + (1,1) )

        for i,_ in enumerate(expectations):
            layer_diag = tf.linalg.diag(dbm.layers_matrix_sizes[i])
            layer_diag = tf.where(layer_diag==0, 1, layer_diag)
            shape_a = np.hstack((prob_shape_diag[i], layer_diag[0]))
            shape_b = np.hstack((prob_shape_diag[i+1], layer_diag[1]))
            sum_axis = list(range(len(dbm.layers)))

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

    def expectation(self, dbm, data, batch_idx):
        datasize = len(data)

        if self.sampler is None:
            self.sampler = MeanfieldSampler(self.dbm, datasize, self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        batched_values = [None for i in values]

        for i,_ in enumerate(values):
            batched_values[i] = tf.gather(values[i], batch_idx)

        weight = self.dbm.weight_matrix(batched_values)

        return weight