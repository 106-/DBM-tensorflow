
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
    def __init__(self, dbm, initial_update=100, update_time=1):
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
        self.single_marginalize = self.dbm.propagation.single_marginalize
        self.mariginalize = self.dbm.propagation.first_smci_marginalize
    
    def expectation(self, data, batch_idx):

        if self.sampler is None:
            self.sampler = Sampler(self.dbm, len(data), self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)

        signals = [None for i in self.dbm.layers]
        for i in range(1, len(self.dbm.layers)-1):
            signals[i] = self.dbm.signal(values[i-1], i) + self.dbm.signal(values[i+1], -(i+1))
        signals[-1] = self.dbm.signal(values[-2], len(self.dbm.weights))

        multiply_up = [None for i in self.dbm.weights]
        multiply_down = [None for i in self.dbm.weights]
        for i in range(1, len(self.dbm.weights)):
            multiply_up[i] = values[i][:, :, tf.newaxis] * self.dbm.weights[i]
            multiply_down[i] = values[i+1][:, tf.newaxis, :] * self.dbm.weights[i]
        
        expectations = [None for i in self.dbm.weights]
        for i,_ in enumerate(self.dbm.weights):
            if i==0:
                expectations[i] = tf.reduce_mean( values[i][:, :, tf.newaxis] * self.single_marginalize(signals[i+1])[:, tf.newaxis, :], axis=0 )
            else:
                expectations[i] = self.mariginalize( signals[i][:, :, tf.newaxis]-multiply_down[i], signals[i+1][:, tf.newaxis, :]-multiply_up[i], self.dbm.weights[i])

        return expectations

class four_layer_second_smci:
    def __init__(self, dbm, sample_size=500, initial_update=1000, update_time=1):
        if len(dbm.layers) != 4:
            raise ValueError("2-SMCI supports only 4-layered DBM.")

        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time
        self.sample_size = sample_size
        self.single_marginalize = self.dbm.propagation.single_marginalize
        self.mariginalize = self.dbm.propagation.first_smci_marginalize
        self.mariginalize2 = self.dbm.propagation.second_smci_marginalize
    
    def expectation(self, data, batch_idx):

        if self.sampler is None:
            self.sampler = Sampler(self.dbm, len(data), self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)
        
        expectations = [None for i in self.dbm.weights]

        signals = [None for i in self.dbm.layers]
        signals[0] = self.dbm.signal(values[1], -1)
        for i in range(1, len(self.dbm.layers)-1):
            signals[i] = self.dbm.signal(values[i-1], i) + self.dbm.signal(values[i+1], -(i+1))
        signals[-1] = self.dbm.signal(values[-2], len(self.dbm.weights))

        multiply_up = [None for i in self.dbm.weights]
        multiply_down = [None for i in self.dbm.weights]
        for i,_ in enumerate(self.dbm.weights):
            multiply_up[i] = values[i][:, :, tf.newaxis] * self.dbm.weights[i]
            multiply_down[i] = values[i+1][:, tf.newaxis, :] * self.dbm.weights[i]

        # expectation[0]
        x = (self.dbm.signal(values[0], 1)[:, tf.newaxis, :]) + self.mariginalize2(signals[2][:, tf.newaxis, :] - multiply_up[1], self.dbm.weights[1], axis=2)[:, tf.newaxis, :]
        expectations[0] = tf.reduce_mean( values[0][:, :, tf.newaxis] * self.single_marginalize( x ), axis=0)

        # expectation[1]
        x = (self.dbm.signal(values[2],-2)[:, :, tf.newaxis] - multiply_down[1]) + self.dbm.signal(values[0], 1)[:, :, tf.newaxis]
        y = (self.dbm.signal(values[1], 2)[:, tf.newaxis, :] -   multiply_up[1]) + self.mariginalize2( signals[3][:, tf.newaxis, :] -   multiply_up[2], self.dbm.weights[2], axis=2)[:, tf.newaxis, :]
        z = self.dbm.weights[1][tf.newaxis, :, :]
        expectations[1] = self.mariginalize(x, y, z)

        # expectation[2]
        x = signals[3][:, tf.newaxis, :] - multiply_up[2]
        y = (self.dbm.signal(values[3],-3)[:, :, tf.newaxis] - multiply_down[2]) + self.mariginalize2(signals[1][:, :, tf.newaxis] - multiply_down[1], self.dbm.weights[1], axis=1)[:, :, tf.newaxis]
        z = self.dbm.weights[2][tf.newaxis, :, :]
        expectations[2] = self.mariginalize(x, y, z)

        return expectations

class three_layer_second_smci:
    def __init__(self, dbm, sample_size=500, initial_update=1000, update_time=1):
        if len(dbm.layers) != 3:
            raise ValueError("2-SMCI supports only 3-layered DBM.")

        self.sampler = None
        self.dbm = dbm
        self.initial_update = initial_update
        self.update_time = update_time
        self.sample_size = sample_size
        self.single_marginalize = self.dbm.propagation.single_marginalize
        self.mariginalize = self.dbm.propagation.first_smci_marginalize
        self.mariginalize2 = self.dbm.propagation.second_smci_marginalize
    
    def expectation(self, data, batch_idx):

        if self.sampler is None:
            self.sampler = Sampler(self.dbm, len(data), self.initial_update, self.update_time)
        
        values = self.sampler.sampling(data)
        for i,_ in enumerate(values):
            values[i] = tf.gather(values[i], batch_idx)
        
        expectations = [None for i in self.dbm.weights]
        signal_up = [None for i in self.dbm.weights]
        signal_down = [None for i in self.dbm.weights]
        multiply_up = [None for i in self.dbm.weights]
        multiply_down = [None for i in self.dbm.weights]
        for i,_ in enumerate(self.dbm.weights):
            signal_up[i] = self.dbm.signal(values[i], i+1)
            signal_down[i] = self.dbm.signal(values[i+1], -(i+1))
            multiply_up[i] = signal_up[i][:, tf.newaxis, :] - values[i][:, :, tf.newaxis] * self.dbm.weights[i]
            multiply_down[i] = signal_down[i][:, :, tf.newaxis] - values[i+1][:, tf.newaxis, :] * self.dbm.weights[i]
        
        # expectation[0]
        x = (signal_up[0] + self.mariginalize2(multiply_up[1], self.dbm.weights[1], axis=2))[:, tf.newaxis, :]
        expectations[0] = tf.reduce_mean( values[0][:, :, tf.newaxis] * self.single_marginalize( x ), axis=0)

        # expectation[1]
        x = multiply_down[1] + signal_up[0][:, :, tf.newaxis]
        y = multiply_up[1]
        z = self.dbm.weights[1][tf.newaxis, :, :]
        expectations[1] = self.mariginalize(x, y, z)

        return expectations