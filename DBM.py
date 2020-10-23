
import numpy as np
import tensorflow as tf
import propagations
import data_expectations
import model_expectations
import json
from numpy.lib.stride_tricks import as_strided

class DBM:
    def __init__(self, layers, propagation="double", data_expectation="exact", model_expectation="exact", dtype="float32", gauss_init=False):
        self.layers = np.array(layers)
        # [1, 2, 3, 4] -> [[1, 2], [2, 3], [3, 4]]
        self.layers_matrix_sizes = as_strided(self.layers, (len(self.layers)-1, 2), (self.layers.strides[0], self.layers.strides[0]))

        self.dtype = dtype
        self.weights = []
        
        if gauss_init:
            for n,(i,j) in enumerate(self.layers_matrix_sizes):
                name = "weight{}({}x{})".format(n+1, i, j)
                sigma = np.sqrt( 10.0 / (i+j) ) # Maeno's initialization
                self.weights.append( tf.Variable( tf.random.normal((i,j), stddev=sigma, dtype=self.dtype), name=name ) )
        else:
            for n,(i,j) in enumerate(self.layers_matrix_sizes):
                name = "weight{}({}x{})".format(n+1, i, j)
                self.weights.append( tf.Variable( tf.keras.initializers.GlorotUniform()((i, j), dtype=self.dtype), name=name ) )

        self.propagation = getattr(propagations, propagation)()
        self._propagation = propagation

        self.data_expectation = getattr(data_expectations, data_expectation)(self)
        self._data_expectation = data_expectation
        self.model_expectation = getattr(model_expectations, model_expectation)(self)
        self._model_expectation = model_expectation

    def weight_matrix(self, data):
        if len(data) != len(self.layers):
            raise ValueError("data length is not equal with layer.")
        weights = [None for i in self.weights]
        for i,_ in enumerate(weights):
            weights[i] = tf.matmul(tf.transpose(data[i]), data[i+1]) / len(data[i])
        return weights
    
    def signal(self, value, layer_number):
        if layer_number > 0:
            w = layer_number - 1
            if value.shape[1] != self.weights[w].shape[0]:
                raise ValueError("columb of value is not valid (expect {}, but got {})".format(self.weights[w].shape[0], value.shape[1]))
            return tf.matmul( value, self.weights[w] )
        elif layer_number < 0:
            w = abs(layer_number) - 1
            if value.shape[1] != self.weights[w].shape[1]:
                raise ValueError("columb of value is not valid (expect {}, but got {})".format(self.weights[w].shape[1], value.shape[1]))
            return tf.matmul( value, tf.transpose( self.weights[w] ) )
        else:
            raise ValueError("layer_number must not be zero.")

    # P(v, h1, h2) or P(h1, h2 | v)
    def probability(self, visible=None, data=None):
        datas = self.get_bit()
        
        def calc_energy():
            energies = [None for i in self.weights]
            for i,_ in enumerate(energies):
                energies[i] = tf.matmul( tf.matmul(datas[i], self.weights[i]), tf.transpose( datas[i+1] ) )
        
            energy_all = energies[0]
            for i in range(len(energies)-1):
                energy_all = tf.expand_dims(energy_all, len(energy_all.shape)) + energies[i+1]
            
            return energy_all

        # P(v, h1, h2)
        if visible is None:
            energy_all = calc_energy()
            energy_max = tf.reduce_max(energy_all)
            energy_exp = tf.exp( energy_all - energy_max )
            state_sum = tf.reduce_sum(energy_exp)

            # P(V, h1, h2), V is given data.
            if not data is None:
                datas[0] = data
                energy_all = calc_energy()
                energy_exp = tf.exp( energy_all - energy_max )

        # P(h1, h2 | v)
        else:
            datas[0] = visible
            energy_all = calc_energy()
            sum_axis = list(range(1, len(self.layers)))
            energy_exp = tf.exp( energy_all - tf.reduce_max(energy_all, axis=sum_axis, keepdims=True) )
            state_sum = tf.reduce_sum(energy_exp, axis=sum_axis, keepdims=True)

        return energy_exp / state_sum
    
    def log_likelihood(self, data):
        sum_axis = list(range(1, len(self.layers)))
        logprobs = tf.math.log( tf.reduce_sum(self.probability(data=data), axis=sum_axis) )
        return tf.reduce_mean(logprobs)
    
    @tf.function
    def kl_divergence(self, gen_dbm):
        sum_axis = list(range(1, len(self.layers)))
        probs = tf.reduce_sum(self.probability(), axis=sum_axis)
        sum_axis = list(range(1, len(gen_dbm.layers)))
        gen_probs = tf.reduce_sum(gen_dbm.probability(), axis=sum_axis)
        return tf.reduce_sum( gen_probs * tf.math.log( gen_probs / probs ) )

    def train(self, train_epoch, minibtach_size, optimizer, train_data, gen_dbm, learninglog):
        dataset_idx = tf.data.Dataset.from_tensor_slices( (np.arange(len(train_data))) )
        
        def per_epoch(epoch):
            kld = self.kl_divergence(gen_dbm)
            ll = self.log_likelihood(train_data)
            template = "[ {} / {} ] KL-Divergence: {}, log-likelihood: {}"
            print(template.format(epoch, train_epoch, float(kld), float(ll)))
            
            learninglog.make_log(epoch, "kl-divergence", float(kld))
            learninglog.make_log(epoch, "loglikelihood", float(ll))
        
        per_epoch(0)

        for epoch in range(train_epoch):
            for idx in dataset_idx.shuffle(len(train_data)).batch(minibtach_size):
                data_exp = self.data_expectation.expectation(train_data, idx)
                model_exp = self.model_expectation.expectation()

                grads = []
                for i,j in zip(data_exp, model_exp):
                    grads.append(-(i-j)) # make grad negative (stochastic gradient *ascent*)
                
                optimizer.apply_gradients(zip(grads, self.weights))

            per_epoch(epoch+1)

    def get_bit(self):
        length = np.max(self.layers)
        x = np.expand_dims(np.arange(2**length), 1)
        y = np.expand_dims(2**np.arange(length), 0)
        bits = np.where((x & y)==0, -1, 1)

        datas = [None for i in self.layers]
        for i,_ in enumerate(datas):
            l = self.layers[i]
            datas[i] = tf.convert_to_tensor( bits[:2**l, :l], dtype=self.dtype )
        return datas
    
    def save(self, filename):
        obj_names = ["_data_expectation", "_model_expectation", "_propagation"]
        data = {}

        data["layers"] = self.layers.tolist()
        data["dtype"] = self.dtype
        for i in obj_names:
            data[i[1:]] = getattr(self, i)
        
        data["params"] = {}
        for i,j in enumerate(self.weights):
            key = "w%d"%(i+1)
            data["params"][key] = j.numpy().tolist()
        
        json.dump(data, open(filename, "w+"), indent=2)
    
    @staticmethod
    def load(filename):
        data = json.load(open(filename, "r"))
        dbm = DBM(data["layers"], data["propagation"], data["data_expectation"], data["model_expectation"], data["dtype"])

        for i,j in enumerate(dbm.weights):
            key = "w%d"%(i+1)
            dbm.weights[i] = tf.Variable( data["params"][key], dtype=data["dtype"], name=key )
        
        return dbm
