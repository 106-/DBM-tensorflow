
import tensorflow as tf

class Sampler:
    def __init__(self, dbm, datasize, initial_update=1000, update_time=1):
        self.dbm = dbm
        self.initial_update = initial_update
        self.inital_state = True
        self.update_time = update_time

        self.values = [(tf.keras.backend.random_binomial((datasize, r), p=0.5, dtype=dbm.dtype)*2.)-1. for r in dbm.layers]
        self.propagation = getattr(dbm.propagation, "propagation")

    def sampling(self, fixed_visible=None):
        if self.inital_state:
            approximition_time = self.initial_update
            self.inital_state = False
        else:
            approximition_time = self.update_time

        for t in range(approximition_time):
            # 往路
            if fixed_visible is None:
                signal = tf.transpose( tf.matmul(self.dbm.weights[0], tf.transpose(self.values[1])) )
                self.values[0] = self.propagation(signal)

            else:
                self.values[0] = fixed_visible

            for l in range(1, len(self.dbm.layers)-1):
                signal = tf.matmul(self.values[l-1], self.dbm.weights[l-1]) + tf.transpose( tf.matmul(self.dbm.weights[l], tf.transpose(self.values[l+1])) )
                self.values[l] = self.propagation(signal)

            signal = tf.matmul( self.values[-2], self.dbm.weights[-1] )
            self.values[-1] = self.propagation(signal)

            # 復路
            for l in reversed(range(1, len(self.dbm.layers)-1)):
                signal = tf.matmul(self.values[l-1], self.dbm.weights[l-1]) + tf.transpose( tf.matmul(self.dbm.weights[l], tf.transpose(self.values[l+1])) )
                self.values[l] = self.propagation(signal)
        
        if fixed_visible is None:
            signal = tf.transpose( tf.matmul(self.dbm.weights[0], tf.transpose(self.values[1])) )
            self.values[0] = self.propagation(signal)

        return self.values

def oneshot_sampling(dbm, datasize, update_time=1000):
    sampler = Sampler(dbm, datasize, initial_update=update_time)
    return sampler.sampling()

class MeanfieldSampler(Sampler):
    def __init__(self, dbm, datasize, initial_update=1000, update_time=1):
        super().__init__(dbm, datasize, initial_update, update_time)
        self.propagation = getattr(dbm.propagation, "meanfield_propagation")
        self.values = [tf.random.uniform((datasize, r), minval=-1., maxval=1., dtype=self.dbm.dtype) for r in dbm.layers]