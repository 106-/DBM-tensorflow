
import tensorflow as tf
import copy

class Sampler:
    def __init__(self, dbm, datasize, initial_update=1000, update_time=1):
        self.dbm = dbm
        self.initial_update = initial_update
        self.inital_state = True
        self.update_time = update_time

        seed_generator = tf.random.experimental.Generator.from_non_deterministic_state()
        seeds = [seed_generator.make_seeds(2)[0] for _ in dbm.layers]

        self.values = [(tf.cast(tf.random.stateless_binomial(shape=(datasize, r), seed=seeds[i], counts=1.0, probs=0.5), dtype=dbm.dtype)*2.)-1. for i, r in enumerate(dbm.layers)]
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
                signal = self.dbm.signal(self.values[1], -1)
                self.values[0] = self.propagation(signal)
            else:
                self.values[0] = fixed_visible

            for l in range(1, len(self.dbm.layers)-1):
                signal = self.dbm.signal(self.values[l-1], l) + self.dbm.signal(self.values[l+1], -(l+1))
                self.values[l] = self.propagation(signal)

            signal = self.dbm.signal(self.values[-2], len(self.dbm.weights))
            self.values[-1] = self.propagation(signal)

            # 復路
            for l in reversed(range(1, len(self.dbm.layers)-1)):
                signal = self.dbm.signal(self.values[l-1], l) + self.dbm.signal(self.values[l+1], -(l+1))
                self.values[l] = self.propagation(signal)
        
        if fixed_visible is None:
            signal = self.dbm.signal(self.values[1], -1)
            self.values[0] = self.propagation(signal)

        return copy.deepcopy(self.values)

def oneshot_sampling(dbm, datasize, update_time=1000):
    sampler = Sampler(dbm, datasize, initial_update=update_time)
    return sampler.sampling()

class MeanfieldSampler(Sampler):
    def __init__(self, dbm, datasize, initial_update=1000, update_time=1):
        super().__init__(dbm, datasize, initial_update, update_time)
        self.propagation = getattr(dbm.propagation, "meanfield_propagation")
        self.values = [tf.random.uniform((datasize, r), minval=-1., maxval=1., dtype=self.dbm.dtype) for r in dbm.layers]