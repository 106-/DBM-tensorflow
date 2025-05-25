
import tensorflow as tf

class double:
    def __init__(self):
        # 乱数生成器の作成
        self.seed_generator = tf.random.experimental.Generator.from_non_deterministic_state()
        # シード値を固定する場合は以下のように設定
        # self.seed_generator = tf.random.experimental.Generator.from_seed(5678)

    @tf.function
    def propagation(self, input):
        prob = ( tf.tanh(input) + 1. ) / 2.
        seed = self.seed_generator.make_seeds(2)[0]
        return (tf.cast(tf.random.stateless_binomial(shape=tf.shape(prob), seed=seed, counts=1.0, probs=prob), dtype=input.dtype)*2.)-1.
    
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