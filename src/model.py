import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Input, Flatten, Concatenate, RepeatVector
from tensorflow.keras import Model

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class T2V(tf.keras.layers.Layer):
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.output_dim), initializer='uniform', trainable=True)
        self.P = self.add_weight(name='P', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        self.p = self.add_weight(name='p', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w * x + self.p
        sin_trans = tf.sin(tf.linalg.matmul(x, self.W) + self.P)
        return tf.concat([sin_trans, original], -1)

def attention_model(EMR_N):
    n_hidden = 20
    input_train = Input(shape=(7, 3))
    output_train = Input((1,))
    EMR_input = Input(shape=(EMR_N,))
    
    x = T2V(16)(input_train)
    encoder_last_h, encoder_last_c = GRU(n_hidden, activation='tanh', return_state=True, return_sequences=False)(x)
    encoder_stack_h = GRU(n_hidden, activation='tanh', return_sequences=True)(x)
    
    encoder_last_h = BatchNormalization()(encoder_last_h)
    decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
    decoder_stack_h = GRU(n_hidden, activation='tanh', return_sequences=True)(decoder_input)

    attention = BahdanauAttention(8)
    context_vector, attention_weights = attention(encoder_stack_h, decoder_stack_h)
    context = BatchNormalization()(context_vector)

    decoder_combined_context = tf.keras.layers.dot([decoder_stack_h, context], axes=[2, 2])
    hidden = Flatten()(decoder_combined_context)
    concatted = Concatenate()([EMR_input, hidden])
    out_clas = Dense(3, activation='softmax')(concatted)

    model2 = Model(inputs=[input_train, EMR_input], outputs=out_clas)
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model2
