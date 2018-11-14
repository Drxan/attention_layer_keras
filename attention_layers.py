# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 10:33
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : my_layers.py
# @Software: PyCharm
import keras.backend as K
from keras.engine.topology import Layer
from keras.activations import softmax
import os
from keras.initializers import Ones, Zeros
from keras.layers import Wrapper, InputSpec, RNN
from keras.layers import Add
import keras.backend as K
from keras.engine.topology import Layer
from keras.initializers import orthogonal, random_normal
from keras.activations import softmax
from keras.layers import Concatenate

"""
主要实现了一些注意力层
"""

class LayerNormalization(Layer):
    """
        Implementation according to:
            "Layer Normalization" by JL Ba, JR Kiros, GE Hinton (2016)
    """

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self._g = self.add_weight(
            name='gain',
            shape=(input_shape[-1],),
            initializer=Ones(),
            trainable=True
        )
        self._b = self.add_weight(
            name='bias',
            shape=(input_shape[-1],),
            initializer=Zeros(),
            trainable=True
        )

    def call(self, x):
        mean = K.mean(x, axis=-1)
        std = K.std(x, axis=-1)

        if len(x.shape) == 3:
            mean = K.permute_dimensions(
                K.repeat(mean, x.shape.as_list()[-1]),
                [0, 2, 1]
            )
            std = K.permute_dimensions(
                K.repeat(std, x.shape.as_list()[-1]),
                [0, 2, 1]
            )

        elif len(x.shape) == 2:
            mean = K.reshape(
                K.repeat_elements(mean, x.shape.as_list()[-1], 0),
                (-1, x.shape.as_list()[-1])
            )
            std = K.reshape(
                K.repeat_elements(mean, x.shape.as_list()[-1], 0),
                (-1, x.shape.as_list()[-1])
            )

        return self._g * (x - mean) / (std + self._epsilon) + self._b


class AttentionRNNWrapper(Wrapper):
    """
        The idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.
        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.
        This technique is similar to the input-feeding method described in the paper cited
    """

    def __init__(self, layer, weight_initializer="glorot_uniform", **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer

        super(AttentionRNNWrapper, self).__init__(layer, **kwargs)

    def _validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Layer received an input with shape {0} but expected a Tensor of rank 3.".format(input_shape[0]))

    def build(self, input_shape):
        self._validate_input_shape(input_shape)

        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        input_dim = input_shape[-1]

        output_dim = self.layer.compute_output_shape(input_shape)[-1]

        self._W1 = self.add_weight(shape=(input_dim, input_dim), name="{}_W1".format(self.name),
                                   initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, input_dim), name="{}_W2".format(self.name),
                                   initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(2 * input_dim, input_dim), name="{}_W3".format(self.name),
                                   initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(input_dim,), name="{}_b2".format(self.name),
                                   initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(input_dim,), name="{}_b3".format(self.name),
                                   initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(input_dim, 1), name="{}_V".format(self.name),
                                  initializer=self.weight_initializer)

        super(AttentionRNNWrapper, self).build()

    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        return self.layer.compute_output_shape(input_shape)

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights

    def step(self, x, states):
        h = states[0]
        # states[1] necessary?

        # equals K.dot(X, self._W1) + self._b2 with X.shape=[bs, T, input_dim]
        total_x_prod = states[-1]
        # comes from the constants (equals the input sequence)
        X = states[-2]

        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_prod + hw
        attention = softmax(K.dot(additive_atn, self._V), axis=1)
        x_weighted = K.sum(attention * X, [1])

        x = K.dot(K.concatenate([x, x_weighted], 1), self._W3) + self._b3

        h, new_states = self.layer.cell.call(x, states[:-2])

        return h, new_states

    def call(self, x, constants=None, mask=None, initial_state=None):
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec.shape

        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]

            base_initial_state = self.layer.get_initial_state(x)
            if len(base_initial_state) != len(initial_states):
                raise ValueError(
                    "initial_state does not have the correct length. Received length {0} but expected {1}".format(
                        len(initial_states), len(base_initial_state)))
            else:
                # check the state' shape
                for i in range(len(initial_states)):
                    if not initial_states[i].shape.is_compatible_with(
                            base_initial_state[i].shape):  # initial_states[i][j] != base_initial_state[i][j]:
                        raise ValueError(
                            "initial_state does not match the default base state of the layer. Received {0} but expected {1}".format(
                                [x.shape for x in initial_states], [x.shape for x in base_initial_state]))
        else:
            initial_states = self.layer.get_initial_state(x)

        if not constants:
            constants = []

        constants += self.get_constants(x)

        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )

        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output

            # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, K.dot(x, self._W1) + self._b2]

        return constants

    def get_config(self):
        config = {'weight_initializer': self.weight_initializer}
        base_config = super(AttentionRNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionScaledDotProduct(Layer):
    """
    [Ref] Attention is all you need. https://arxiv.org/pdf/1706.03762.pdf
    注意力，对Query和Key求点积，并利用Query的维度k的平方根对点积进行缩放。Query、Key及value可以是同样的，也可以不一样
    """
    def __init__(self, scale=True, agg_mode=None, keepdims=False, **kwargs):
        self.scale=scale
        if agg_mode not in ['sum', 'mean', 'min', 'max', None]:
            raise ValueError('Invalid aggregate mode. '
                             'aggregate mode should be one of '
                             '{"sum", "mean", "min", "max", None}')
        self.agg_mode = agg_mode
        self.keepdims = keepdims
        super(AttentionScaledDotProduct, self).__init__(**kwargs)

    def call(self, inputs):
        """
        分别利用query中的每一步状态对齐key中各步状态，得到权重
        :param inputs: 列表， 按顺序存放query,key,value
        :return:  注意力对齐结果
        """
        querys = inputs[0]  # Querys
        keys = inputs[1]  # Keys
        values = inputs[2]  # Values
        weight_margins = K.batch_dot(querys, K.permute_dimensions(keys, [0, 2, 1]))
        if self.scale:
            k = K.int_shape(querys)[-1]
            weight_margins = weight_margins/np.sqrt(k)
        weights = softmax(weight_margins, axis=-1)
        outputs = K.batch_dot(weights, values)
        if self.agg_mode == 'max':
            outputs = K.max(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'min':
            outputs = K.min(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'mean':
            outputs = K.mean(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'sum':
            outputs = K.sum(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode is None:
            pass
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)

class MultiHeadAttention(Layer):
    """
    [Ref] Attention is all you need. https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, attention_blocks, dk=64, dv=64, dmodel=512, initializer="glorot_uniform", regularizer=None,
                 constraint=None, **kwargs):
        assert len(attention_blocks) >= 2
        self.attention_blocks = attention_blocks
        self.head_num = len(attention_blocks)
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(self.head_num):
            wq = self.add_weight(name="{0}_WQ{1}".format(self.name, i),
                                 shape=(self.dmodel, self.dk),
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)
            wk = self.add_weight(name="{0}_WK{1}".format(self.name, i),
                                 shape=(self.dmodel, self.dk),
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)
            wv = self.add_weight(name="{0}_WV{1}".format(self.name, i),
                                 shape=(self.dmodel, self.dv),
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)
            self.WQ.append(wq)
            self.WK.append(wk)
            self.WV.append(wv)

        self.wo = self.add_weight(name="{0}_WO".format(self.name),
                                  shape=(self.head_num*self.dv, self.dmodel),
                                  initializer=self.initializer,
                                  regularizer=self.regularizer,
                                  constraint=self.constraint)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        attention_out = []
        for i in range(self.head_num):
            query = K.dot(inputs[0], self.WQ[i])
            key = K.dot(inputs[1], self.WK[i])
            value = K.dot(inputs[2], self.WV[i])
            out_put = self.attention_blocks[i]([query, key, value])
            attention_out.append(out_put)
        concat_out = Concatenate()(attention_out)
        out_puts = K.dot(concat_out, self.wo)
        return out_puts

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)

class SelfAttentionWeight(Layer):
    """
       输入x为shape=[batch_size,time_step,feature_dim]的序列。构造一个外部权重向量(表示文本的潜在语义向量)W，shape=(1,feature_dim),利用W对齐
       x中的每一个时间步以得到各个时间步状态量的权重，利用权重对各个时间步求和或进行其他处理，得到最终输出。
    """
    def __init__(self, agg_mode='sum', W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None,
                 bias=True, keepdims=False, **kwargs):
        if agg_mode not in ['sum', 'mean', 'min', 'max', None]:
            raise ValueError('Invalid aggregate mode. '
                             'aggregate mode should be one of '
                             '{"sum", "mean", "min", "max", None}')
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.keepdims = keepdims
        self.agg_mode = agg_mode
        super(SelfAttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        self.step_dim = input_shape[-2]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = K.dot(x, K.reshape(self.W, (-1, 1)))
        eij = K.squeeze(eij, axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        outputs = x * a
        if self.agg_mode == 'max':
            outputs = K.max(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'min':
            outputs = K.min(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'mean':
            outputs = K.mean(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'sum':
            outputs = K.sum(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode is None:
            pass
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    
class AttentionConcat(Layer):

    def __init__(self, **kwargs):
        super(AttentionConcat, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 3

        dim = max(input_shape[0][-1], input_shape[1][-1])
        self.W0 = self.add_weight(name='weight0',
                                  shape=(input_shape[0][-1], dim),
                                  initializer=orthogonal(seed=9),
                                  trainable=True)

        self.W1 = self.add_weight(name='weight1',
                                  shape=(input_shape[1][-1], dim),
                                  initializer=orthogonal(seed=9),
                                  trainable=True)
        self.vc = self.add_weight(name='vc',
                                 shape=(dim, 1),
                                 initializer=random_normal(seed=9),
                                 trainable=True)

        super(AttentionConcat, self).build(input_shape)

    def call(self, inputs):
        q = inputs[0]
        p = inputs[1]

        q_shape = K.int_shape(q)
        prod_p = K.dot(p, self.W1)
        prod_q = K.dot(q, self.W0)
        prod_q_shape = K.int_shape(prod_q)
        p_time_step = K.int_shape(prod_p)[-2]
        prod_p = K.expand_dims(prod_p, axis=-2)
        prod_q = K.reshape(K.tile(prod_q, n=[1, p_time_step, 1]), shape=[-1, p_time_step, prod_q_shape[1], prod_q_shape[2]])
        concat = K.tanh(prod_q + prod_p)
        weights = softmax(K.squeeze(K.dot(concat, self.vc), axis=-1), axis=-2)
        weights = K.expand_dims(weights, axis=-1)
        q = K.reshape(K.tile(q, n=[1, p_time_step, 1]), shape=[-1, p_time_step, q_shape[1], q_shape[2]])
        attention_q = K.sum(q*weights, axis=-2)
        return attention_q

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionBilinear(Layer):

    def __init__(self, **kwargs):
        super(AttentionBilinear, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 3

        self.W = self.add_weight(name='bilinear_weight',
                                 shape=(input_shape[1][-1], input_shape[0][-1]),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)

        super(AttentionBilinear, self).build(input_shape)

    def call(self, inputs):
        q = inputs[0]
        p = inputs[1]

        q_shape = K.int_shape(q)
        p_time_step = K.int_shape(p)[-2]

        q = K.reshape(K.tile(q, n=[1, p_time_step, 1]), shape=[-1, p_time_step, q_shape[-2], q_shape[-1]])
        prod_p = K.expand_dims(K.dot(p, self.W), -2)
        prod = K.sum(q*prod_p, axis=-1)
        weights = softmax(prod, axis=-2)
        weights = K.expand_dims(weights, -1)
        attention_q = K.sum(q*weights, axis=-2)
        return attention_q

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionDot(Layer):

    def __init__(self, **kwargs):
        super(AttentionDot, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 3
        assert input_shape[0][-1] == input_shape[1][-1]

        m = input_shape[0][-1]
        n = input_shape[0][-1]
        self.W = self.add_weight(name='dot_weight',
                                 shape=(m, n),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)

        self.vd = self.add_weight(name='vd',
                                  shape=(n, 1),
                                  initializer=random_normal(seed=9),
                                  trainable=True)
        super(AttentionDot, self).build(input_shape)

    def call(self, inputs):
        q = inputs[0]
        p = inputs[1]

        q_shape = K.int_shape(q)
        p_time_step = K.int_shape(p)[-2]

        q = K.reshape(K.tile(q, n=[1, p_time_step, 1]), shape=[-1, p_time_step, q_shape[-2], q_shape[-1]])
        p = K.expand_dims(p, axis=-2)
        prod = q*p
        weights = K.dot(K.tanh(K.dot(prod, self.W)), self.vd)
        weights = K.expand_dims(softmax(K.squeeze(weights, axis=-1), axis=-2), axis=-1)
        attention_q = K.sum(q*weights, axis=-2)
        return attention_q

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionMinus(Layer):

    def __init__(self, **kwargs):
        super(AttentionMinus, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 3
        assert input_shape[0][-1] == input_shape[1][-1]

        m = input_shape[0][-1]
        n = input_shape[0][-1]
        self.W = self.add_weight(name='minus_weight',
                                 shape=(m, n),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)

        self.vm = self.add_weight(name='vd',
                                  shape=(n, 1),
                                  initializer=random_normal(seed=9),
                                  trainable=True)
        super(AttentionMinus, self).build(input_shape)

    def call(self, inputs):
        q = inputs[0]
        p = inputs[1]

        q_shape = K.int_shape(q)
        p_time_step = K.int_shape(p)[-2]

        q = K.reshape(K.tile(q, n=[1, p_time_step, 1]), shape=[-1, p_time_step, q_shape[-2], q_shape[-1]])
        p = K.expand_dims(p, axis=-2)
        minus = q-p
        weights = K.dot(K.tanh(K.dot(minus, self.W)), self.vm)
        weights = K.expand_dims(softmax(K.squeeze(weights, axis=-1), axis=-2), axis=-1)
        attention_q = K.sum(q*weights, axis=-2)
        return attention_q

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class InsideAggregation(Layer):

    def __init__(self, **kwargs):
        super(InsideAggregation, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.W = self.add_weight(name='inside_agg_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        super(InsideAggregation, self).build(input_shape)

    def call(self, inputs):
        gates = K.sigmoid(K.dot(inputs, self.W))
        outputs = inputs*gates
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class MixedAggregation(Layer):

    def __init__(self, **kwargs):
        super(MixedAggregation, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        m = input_shape[0][-1]
        n = input_shape[0][-1]
        self.W1 = self.add_weight(name='mixed_agg_weight1',
                                 shape=(m, n),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        self.W2 = self.add_weight(name='mixed_agg_weight2',
                                 shape=(n, n),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        self.vm = self.add_weight(name='vmix',
                                  shape=(1, n),
                                  initializer=random_normal(seed=9),
                                  trainable=True)
        super(MixedAggregation, self).build(input_shape)

    def call(self, inputs):
        x = [K.expand_dims(v, axis=-1) for v in inputs]
        x = K.concatenate(x, axis=-1)
        x = K.permute_dimensions(x, pattern=[0, 1, 3, 2])
        weights = K.tanh(K.dot(x, self.W1) + K.dot(self.vm, self.W2))
        weights = K.dot(weights, K.transpose(self.vm))
        weights = K.squeeze(weights, axis=-1)
        weights = K.expand_dims(softmax(weights, axis=-2), axis=-1)
        outputs = K.sum(x*weights, axis=-2)
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionPool(Layer):
    """
     对vecs=(T,m),T为序列长度，w为各个时间步的维度。利用自注意力对各个时间步的状态加权，在时间维度上求（max、mean、min）等。
     得到最终结果vec=(1, m)或vec=(T,m)。
    """
    def __init__(self, weight_initializer="glorot_uniform", agg_mode='sum', keepdims=True, **kwargs):
        if agg_mode not in ['sum', 'mean', 'min', 'max', None]:
            raise ValueError('Invalid aggregate mode. '
                             'aggregate mode should be one of '
                             '{"sum", "mean", "min", "max", None}')
        self.weight_initializer = weight_initializer
        self.agg_mode = agg_mode
        self.keepdims = keepdims
        super(AttentionPool, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        m = input_shape[-1]
        n = input_shape[-1]
        self.W1 = self.add_weight(name="{}_W1".format(self.name),
                                 shape=(m, n),
                                 initializer=self.weight_initializer,
                                 trainable=True)
        self.W2 = self.add_weight(name="{}_W2".format(self.name),
                                 shape=(n, n),
                                 initializer=self.weight_initializer,
                                 trainable=True)
        self.vm = self.add_weight(name="{}_v".format(self.name),
                                  shape=(1, n),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(AttentionPool, self).build(input_shape)

    def call(self, inputs):
        weights = K.tanh(K.dot(inputs, self.W1) + K.dot(self.vm, self.W2))
        weights = K.dot(weights, K.transpose(self.vm))
        weights = softmax(weights, axis=-2)
        outputs = inputs*weights
        if self.agg_mode == 'max':
            outputs = K.max(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'min':
            outputs = K.min(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'mean':
            outputs = K.mean(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'sum':
            outputs = K.sum(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode is None:
            pass
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionSelect(Layer):
    """
     利用reader=(1,m)对story=(T,m)进行注意力加权,得到story中各个item相对于reader的重要性。在时间维度上求（max、mean、min）等。
     得到最终结果result,其shape=(1, m)或shape=(T,m)。
    """
    def __init__(self, weight_initializer="glorot_uniform", agg_mode='sum', keepdims=False, **kwargs):
        if agg_mode not in ['sum', 'mean', 'min', 'max', None]:
            raise ValueError('Invalid aggregate mode. '
                             'aggregate mode should be one of '
                             '{"sum", "mean", "min", "max", None}')
        self.weight_initializer = weight_initializer
        self.agg_mode = agg_mode
        self.keepdims = keepdims
        super(AttentionSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        m1 = input_shape[0][-1]
        m2 = input_shape[1][-1]
        n = input_shape[0][-1]
        self.W1 = self.add_weight(name="{}_W1".format(self.name),
                                 shape=(m1, n),
                                 initializer=self.weight_initializer,
                                 trainable=True)
        self.W2 = self.add_weight(name="{}_W2".format(self.name),
                                 shape=(m2, n),
                                 initializer=self.weight_initializer,
                                 trainable=True)
        self.vt = self.add_weight(name="{}_v".format(self.name),
                                  shape=(n, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(AttentionSelect, self).build(input_shape)

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        weights = K.dot(K.tanh(K.dot(x1, self.W1) + K.dot(x2, self.W2)), self.vt)
        weights = softmax(weights, axis=-2)
        outputs = x1*weights
        if self.agg_mode == 'max':
            outputs = K.max(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'min':
            outputs = K.min(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'mean':
            outputs = K.mean(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode == 'sum':
            outputs = K.sum(outputs, axis=-2, keepdims=self.keepdims)
        elif self.agg_mode is None:
            pass
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class AttentionSelf(Layer):

    def __init__(self, **kwargs):
        super(AttentionSelf, self).__init__(**kwargs)
    '''
    def build(self, input_shape):
        assert len(input_shape) == 3
        m = input_shape[-1]
        n = input_shape[-1]

        self.W = self.add_weight(name='self_att_weight',
                                 shape=(m, n),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        super(AttentionSelf, self).build(input_shape)
    '''
    def call(self, inputs):
        weights = K.batch_dot(inputs, K.permute_dimensions(inputs, pattern=[0, 2, 1]))
        outputs = K.batch_dot(weights, inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class MatrixInteraction(Layer):
    """
      对两个矩阵M1,M2进行交互。初始化一个权重矩阵W,交互结果=M1 x W x M2，x表示矩阵乘法。
    """
    def __init__(self, **kwargs):
        super(MatrixInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 3
        m1 = input_shape[0][-1]
        m2 = input_shape[1][-1]
        self.W = self.add_weight(name='interaction_weight',
                                 shape=(m1, m2),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        super(MatrixInteraction, self).build(input_shape)

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        outputs = K.batch_dot(K.dot(x1, self.W), K.permute_dimensions(x2, pattern=[0, 2, 1]))
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


class MergeChannel(Layer):

    def __init__(self, **kwargs):
        super(MergeChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 4
        m1 = input_shape[-2]
        m2 = input_shape[-1]

        self.W = self.add_weight(name='merge_weight',
                                 shape=(m1, m2),
                                 initializer=orthogonal(seed=9),
                                 trainable=True)
        super(MergeChannel, self).build(input_shape)

    def call(self, inputs):
        outputs = inputs * self.W
        outputs = K.sum(outputs, axis=-1, keepdims=False)
        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            xs = [K.placeholder(shape=shape) for shape in input_shape]
            x = self.call(xs)
        else:
            x = K.placeholder(shape=input_shape)
            x = self.call(x)
        if isinstance(x, list):
            return [K.int_shape(x_elem) for x_elem in x]
        else:
            return K.int_shape(x)


def get_Multi_hop_vec(story, reader, attention_layer=None):
    if attention_layer is None:  # 默认使用点积加权求和注意力
        attention_layer = AttentionSelect()

    weighted_story_vec0 = attention_layer([story, reader])
    reader_vec1 = Add()([reader, weighted_story_vec0])

    weighted_story_vec1 = attention_layer([story, reader_vec1])
    reader_vec2 = Add()([reader_vec1, weighted_story_vec1])

    weighted_story_vec2 = attention_layer([story, reader_vec2])
    reader_vec3 = Add()([reader_vec2, weighted_story_vec2])

    weighted_story_vec3 = attention_layer([story, reader_vec3])
    reader_vec_final = Add()([reader_vec3, weighted_story_vec3])

    return reader_vec_final


def get_attention_vec(story, readers, attention_layer, merge=True):
    att_vecs = [attention_layer([story, reader]) for reader in readers]
    if merge:
        att_vecs = [Concatenate()([v1, v2]) for v1, v2 in zip(att_vecs, readers)]
    return att_vecs
