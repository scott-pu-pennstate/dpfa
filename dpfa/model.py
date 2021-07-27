import time
import os

from official.nlp.modeling import layers
from official.nlp.transformer.model_utils import get_padding_bias, get_decoder_self_attention_bias
import tensorflow as tf

__last_modified__ = time.time() - os.path.getmtime(__file__)


# this is the basic version of DPFA
class DPFABase(tf.keras.Model):
    r""" this is an implementation of the neural Performance Analysis"""

    def __init__(
            self,
            hidden_size,
            dropout,
            item_vocab_size,
            regulate_dot_product=False,
            normalize_embedding=True,
            time_decay=True,
            **kwargs):
        super(DPFABase, self).__init__(**kwargs)

        # use to learn the embedding of items
        self.item_embedding_layer = tf.keras.layers.Embedding(
            item_vocab_size,
            hidden_size,
            embeddings_initializer=tf.random_normal_initializer(mean=0., stddev=hidden_size ** -0.5),
        )

        # decay the importance item responses over time
        if time_decay:
            self.time_decay_layer = layers.DenseEinsum(
                output_shape=1,
                kernel_initializer='glorot_uniform',
                use_bias=True,
                activation='linear',
                name='time_decay_layer'
            )

        # layer to model the difficulty of item
        self.item_beta_weights = self.add_weight(
            shape=[item_vocab_size],
            initializer=tf.random_normal_initializer(mean=0., stddev=1),
            name='item_beta_weights')

        # layer to model the expected mastery based on right and wrong response
        self.item_response_vals = self.add_weight(
            shape=[item_vocab_size, 2],
            initializer=tf.random_normal_initializer(mean=0., stddev=1),
            name='item_response_weights')

        self.params = {
            'hidden_size': hidden_size,
            'item_vocab_size': item_vocab_size,
            'dropout': dropout,
            'regulate_dot_product': regulate_dot_product,
            'normalize_embedding': normalize_embedding,
            'time_decay': time_decay,
        }

    def get_item_difficulty(self, items: tf.Tensor) -> tf.Tensor:
        """ get item difficulties
        Parameters
        ----------
        items: tf.Tensor
            item index. shape = [batch_size, seq_len], dtype = tf.int32/64

        Returns
        -------
        items_difficulty: tf.Tensor
            item difficulty, shape = [batch_size, seq_len], dtype = tf.float32
        """
        item_params = self.item_beta_weights  # shape = [item_vocab_size]
        item_idx_params = tf.gather(item_params, items)

        return item_idx_params

    def get_item_embedding(self, items: tf.Tensor) -> tf.Tensor:
        """ get item embedding, used to find the correlation between items
        Parameters
        ----------
        items: tf.Tensor
            item index. shape = [batch_size, seq_len], dtype = tf.int32/64

        Returns
        -------
        item_embedding: tf.Tensor
            item embedding. shape = [batch_size, seq_len, hidden_size],
            dtype = tf.float32
        """
        # shape = [batch_size, seq_len, hidden_size]
        item_embedding = self.item_embedding_layer(items)
        if self.params['normalize_embedding']:
            item_norms = tf.linalg.norm(item_embedding, keepdims=True, axis=-1)
            item_embedding /= item_norms

        return item_embedding

    def get_stu_ability(
            self,
            hist_item_embeddings: tf.Tensor,
            next_item_embeddings: tf.Tensor,
            hist_item_mastery: tf.Tensor,
            hist_items: tf.Tensor,
            training: bool = True
    ) -> tf.Tensor:
        """ infer a student's ability from past item responses
        Parameters
        ----------
        hist_item_embeddings: tf.Tensor
            item embedding for historical items.
            shape = [batch_size, seq_len, hidden_size], dtype = tf.float32
        next_item_embeddings: tf.Tensor
            item embedding for next item
            shape = [batch_size, seq_len, hidden_size], dtype = tf.float32
        hist_item_mastery: tf.Tensor
            expected mastery value from the history item response
            shape = [batch_size, seq_len], dtype = tf.float32
        hist_items: tf.Tensor
            history item index, shape = [batch_size, seq_len], dtype = tf.int32
        training: bool
            if True, activate dropout layers. Used during training stage

        Returns
        -------
        theta: tf.Tensor
            student ability. shape = [batch_size, seq_len], dtype = tf.float32
        """
        # correlation between history and next items
        weight = tf.einsum(
                'BQH, BSH -> BQS',
                next_item_embeddings,
                hist_item_embeddings)

        # force the correlation between two items > 0
        # intuition:
        # answering a history item right cannot decrease the possibility of
        # correctly answering future
        if self.params['regulate_dot_product']:
            weight = tf.nn.relu(weight)

        # mask out present, future, and padding
        bias = self._get_bias(hist_items)

        # put more weight to recent history than distant history
        if self.params['time_decay']:
            time_decay = self._get_time_decay_logit(
                hist_items,
                training=training)
        else:
            time_decay = 0

        # shape = [batch_size, q_seq_len, s_seq_len]
        weight = tf.nn.softmax(weight + bias + time_decay)
        # shape = [batch_size, 1, s_seq_len],
        # history item value is constant across query time
        hist_item_mastery = tf.expand_dims(hist_item_mastery, axis=-2)
        # shape = [batch_size, q_seq_len, s_seq_len]
        ability = weight * hist_item_mastery
        ability = tf.reduce_sum(ability, axis=-1, keepdims=False)
        return ability

    def get_item_expected_mastery(
            self,
            hist_items: tf.Tensor,
            hist_corrects: tf.Tensor,
            training=True,
    ):
        """ calculate expected item mastery from item responses
        Parameters
        ----------
        hist_items: tf.Tensor
            index for history items.
            shape = [batch_size, seq_len], dtype = tf.int32
        hist_corrects: tf.Tensor
            whether an item is correctly responded.
            shape = [batch_size, seq_len], dtype = tf.int32
            padding = 0, wrong = 1, right = 2
        hist_item_times: tf.Tensor, default = None
            time used to answer an item
            shape = [batch_size, seq_len], dtype = tf.float32
        training: bool, default = True
            if True, activate dropout layers

        Returns
        -------
        hist_item_mastery, tf.Tensor
            expected mastery value from item responses
            shape = [batch_size, seq_len], dtype = tf.float32
        """
        # convert 2 -> 1, 1 -> 0, 0 -> 0
        hist_is_correct = tf.cast(
            tf.equal(hist_corrects, 2),
            dtype=hist_items.dtype)

        idx = tf.stack([hist_items, hist_is_correct], axis=-1)
        params = self.item_response_vals
        hist_mastery = tf.gather_nd(params, idx)

        return hist_mastery

    def call(self, inputs, training=True):
        """ get students' predicted probability of answering next question right
        Parameters
        ----------
        inputs: dict
            input dictionary
        training: bool
            if True, activate dropout layers

        Returns
        -------
        predicted_probability: tf.Tensor
            predicted probability of correctly answering next question,
            shape = [batch_size, seq_len], dtype = tf.float32

        """
        # shape = [batch_size, seq_len, hidden_size]
        hist_item_embeddings = self.get_item_embedding(
            inputs['history_items'])

        next_item_embeddings = self.get_item_embedding(
            inputs['next_items'])

        # dropout during training
        if training:
            hist_item_embeddings = tf.nn.dropout(
                hist_item_embeddings,
                rate=self.params['dropout'])

        # get history item mastery
        hist_item_mastery = self.get_item_expected_mastery(
            hist_items=inputs['history_items'],
            hist_corrects=inputs['history_corrects'],
            training=training)

        # calculate study ability
        stu_ability = self.get_stu_ability(
            hist_item_embeddings,
            next_item_embeddings,
            hist_item_mastery,
            inputs['history_items'],
            training=training)

        # get item difficulty for next items
        next_item_diff = self.get_item_difficulty(items=inputs['next_items'])

        # probability of answering next question correct
        probs = tf.nn.sigmoid(stu_ability - next_item_diff)

        return probs

    def _get_time_decay_logit(
            self, history_items: tf.Tensor,
            training: int = True) -> tf.Tensor:
        """ calculate time decay logits for history item responses.
        Parameters
        ----------
        history_items: tf.Tensor
        training: bool

        Returns
        -------
        decay_weight: tf.Tensor
            logit for time decay
        """
        # position shape = [batch_size, seq_len]
        batch_size, seq_len = tf.shape(history_items)[0], tf.shape(history_items)[1]
        hist_position = tf.cast(tf.range(seq_len), dtype=tf.float32)
        hist_position = tf.expand_dims(hist_position, axis=0)
        hist_position = tf.broadcast_to(hist_position, shape=[batch_size, seq_len])
        next_position = 1 + hist_position

        # expand dimensions
        hist_position = tf.expand_dims(hist_position, axis=-2)
        next_position = tf.expand_dims(next_position, axis=-1)

        # shape = [batch_size, q_seq_len, s_seq_len]
        next_hist_dist = next_position - hist_position
        # shape = [batch_size, q_seq_len, s_seq_len, 1]
        next_hist_dist = tf.expand_dims(next_hist_dist, axis=-1)

        decay_weights = self.time_decay_layer(next_hist_dist)[:, :, :, 0]
        # decay_weights = self.time_decay_layer1(next_hist_dist)
        # if training:
        #     decay_weights = tf.nn.dropout(decay_weights, rate=self.params['dropout'])
        # decay_weights = self.time_decay_layer2(decay_weights)[:, :, :, 0]

        return decay_weights

    @staticmethod
    def _get_bias(history_items: tf.Tensor) -> tf.Tensor:
        """ get bias to mask out present, future, and padding
        Parameters
        ----------
        history_items: tf.Tensor
            history item index, shape = [batch_size, seq_len], shape = tf.int32

        Returns
        -------
        bias: tf.Tensor
            -inf for mask out items, 0 other wise.
            shape = [batch_size, seq_len, seq_len], shape = tf.float32

        """
        seq_len = tf.shape(history_items)[1]

        # 1. mask out padding
        # bias shape = [batch_size, 1, seq_len]
        bias = get_padding_bias(history_items)[:, :, 0, :]

        # 2. mask out future
        # the last two dimension of the future bias, B, equals 0 in the lower
        # triangle and main diagonal, and equals -inf everywhere else.
        # e.g.,
        # [0, -inf, -inf]
        # [0, 0   , -inf]
        # [0, 0   , 0   ]
        # this is used to mask out present and future item responses when infer
        # a student's ability
        # B_{ij} refers to the bias to time j when a student is at time i + 1
        # So diagonal refers to bias to time t when user is at time t + 1.
        # Therefore, the right of the diagonal need to be mask out
        future_bias = get_decoder_self_attention_bias(seq_len)[0, :, :, :]

        # combine 1 and 2
        bias = tf.math.minimum(bias, future_bias)

        return bias