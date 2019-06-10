# tensorflow 2.0 based GPT2-model
import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
                                
    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

    return output, attention_weights

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % self.num_heads == 0

        self.depth = self.embed_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)
        
        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
            
        q = self.wq(q)  # (batch_size, seq_len, embed_dim)
        k = self.wk(k)  # (batch_size, seq_len, embed_dim)
        v = self.wv(v)  # (batch_size, seq_len, embed_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, embed_dim)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, embed_dim)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, embed_dim)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, embed_dim)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_v, num_heads, embed_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_v, embed_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, embed_dim)

        return output, attention_weights

class GPTwo(tf.keras.Model):
    def __init__(self, embed_dim, hidden_dim, vocab_size, max_seq_len, num_heads, num_layers, dropout_ratio):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(max_seq_len, embed_dim)

        self.dropout = tf.keras.layers.Dropout(dropout_ratio)

        self.layer_norm1 = []
        self.attentions = []
        self.layer_norm2 = []
        self.feed_forwards = []
        
        self.vocab_predictor = tf.keras.layers.Dense(vocab_size)

        for _ in range(num_layers):
            self.attentions.append(MultiHeadAttention(embed_dim, num_layers))
            self.feed_forwards.append(tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                                tf.keras.layers.Dense(embed_dim)]))
            self.layer_norm1.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.layer_norm2.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def call(self, inputs):
        positions = tf.expand_dims(tf.range(len(inputs)), axis=0)
        h = self.token_embeddings(inputs)
        h = h + tf.broadcast_to(self.position_embeddings(positions), h.shape)
        h = self.dropout(h)

        attn_mask = create_padding_mask(inputs)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norm1, self.attentions, self.layer_norm2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forwad(h)
            x = self.dropout(x)
            h = x + h

        h = self.vocab_predictor(h)

        return h
