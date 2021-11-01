import tensorflow as tf
from tensorflow.keras import layers as tfnn



# globales
max_seq_length = 40
vocab_size = 10000
embedding_dims = 256



class TransformerInputEmbeddingBlock(tfnn.Layer):
    """prend les mots tokenizés sous forme d'une séquence de longueur max 
    max_seq_length et renvoie leur embedding dans un espace vectoriel de dim
    embedding_dim

    Args:
        layers ([type]): [description]
    """
    def __init__(self, max_seq_length, vocab_size, embedding_dim):
        
        super(TransformerInputEmbeddingBlock, self).__init__()
        
        # layers
        self.tokens_embedding = tfnn.Embedding(input_dim = vocab_size, output_dim = embedding_dim) # input_dim = vocab_size 10 000 : int entre 0 et 10 000-1 --> output_dim = embedding_dim 256 : vecteur de dim 256
        self.position_embedding = tfnn.Embedding(input_dim = max_seq_length, output_dim = embedding_dim) # input_dim = max_seq_length 40 : int entre 0 et 40-1 --> output_dim = embedding_dim 256 : vecteur de dim 256
                
    def call(self, x):
        """
        s'il y a du padding, on ne le prend pas en compte, on redefinit max_seq_length = seq_length via n_words

        Args:
            x ([type]): shape de [n_seq, n_words], n_words in [|0, vocab_size - 1|]

        Returns:
            [type]: [description]
        """
        n_words = tf.shape(x)[-1]
        pos = tf.range(start = 0, limit = n_words) # pas besoin de tile ici, meme si n_seq > 1, ca va l'ajouter correctement dans le dernier '+'
        
        out = self.tokens_embedding(x)
        pos = self.position_embedding(pos)

        return out + pos



class Mask:
    def __init__(self, batch_size, length):
        
        self.length = length
        self.batch_size = batch_size
    
    def generate_mask(self):
        
        x, y = tf.expand_dims(tf.range(self.length), -1), tf.range(self.length) # [[0], ..., [length - 1] ], [0, ..., length - 1]
        mask = x >= y # reviens à faire pour chaque sous-array a de x, (pour chaque n de y est ce que a >= y )
        mask = tf.tile(mask, [self.batch_size, 1]) # repete le mask batch_size fois



class AttentionBlock(tfnn.Layer):
    """Attention layer

    Args:
        tfnn ([type]): [description]
    """
    def __init__(self, n_heads, embedding_dim, dropout_rate):
        
        super(AttentionBlock, self).__init__()
        
        self.attention = tfnn.MultiHeadAttention(num_heads = n_heads, key_dim = embedding_dim, value_dim = embedding_dim)
        self.dropout = tfnn.Dropout(dropout_rate)
        self.norm = tfnn.LayerNormalization(epsilon = 1e-6)
        
        
    def call(self, x):
        
        batch_size, length = tf.shape(x)[0], tf.shape(x)[1] # x : [b, l, emb_dim]
        mask = Mask(batch_size, length)
        mask_decoder = mask.generate_mask()
        
        out = self.attention(query = x, value = x, attention_mask = mask_decoder, return_attention_score = False)
        out = out + x
        
        out = self.dropout(out)
        out = self.norm(out)
        
        return out
        
        

class FeedForwardNeuralNetworkBlock(tfnn.Layer):
    """[summary]

    Args:
        tfnn ([type]): [description]
    """
    def __init__(self, n_units = embedding_dims, dropout_rate = .1):
        
        super(FeedForwardNeuralNetworkBlock, self).__init__()
        
        self.layer_0 = tfnn.Dense(n_units, activation = 'relu')
        self.layer_1 = tfnn.Dense(n_units, activation = 'linear')
        
        self.dropout = tfnn.Dropout(dropout_rate)
        self.norm = tfnn.LayerNormalization(epsilon = 1e-6)
        
        
    def call(self, x):
        
        out = self.layer_0(x)
        out = self.layer_2(out)
        out = out + x
        
        out = self.dropout(out)
        out = self.norm(out)
        
        return out



class DecoderGPT:
    """[summary]
    """
    def __init__(self, max_seq_length = 40, 
                 vocab_size = 10000, 
                 embedding_dims = 256,
                 n_heads = 2,
                 dropout_rate = .1):
        
        self.E = TransformerInputEmbeddingBlock(max_seq_length = max_seq_length, vocab_size = vocab_size, embedding_dim = embedding_dims)
        self.A = AttentionBlock(n_heads = n_heads, embedding_dims = embedding_dims, dropout_rate = dropout_rate)
        self.F = FeedForwardNeuralNetworkBlock(n_units = embedding_dims, dropout_rate = dropout_rate)
        
    def decoder(self, x):
        
        out = self.E(x)
        out = self.A(out)
        out = self.F(out)
        
        return out
        
        
        
        
        
        
        
            
                 







































