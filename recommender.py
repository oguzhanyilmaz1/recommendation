import tensorflow as tf

class Only_Embeddings_Recommender(tf.keras.Model):
    def __init__(self, users, songs,length_of_embedding):
        super(Only_Embeddings_Recommender, self).__init__()
       
        self.songs=songs
        self.users=users
        self.user_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.users, range(len(users))), -1)
        self.song_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.songs, range(len(songs))), -1)
        
        #Embedding layers is defined for user and song pipelines
        self.user_embedding = tf.keras.layers.Embedding(len(users), length_of_embedding)
        self.song_embedding = tf.keras.layers.Embedding(len(songs), length_of_embedding)
        #Dot production is defined
        self.dot=tf.keras.layers.Dot(axes=-1)
        
    def call(self,inputs):
        user=inputs[0]
        songs=inputs[1]
        user_embedding_index=self.user_table.lookup(user)
        song_embedding_index=self.song_table.lookup(songs)

        user_embedding_values=self.user_embedding(user_embedding_index)
        song_embedding_values=self.song_embedding(song_embedding_index)

        #Tensor is squeezing to get rif of extra "1" dimension coming from matrix production.
        r=tf.squeeze(self.dot([user_embedding_values,song_embedding_values]),1)

        return r
    
class Recommender(tf.keras.Model):
    def __init__(self, users, songs,length_of_embedding):
        super(Recommender, self).__init__()
       

        self.songs=songs
        self.users=users
        self.user_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.users, range(len(users))), -1)
        self.song_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.songs, range(len(songs))), -1)
        self.user_embedding = tf.keras.layers.Embedding(len(users), length_of_embedding)
        self.song_embedding = tf.keras.layers.Embedding(len(songs), length_of_embedding)
        self.dot=tf.keras.layers.Dot(axes=-1)
        self.ratings = tf.keras.Sequential([
              # Learn multiple dense layers.
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128, activation="relu"),
              tf.keras.layers.Dense(64, activation="relu"),
              # Make rating predictions in the final layer.
              tf.keras.layers.Dense(10, activation="softmax")
              
          ])
        
        
    def call(self,inputs):
        user=inputs[0]
        songs=inputs[1]
        user_embedding_index=self.user_table.lookup(user)
        song_embedding_index=self.song_table.lookup(songs)
        #Embedding layers are implemented.
        user_embedding_values=self.user_embedding(user_embedding_index)
        song_embedding_values=self.song_embedding(song_embedding_index)
        #Embeddings are concatenated.
        r=tf.concat([user_embedding_values, song_embedding_values], axis=1) 
        #Flattened and dense layers were applied, respectively.
        r=self.ratings(r) 

        return r 
    
