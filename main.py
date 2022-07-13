import numpy as np
import tensorflow as tf
import random
from recommender import Only_Embeddings_Recommender, Recommender
from processor import Processor, Data_Parser

#Getting unique users, unique songs, song and rating part of data
users,songs,data_song,data_rating=Processor().data_creater()

#Getting unique users, unique songs, song, rating part and scored data
#This line is optional dataset using score that calculated by us instead of rating data
#But getting score for 2M line takes time.
#users,songs,data_song,data_rating=Processor().data_creater_with_scored()


embedding_size=16
learning_rate=0.001
epochs=100
#Model is defined with embedding vector size of 8 
#model=Recommender(users, songs, embedding_size)#This model includes embedding and feed forward layers
model=Only_Embeddings_Recommender(users, songs, embedding_size) #This model includes only embedding layers.

batch_number=1000
#MSE is used for loss function.
#For optimizer Adam is choosed with 0.001 learning rate.
#Metric is RMSE.
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
              )
#Model is fitted with 200_000 user, 2_000_000 rated songs and song ratings  
model.fit(Data_Parser(users,data_song,data_rating,batch_number),epochs=epochs)

#After training the model. A user is defined randomly and all songs are predicted for the user.
#End of the prediction, predicted values are sorted and 5 most relevant items are defined. 

test_songs=songs[:-1]
test_rating=data_rating[:len(test_songs)]
test_user=np.ones([int(len(songs)/10)])*random.randint(0,200_000)
test_user=tf.constant(test_user, dtype=tf.int32)

predictions=np.zeros([1])
for (u,s),_ in Data_Parser(test_user,test_songs,data_rating[:127770],batch_number):
    
    predictions=np.append(predictions,model.predict([tf.constant(u),tf.constant(s)]))
    
    
top_5_idx = np.argsort(predictions[1:])[-5:]
top_5_values = [predictions[i] for i in top_5_idx]    
    
print(f"5 most relevant items for user {test_user[0]} are: {top_5_idx} ")    
    
    
    
    
    
    
    
    
    
    
    
    