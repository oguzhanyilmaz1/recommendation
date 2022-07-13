import numpy as np
import tensorflow as tf
from numpy import genfromtxt
class Processor():
        
        def data_creater(self):
           #Getting data from .csv file.
           data = genfromtxt('songsDataset.csv', delimiter=',')[1:]
           #Defining number of rated song for by each user.
           rating_number=10 
           #length of overall data 200_000(user)*rating_number(10)=2_000_000.
           data_length=len(data)
           #Rated 10 songs are shuffled for each user .
           #To do that data is reshaped to 200_000,10,3.
           data = data.reshape((int(data_length/rating_number),rating_number,3))
           #Data is shuffled
           np.apply_along_axis(np.random.shuffle, 1, data);
           #Data is reshaped back to initial sizes.
           data = data.reshape((2000000,3))
           
           #Arrays are converted to tensors
           #*int32 is casted because of compatibility(rather than *64).
           #Also float type is not required for our data.
           data=tf.convert_to_tensor(data)
           data=tf.cast(data, tf.int32)
           
           #User, song and rating columns are seperated to create dataset.
           users=data[:,0]
           data_song=data[:,1]
           data_rating=data[:,2]/10
           
           #Unique users are defined in case the data is out of order. 
           users,_=tf.unique(users)
           #Unique songs are defined. Repetitions are defined as song_count. This value represent the number of plays.
           songs,_=tf.unique(data_song)
           
           
           return users,songs,data_song,data_rating
       
        def data_creater_with_scored(self):
          #Getting data from .csv file.
          data = genfromtxt('songsDataset.csv', delimiter=',')[1:]
          #Defining number of rated song for by each user.
          rating_number=10 
          #length of overall data 200_000(user)*rating_number(10)=2_000_000.
          data_length=len(data)
          #Rated 10 songs are shuffled for each user .
          #To do that data is reshaped to 200_000,10,3.
          data = data.reshape((int(data_length/rating_number),rating_number,3))
          #Data is shuffled
          np.apply_along_axis(np.random.shuffle, 1, data);
          #Data is reshaped back to initial sizes.
          data = data.reshape((2000000,3))
          
          #Arrays are converted to tensors
          #*int32 is casted because of compatibility(rather than *64).
          #Also float type is not required for our data.
          data=tf.convert_to_tensor(data)
          data=tf.cast(data, tf.int32)
          
          #User, song and rating columns are seperated to create dataset.
          users=data[:,0]
          data_song=data[:,1]
          data_rating=data[:,2]/10
          
          #Unique users are defined in case the data is out of order. 
          users,_=tf.unique(users)
          #Unique songs are defined. Repetitions are defined as song_count. This value represent the number of plays.
          songs,_,song_count=tf.unique_with_counts(data_song)
          
          #Scoring function is the multiplication of number of plays and rating points.
          
          #Songs and plays numbers are defined as song_frequency
          song_frequency=tf.concat([tf.reshape(songs, [len(songs),1]), tf.reshape(song_count, [len(songs),1])], axis=1)
          
          #Score matrix is defined.
          new_rating_score=np.ones([2_000_000])
          
          #A loop is defined for each song rating listed in dataset
          for i in range(len(data_rating)):
              
              #Getting ID of song each step.
              song=int(data[:,1:3][i,0])
              
              #Looking for number of play of the song from song_frequency
              frequency=int(song_frequency[song_frequency[:,0]==song][0,1])
              
              #Score is calculated by multiplying the plays number and rating given from users.
              new_rating_score[i]=data_rating[i]*frequency
          
            #Score matrix is normalized because there are songs that rated one time and some of them's playing number is huge.
          new_rating_score=tf.convert_to_tensor(new_rating_score)
          new_rating_score=tf.cast(new_rating_score, tf.float32)    
          new_rating_score=tf.linalg.normalize(
              new_rating_score, ord='euclidean', axis=None, name=None
          )
          return users,songs,data_song,new_rating_score
      
       
class Mapper():
    def __init__(self):

        pass
        
    def __call__(self, user,songs,ratings):
       
        return (user, songs) ,ratings
       
def Data_Parser(users,data_song,data_rating,batch_number):
    #Song and rating columns of dataset are divided into 10. So that each row represents one user.
    song_len=int(len(data_song)/(10))
    rating_len=int(len(data_rating)/(10))
    users=tf.expand_dims(users, axis=-1)
    dataset=tf.data.Dataset.from_tensor_slices((users ,tf.reshape(data_song, [song_len,10]), tf.cast(tf.reshape(data_rating, [rating_len,10]), tf.float32))).map(Mapper())
   
    #data is divided into batches  
    dataset=dataset.batch(batch_number)
    
    return dataset        