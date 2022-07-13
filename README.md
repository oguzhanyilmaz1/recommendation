# recommendation
Collaborative Filtering Recommendation System Based on Deep Learning
Python Version: 3.9
IDE:Sypder 5.1.5
Modules:
numpy: 1.21.5
tensorflow: 2.9.1

---------Welcome------------
Scripts consist of 3 Scripts
Script 1:main.py
Script 2:processor.py
Script 3:recommender.py

main.py: Just run the main.py others will be called by main.

-------------------------------------------------------------------------------------
**There are 2 option to train dataset;
To change this option make comment line 8 or 13 in main.py.(Default is first option)
First one is using ratings from dataset as a "Y" groundtruth

Second one is using scores calcuated in processor.py. (Takes long time)


***There are two optional model
To change this option make comment the line 18 or 19 in main.py.(Default is first option)
First one is Only Embedding Layer Model

Second one Embedding layer+ Flattaen+ Dense Layers Model
--------------------------------------------------------------------------------------------
processor.py:
--Duties--
-Getting dataset
-Preprocessing dataset
-Parsing and mapping data into model


recommender.py:
--Duties--
-Building Model


