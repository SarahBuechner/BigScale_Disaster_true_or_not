# Gathering Insight from Real or Not real disaster Data

_A project by Team Google (Big Scale analytics, HEC Lausanne, Summer 2019)_

## Abstract
Real or Not? NLP with Disaster Tweets: In this project we are challenged to build a Machine Learning model that can predict which tweets are about a real disaster and which are not. The project topic is based around a Kaggle competition. You can find the link to the competition [here](https://www.kaggle.com/c/nlp-getting-started/overview). You will find more information about the project and the dataset in the competition page.

In this project, we always compare the results of our predictions with those of other teams and also with those of other Kaggle users). 
We are UNIL_GOOGLE on Kaggle. You will see our ranking there

## DATABASE Kaggle

* __id__ - a unique identifier for each tweet
* __text__ - the text of the tweet
* __location__ - the location the tweet was sent from (may be blank)
* __keyword__ - a particular keyword from the tweet (may be blank)
* __target__ - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


## Team Members
*Florian Emery
*Ibrahim Ounon
*Pau Gallardo Campos
*Sarah Büchner


## Link to Video

The video for our project is available on [YouTube](https://youtu.be/tdJWsxcjBZs). 

## Project Structure

To make the project work smoothly, it is advised to use an online platform such as Google Colaboratory

### /code
This folder contains **1 notebook**:

* `BSA_P2_Team_google.ipynb`: Contains an introduction, explanations about the data cleaning process, the exploratory data analysis (EDA), ML models.


### /data

*train.csv - the training set 
*test.csv - the test set
*sample_submission.csv - a sample submission file in the correct format
*disaster_words - contains a list of disaster-related words


