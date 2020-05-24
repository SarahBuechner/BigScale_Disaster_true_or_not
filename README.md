# Gathering Insight from Real or Not real disaster Data

_A project by Team Google (Big Scale analytics, HEC Lausanne, Summer 2019)_

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

Our data comes from the Kaggle Competition: 'Real or Not? NLP with Disaster Tweets'
https://www.kaggle.com/c/nlp-getting-started/overview

Our data set has the following features: id,keyword,location,text. The column target is 1 if the tweet correponds to a disaster and zero if not. 

### Imports
For our project it is necessary to import the following libraries: numpy, pandas, seaborn, nltk, matplotlib, io, sklearn and yellowbrick. 

In fact, you can copy and paste the code below: 
```python
# EDA and data processing
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
import io
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Machine Learning
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.draw import manual_legend
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
```


