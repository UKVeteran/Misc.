import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import math
import array
import matplotlib.pyplot as plt
from keras.regularizers import l1,l2
from tensorflow import feature_column
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
%matplotlib inline

from numpy.random import seed
import random

dat = pd.read_csv('C://Datasets//demo.csv')
dat.head(30)
dat.dtypes

dat["col"] = dat["col"].astype(str).astype(int)
print(dat.dtypes)


#one value at a time
dat.replace(2, 1700)

#multiple stated values replaced
dat.replace({'col': {1: 100, 2: 400, 4:6050}})

dat.replace([0,1,2,3,4], 42, inplace=True)
dat

def my_mean(sample):
     return sum(sample) / len(sample)
my_mean([1,2,3,4,5])

c1=dat.mean()
c2=dat.median()
c3=dat.mode()
c4=dat.max()
c5=dat.min()
#For NaNs
dat.apply(lambda x: x.fillna(x.mean()),axis=0)

for col in dat.columns:
    if np.issubdtype(dat[col].dtype, np.number):
        dat[col].values[1:1642] = c1
dat

def variance(data):
# Number of observations
    n = len(data)
# Mean of the data
    mean = sum(data) / n
# Square deviations
    deviations = [(x - mean) ** 2 for x in data]
# Variance
    variance = sum(deviations) / n
    return variance

c6=variance(dat.col)

def stdev(data):
        var = variance(data)
        std_dev = math.sqrt(var)
        return std_dev


c7=stdev(dat.col)
c7

for col in dat.columns:
    if np.issubdtype(dat[col].dtype, np.number):
        dat[col].values[1:1642] = c7
dat




