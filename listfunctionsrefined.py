import numpy as np
import pandas as pd
import math

def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance

def stdev(data):
        var = variance(data)
        std_dev = math.sqrt(var)
        return std_dev

dat = pd.read_csv('C://Datasets//demo.csv')

c1=dat.mean()
c2=dat.median()
c3=dat.mode()
c4=dat.max()
c5=dat.min()
c6=variance(dat.col)
c7=stdev(dat.col)


for col in dat.columns:
    if np.issubdtype(dat[col].dtype, np.number):
        dat[col].values[1:10] = c7
dat

dat.truncate(before=1, after=4)
dat.truncate(before=1640, after=1643)

data = dat.drop(labels=2, axis=0)
data

dat.drop(range(0,3))
dat.drop(range(1640,1645))


