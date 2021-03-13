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
        dat[col].values[1:1642] = c7
dat