import json
import pandas as pd
from pandas import DataFrame

ftrain = open('./stsa.fine.train')
train = []
for line in ftrain:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    train.append([label, text])
train = DataFrame(train)
train.to_csv('train.csv', header=False, index=False)

ftest = open('./stsa.fine.test')
test = []
for line in ftest:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    test.append([label, text])
test = DataFrame(test)
test.to_csv('test.csv', header=False, index=False)
