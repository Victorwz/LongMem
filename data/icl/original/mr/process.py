import json
import pandas as pd
from pandas import DataFrame

f = open('./mr.all')
class_num = 2

mapping = {}

for line in f:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    if not label in mapping:
        mapping[label] = []
    mapping[label].append([label, text])

test = []
train = []
for label in mapping:
    test += mapping[label][:1000]
    train += mapping[label][1000:]

test = DataFrame(test)
train = DataFrame(train)
test.to_csv('test.csv', header=False, index=False)
train.to_csv('train.csv', header=False, index=False)

