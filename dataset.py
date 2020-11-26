import os, shutil
import numpy as np

path = './kanjiStk/'
train_path = './train_data/'
test_path = './test_data/'

data = os.listdir('./kanjiStk/')
total = len(data)
np.random.shuffle(data)

rate = 0.8
num = int(rate * total)
train_set = data[0:num]
test_set = data[num:]

for i in train_set:
    shutil.copyfile(path+i,train_path+i)

for i in test_set:
    shutil.copyfile(path+i,test_path+i)


