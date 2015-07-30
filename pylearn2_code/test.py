import os
import pylearn2

from pylearn2.config import yaml_parse

with open('reduceFAmodel.yaml', 'r') as f:
    train_2 = f.read()

train_2 = yaml_parse.load(train_2)
train_2.main_loop()
