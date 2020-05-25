import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from multiprocessing import Process

def dump(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))


pattern = re.compile(r'[a-z0-9]{2,}')
stem = SnowballStemmer('english').stem

def tokenizer(s):
    s = s.lower()
    return map(stem, re.findall(pattern, s))

split_seed = 19990321

models = [LinearSVC, DecisionTreeRegressor, DecisionTreeClassifier, LinearSVR, MLPClassifier]
args = [{'max_iter':3000}, {'min_samples_leaf':10}, {'min_samples_leaf':10}, {'max_iter':3000}, {}]
names = ['svm', 'dt', 'dtc', 'svr', 'mlp']
total_train_size = 0.8
train_size = 0.6

counts = [32] * len(models)
concurrent = 32

ada_iter = 24
ada_correct_threshold = 1.2

