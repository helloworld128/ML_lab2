from util import *
from main import prepare, valid, submit
from bagging import build_models

prepare(CountVectorizer, 'bag_of_words')
build_models('bag_of_words', 1, [0, 1, 2, 3])
for name in names:
    valid('bag_of_words', name, 1, 32)
    submit('bag_of_words', name, 1, 32)
