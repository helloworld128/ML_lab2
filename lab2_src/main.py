from util import *
from bagging import predict


def summary():
    df = pd.read_csv('data/train.csv', sep='\t')
    print(df.shape)
    print(df.columns)

    print(mean_squared_error(np.ones(df.shape[0]) * np.mean(df.overall), df.overall, squared=False))

    user_count = df.groupby('reviewerID').count().overall
    user_count.sort_values(ascending=False)
    print(user_count.describe(percentiles=np.arange(0.5, 1.0, 0.1)))

    item_count = df.groupby('asin').count().overall
    item_count.sort_values(ascending=False)
    print(item_count.describe(percentiles=np.arange(0.5, 1.0, 0.1)))


def prepare(Vectorizer, name):
    vectorizer = Vectorizer(tokenizer=tokenizer, min_df=3, stop_words=stopwords.words('english'))
    df = pd.read_csv('data/train.csv', sep='\t')
    dump(df.overall.astype(np.int8), 'data/y')
    corpus = df.summary + ' ' + df.reviewText
    corpus.fillna('', inplace=True)

    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    dump(X, f'data/{name}')
    names = vectorizer.get_feature_names()
    print(*names, file=open(f'data/tokens_{name}.txt', 'w'))
    dump(names, f'data/tokens_{name}')

    df_test = pd.read_csv('data/test.csv', sep='\t')
    corpus = df_test.summary + ' ' + df_test.reviewText
    corpus.fillna('', inplace=True)
    vectorizer = Vectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'), vocabulary=load(f'data/tokens_{name}'))
    X_test = vectorizer.fit_transform(corpus)
    dump(X_test, f'data/{name}_test')

    dump(df_test.id, 'data/id_test')


def valid(data_name, model_name, ver, count):
    X = load(f'data/{data_name}')
    y = load('data/y')

    np.random.seed(split_seed)
    _, X_valid, _, y_valid = train_test_split(X, y, train_size=total_train_size)

    predict(X_valid, model_name, ver, count, y_valid)


def submit(data_name, model_name, ver, count):
    X = load(f'data/{data_name}_test')
    id = load('data/id_test')

    y_pred = predict(X, model_name, ver, count)
    df = pd.DataFrame({'id':id, 'predicted':y_pred})
    df.to_csv(f'output/{data_name}_{model_name}_{ver}_{count}.csv', index=None)


if __name__ == "__main__":
    summary()
    # prepare(CountVectorizer, 'bag_of_words')
    # prepare(TfidfVectorizer, 'tfidf')
    
    # bagging

    # valid('bag_of_words', 'svm', 2, 32) # 0.887 1.08
    # valid('bag_of_words', 'dt', 2, 32) # 0.909 1.01
    # valid('bag_of_words', 'svr', 3, 32) # 1.078 1.25

    # submit('bag_of_words', 'svm', 2, 32)
    # submit('bag_of_words', 'dt', 2, 32)

    # valid('tfidf', 'svm', 3, 32) # 0.896 1.00
    # valid('tfidf', 'dt', 3, 32) # 0.907 1.04
    # valid('tfidf', 'dtc', 3, 32) # 0.951 1.20
    # valid('tfidf', 'svr', 3, 32) # 1.090 1.09

    # adaboost

    # valid('bag_of_words', 'svr', 4, 3) # 1.187
    # valid('bag_of_words', 'dt', 4, 3) # 0.969
    # valid('bag_of_words', 'dtc', 4, 3) # 1.048

    # valid('bag_of_words', 'svr', 5, 32) # 1.151
    # valid('bag_of_words', 'dt', 5, 32) # 0.955
    # valid('bag_of_words', 'dtc', 5, 32) # 0.974
    # valid('bag_of_words', 'svm', 5, 14) # 0.969

    # submit('bag_of_words', 'dt', 5, 32)
    # submit('bag_of_words', 'svm', 5, 14)

    # bagging

    # valid('bag_of_words', 'mlp', 7, 32) # 0.781, 1.03
    # submit('bag_of_words', 'mlp', 7, 32)
    
    pass

