from util import *

def work(model, name, X_train, y_train):
    print('begin training: %s' % name)
    model.fit(X_train, y_train)
    print('finish training: %s' % name)
    dump(model, 'models/' + name)


def build_models(data_name, ver, model_index):
    X = load(f'data/{data_name}')
    y = load('data/y')

    np.random.seed(split_seed)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=total_train_size)

    for idx, model, arg, name, count in zip(range(len(models)), models, args, names, counts):
        if idx not in model_index:
            continue
        for i in range(count // concurrent):
            procs = [None] * concurrent
            for j in range(concurrent):
                xt, _, yt, _ = train_test_split(X_train, y_train, train_size=train_size)
                procs[j] = Process(target=work, args=(model(**arg), f'{name}_{ver}_{i * concurrent + j}', xt, yt))
                procs[j].start()
            for j in range(concurrent):
                procs[j].join()


def predict(X, name, ver, count, *y):
    pred = np.zeros((count, X.shape[0]))
    for i in range(count):
        model = load(f'models/{name}_{ver}_{i}')
        pred[i, :] = model.predict(X)
    mean = pred.mean(axis=0)
    # given y, calculate and print rmse
    if y:
        y = y[0]
        errors = [mean_squared_error(pred[i, :], y, squared=False) for i in range(count)]
        errors = map(lambda x: '%.2f' % x, errors)
        print('separate models:', *errors)
        print(name, mean_squared_error(mean, y, squared=False))
    else:
        return mean

if __name__ == "__main__":
    # build_models('tfidf', 3, [3])
    # build_models('bag_of_words', 3, [3])

    build_models('bag_of_words', 7, [4])