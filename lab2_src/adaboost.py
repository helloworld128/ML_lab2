from util import *


class ThresholdTooSmall(Exception):
    pass

def inner_work(model, name, X_train, y_train, ada_iter, ada_correct_threshold):

    # ada_iter = 14

    fp = open(f'logs/{name}', 'w')
    n_samples = y_train.shape[0]
    model_weights = np.zeros(ada_iter)
    samples_weights = np.ones(n_samples)
    print('begin training: %s' % name, file=fp)

    for i in range(ada_iter):
        print(f'{name}: iter {i}', file=fp)
        fp.flush()
        model.fit(X_train, y_train, samples_weights)
        model = load(f'models/{name}_{i}')
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_pred, y_train, squared=False)
        correct_mask = np.abs(y_pred - y_train) <= ada_correct_threshold
        wrong_mask = ~correct_mask
        epsilon = np.sum(samples_weights * wrong_mask) / n_samples
        if epsilon > 0.5:
            print(f'{name}:iter {i} bad epsilon: {epsilon}, aborting', file=fp)
            return 1
        beta = epsilon / (1 - epsilon)
        print(f'epsilon = {epsilon}, beta = {beta}, rmse = {rmse}', file=fp)
        model_weights[i] = np.log(1 / beta)
        mask = correct_mask * beta + wrong_mask
        samples_weights *= mask
        samples_weights *= n_samples / np.sum(samples_weights)
        dump(model, f'models/{name}_{i}')
    dump(model_weights, f'models/{name}_weights')
    print('finish training: %s' % name, file=fp)
    return 0


def work(model, name, X_train, y_train, ada_iter):
    threshold = ada_correct_threshold
    while inner_work(model, name, X_train, y_train, ada_iter, threshold):
        threshold += 0.2


def build_models(data_name, ver, model_index):
    X = load(f'data/{data_name}')
    y = load('data/y')

    np.random.seed(split_seed)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=total_train_size)

    procs = []
    for i, model, arg, name in zip(range(len(models)), models, args, names):
        if i not in model_index:
            continue
        p = Process(target=work, args=(model(**arg), f'{name}_{ver}', X_train, y_train, ada_iter))
        p.daemon = True
        p.start()
        procs.append(p)

    for i, p in enumerate(procs):
        p.join()


def predict(X, name, ver, count, *y):
    pred = np.zeros((count, X.shape[0]))
    for i in range(count):
        model = load(f'models/{name}_{ver}_{i}')
        pred[i, :] = model.predict(X)
    weights = load(f'models/{name}_{ver}_weights')
    answer = weights.dot(pred) / np.sum(weights)
    print()
    print(f'low: {np.sum(answer < 1)}, high: {np.sum(answer > 5)}')
    answer[answer < 1] = 1
    answer[answer > 5] = 5
    # given y, calculate and print rmse
    if y:
        y = y[0]
        errors = [mean_squared_error(pred[i, :], y, squared=False) for i in range(count)]
        errors = map(lambda x: '%.2f' % x, errors)
        weights = map(lambda x: '%.2f' % x, weights)
        print('weights:', *weights)
        print('separate models:', *errors)
        print(name, mean_squared_error(answer, y, squared=False))
    else:
        return answer

if __name__ == "__main__":
    build_models('bag_of_words', 5, [0])