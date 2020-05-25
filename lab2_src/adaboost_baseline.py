from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from util import *

def adaboost_baseline(data_name, model, name, ver, ada_iter, ty):
    X = load(f'data/{data_name}')
    y = load('data/y')

    np.random.seed(split_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=total_train_size)
    if ty == 0:
        ada = AdaBoostClassifier(base_estimator=model, n_estimators=ada_iter, algorithm='SAMME')
    else:
        ada = AdaBoostRegressor(base_estimator=model, n_estimators=ada_iter)
    ada.fit(X_train, y_train)
    dump(ada, f'models/ada_{name}_{ver}')
    pred = ada.predict(X_valid)
    dump(pred, f'output/ada_{name}_{ver}')
    pred[pred < 1] = 1
    pred[pred > 5] = 5
    print(name, mean_squared_error(pred, y_valid, squared=False))


if __name__ == "__main__":
    pool = []
    for i, model, arg, name, ty in zip(range(4), models, args, names, [0, 1, 0, 1]):
        p = Process(target=adaboost_baseline, args=('bag_of_words', model(**arg), name, 5, 32, ty))
        p.daemon = True
        p.start()
        pool.append(p)
    for p in pool:
        p.join()