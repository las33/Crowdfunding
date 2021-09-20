from deslib.des.meta_des import METADES
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA
from deslib.static.single_best import SingleBest
from deslib.static.static_selection import StaticSelection
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score



dfp = [True, False]
with_ih = [True, False]
learning_rate_init: [0.0001, 0.001, 0.01]


def return_model_by_name(model_name, pool, k, IH_rate=0):
    if model_name == "METADES":
        return METADES(pool_classifiers=pool, k=k, IH_rate=IH_rate,random_state=888)
    if model_name == "KNORAU":
        return KNORAU(pool_classifiers=pool, k=k, IH_rate=IH_rate, random_state=888)
    if model_name == "KNORAE":
        return KNORAE(pool_classifiers=pool, k=k, IH_rate=IH_rate, random_state=888)
    if model_name == "OLA":
        return OLA(pool_classifiers=pool, k=k, IH_rate=IH_rate, random_state=888)
    if model_name == "LCA":
        return LCA(pool_classifiers=pool, k=k, IH_rate=IH_rate, random_state=888)
    if model_name == "SingleBest":
        return SingleBest(pool_classifiers=pool, random_state=888)
    if model_name == "StaticSelection":
        return StaticSelection(pool_classifiers=pool, random_state=888)
    # if model_name == "OLP":
    #    return OLP(k=k,ds_tech=ds_tech,IH_rate=IH_rate)


def cross_validation_model(X_train, y_train, X_test, y_test, model_name, pool, k, ds_tech=0, IH_rate=0):

    model = return_model_by_name(model_name, pool, k, ds_tech, IH_rate)
    #print(model, model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)

    return acc, roc


def simple_gridSearch(model_name, pool, X_train, y_train, X_test, y_test):

    best_acc = 0
    best_roc = 0
    best_k = 0
    best_with_ih = 0
    best_pool = None
    list_k = [3, 7, 5, 9, 13]
    n_estimators_list = [10, 20, 30, 50, 70, 100]

    if(model_name == "METADES" or model_name == "KNORAU"or model_name == "KNORAE"):
        IH_rate_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        best_ds_tech = ''
        best_ihrate = 0.0
        for k in list_k:
            for IH_rate in IH_rate_list:
                for n in n_estimators_list:
                    pool_p = BaggingClassifier(
                        base_estimator=pool, n_estimators=n)
                    pool_p.fit(X_train, y_train)
                    acc, roc = cross_validation_model(
                        X_train, y_train, X_test, y_test, model_name, pool_p, k, IH_rate)
                    if(roc >= best_roc):
                        if(acc > best_acc):
                            best_acc = acc
                            best_roc = roc
                            best_k = k
                            best_ihrate = IH_rate
                            best_pool = pool_p
        return return_model_by_name(model_name, best_pool, best_k, best_ihrate)

    if(model_name != "SingleBest" and model_name != "StaticSelection"):

        for k in list_k:
            for pool in pools:
                for n in n_estimators_list:
                    pool_p = BaggingClassifier(
                        base_estimator=pool, n_estimators=n)
                    pool_p.fit(X_train, y_train)
                    acc, roc = cross_validation_model(
                        X_train, y_train, X_test, y_test, model_name, pool_p, k)
                    if(roc >= best_roc):
                        if(acc > best_acc):
                            best_acc = acc
                            best_roc = roc
                            best_k = k
                            best_pool = pool_p

        return return_model_by_name(model_name, best_pool, best_k)

    for pool in pools:
        for n in n_estimators_list:
            pool_p = BaggingClassifier(base_estimator=pool, n_estimators=n)
            pool_p.fit(X_train, y_train)
            acc, roc = cross_validation_model(
                X_train, y_train, X_test, y_test, model_name, pool_p, None)
            if(roc >= best_roc):
                if(acc > best_acc):
                    best_acc = acc
                    best_roc = roc
                    best_pool = pool_p
    return return_model_by_name(model_name, best_pool, None)
