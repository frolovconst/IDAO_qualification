import os
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import swifter
import utils
import scoring
import catboost
from time import time
import pickle
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import StratifiedKFold


def objective_zw(params, X, y, train_weights, real_weights):
    params['iterations'] = max(50, int(params['iterations']))
    global best_score
    global iter_no
    scores = []
    skf = StratifiedKFold(n_splits=5, random_state=442)
    model = catboost.CatBoostClassifier(verbose=False, task_type = "GPU", border_count=254, **params)
    for train_val_idx, test_val_idx in skf.split(X, y):
        train_concat_val = X.loc[train_val_idx]
        train_labels_val = y.loc[train_val_idx]
        test_concat_val = X.loc[test_val_idx]
        test_labels_val = y.loc[test_val_idx]
        model.fit(train_concat_val,
                  train_labels_val,
                  sample_weight=train_weights.loc[train_val_idx],
                  plot=True,
                  eval_set=[(test_concat_val, test_labels_val)])
        validation_predictions = model.predict_proba(test_concat_val.loc[test_val_idx].values)[:, 1]
        fold_score = scoring.rejection90(test_labels_val.values, validation_predictions, sample_weight=real_weights.loc[test_val_idx].values)
        scores.append(fold_score)
    mean_fold_score = np.mean(scores)
    with open('../data/hyper_tr2_zw.log', 'a') as f:
        f.write(f'iter {iter_no}\n')
        f.write(str(mean_fold_score))
        f.write('\n')
        f.write(str(params))
        f.write('\n\n')
    if mean_fold_score > best_score:
        best_score = mean_fold_score
    iter_no += 1
    return -mean_fold_score


def main():
    global best_score
    global iter_no
    DATA_PATH = "../data"
    summ_dict = {}

    full_train = utils.load_train_hdf(DATA_PATH)
    if os.path.isfile(DATA_PATH + '/closest_hits_train_bsln.csv'):
        closest_hits_features = pd.read_csv(DATA_PATH + '/closest_hits_train_bsln.csv')
    else:
        s_time = time()
        closest_hits_features = full_train.swifter.apply(
        utils.find_closest_hit_per_station, result_type="expand", axis=1)
        summ_dict['tr_ftr_time'] = time()-s_time

    train_concat = pd.concat(
        [full_train.loc[:, utils.SIMPLE_FEATURE_COLUMNS],
         closest_hits_features], axis=1)

    real_weights = full_train.weight
    train_weights = pd.Series(np.where(real_weights<0, 0, real_weights))

    try:
        summ_dict
    except NameError:
        summ_dict = {}
    with open(DATA_PATH + '/hyper_tr2_zw.log', 'w') as f:
        pass
    s_time = time()
    max_hyperopt_evals = 70
    space = {
        'learning_rate': hp.loguniform('learning_rate', -6.91, -2.38),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', -2.3, 4.6),
        'iterations': hp.quniform('iterations', 200, 700, 30),
        'depth': hp.quniform('depth', 6, 11, 1)
        }
    best = fmin(fn=lambda params: objective_zw(params, train_concat, full_train.label, train_weights, real_weights),
                space=space,
                algo=tpe.suggest,
                max_evals=max_hyperopt_evals
               )
    summ_dict['best_score'] = best_score
    summ_dict[f'hyperopt_{max_hyperopt_evals}_times_time'] = time()-s_time
    with open(DATA_PATH+'/best_params_track2_zw.pkl', 'wb') as f:
        pickle.dump(best, f)

    # During the tuning process, the best CV score was achieved with the following params:
    best={'depth': 10.0,
     'iterations': 700,
     'l2_leaf_reg': 52.543809823589136,
     'learning_rate': 0.05139084422974949}

    model = catboost.CatBoostClassifier(verbose=False, task_type="GPU", border_count=254, **best)
    s_time = time()
    model.fit(train_concat, full_train.label, sample_weight=train_weights)
    summ_dict['fit_whole_train_time'] = time()-s_time
    with open(DATA_PATH+'/summary_zw2.pkl', 'wb') as f:
        pickle.dump(summ_dict, f)
    model.set_params(task_type='CPU', thread_count=-1)
    model.save_model("submission/track_2_model.cbm")
    os.system('zip subm.zip submission/*')

    
if __name__ == '__main__':
    iter_no = 1
    best_score = .0
    main()
    