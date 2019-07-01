import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
# customize
from data.prompt_reco.setting import REDUCED_FEATURES, FEATURES, SELECT_PD
import data.prompt_reco.utility as utility

def main():
    # setting
    MODEL_NAME = 'Isolation_Forest'
    TEST_MODEL = 'Isolation_Forest 1'
    is_reduced_data = True
    data_preprocessing_mode = 'minmaxscalar' # ['standardize, 'normalize', 'minmaxscalar']
    SPLIT_DATA_IN_80 = [1.0 for i in range(10)] # 60% of data
    N_FEATURES = len(REDUCED_FEATURES*7) if is_reduced_data else 2807

    files = utility.get_file_list(chosed_pd=SELECT_PD) # choosing only ZeroBias
    feature_names = utility.get_feature_name(features=FEATURES)
    reduced_feature_names = utility.get_feature_name(features=REDUCED_FEATURES)
    data = pd.DataFrame(utility.get_data(files), columns=feature_names)
    data["run"] = data["run"].astype(int)
    data["lumi"] = data["lumi"].astype(int)
    data.drop(["_foo", "_bar", "_baz"], axis=1, inplace=True)
    if is_reduced_data:
        not_reduced_column = feature_names
        for intersected_elem in reduced_feature_names: not_reduced_column.remove(intersected_elem)
        data.drop(not_reduced_column, axis=1, inplace=True)
    data = data.sort_values(["run", "lumi"], ascending=[True,True])
    data = data.reset_index(drop=True)
    data["label"] = data.apply(utility.add_flags, axis=1)

    rng = np.random.RandomState(42)
    model_list = [IsolationForest(
        n_estimators=200,
        max_samples=512,
        contamination=0.01,
        random_state=rng,
        n_jobs=-1
    ) for i in range(10)]

    file_auc = open('report/reco/eval/roc_auc.txt', 'w')
    file_auc.write("model_name data_fraction roc_auc\n")
    for dataset_fraction, model, model_name in zip(
            np.array(SPLIT_DATA_IN_80),
            model_list,
            ["{} {}".format(MODEL_NAME, i+1) for i in range(len(model_list))]):
        print("{} , Chunk of Training Dataset fraction: {}".format(model_name, dataset_fraction))
        print("Train(Train,Valid) test split...")
        split = int(0.8*len(data))
        # train set
        df_train = data.iloc[:split].copy()
        split_frac = int(dataset_fraction*len(df_train))
        df_train_frac = data.iloc[:split_frac].copy()
        X_train = df_train_frac.iloc[:, 0:N_FEATURES]
        y_train = df_train_frac["label"]
        # test set
        df_test = data.iloc[split:].copy()
        X_test = df_test.iloc[:, 0:N_FEATURES]
        y_test = df_test["label"]
        X_test = pd.concat([X_train[y_train == 1], X_test])
        y_test = pd.concat([y_train[y_train == 1], y_test])
        # train only good condition
        X_train = X_train[y_train == 0]
        print("Number of inliers in training&valid set: {}".format(len(X_train)))
        print("Number of inliers in test set: {}".format(sum((y_test == 0).values)))
        print("Number of anomalies in the test set: {}".format(sum((y_test == 1).values)))
        # Data Preprocessing
        if data_preprocessing_mode == 'standardize':
            transformer = StandardScaler()
        elif data_preprocessing_mode == 'minmaxscalar':
            transformer = MinMaxScaler(feature_range=(0,1))
        transformer.fit(X_train)
        if data_preprocessing_mode == 'normalize':
            X_train = normalize(X_train, norm='l1')
            X_test = normalize(X_test, norm='l1')
        else:
            X_train = transformer.transform(X_train.values)
            X_test = transformer.transform(X_test.values)

        model.fit(X_train)
        try:
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        except FileNotFoundError:
            os.makedirs("./report/reco/eval/")
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        file_eval.write("fpr tpr threshold\n")
        fprs, tprs, thresholds = roc_curve(y_test, -model.decision_function(X_test))
        for fpt, tpr, threshold in zip(fprs, tprs, thresholds):
            file_eval.write("{} {} {}\n".format(fpt, tpr, threshold))
        file_eval.close()

        print("AUC {}".format(auc(fprs, tprs)))
        file_auc.write("{} {} {}\n".format(MODEL_NAME, dataset_fraction, auc(fprs, tprs)))

        # Decision Value log
        if model_name == TEST_MODEL:
            x_good = X_test[y_test == 0]
            run_good = df_test['run'][y_test == 0]
            lumi_good = df_test['lumi'][y_test == 0]
            x_bad = X_test[y_test == 1]
            run_bad = pd.concat([
                df_train['run'][df_train['label'] == 1],
                df_test['run'][df_test['label'] == 1]
            ])
            lumi_bad = pd.concat([
                df_train['lumi'][df_train['label'] == 1],
                df_test['lumi'][df_test['label'] == 1]
            ])
            print('sample', ' good: ',len(x_good), ' bad: ', len(x_bad))
            with open('{}_good_totalSE.txt'.format(TEST_MODEL), 'w') as f:
                f.write('total_se run lumi\n')
                for good_totalsd, run, lumi in zip(-model.decision_function(x_good), run_good, lumi_good):
                    f.write('{} {} {}\n'.format(good_totalsd, run, lumi))
            with open('{}_bad_totalSE.txt'.format(TEST_MODEL), 'w') as f:
                f.write('total_se run lumi\n')
                for bad_totalsd, run, lumi in zip(-model.decision_function(x_bad), run_bad, lumi_bad):
                    f.write('{} {} {}\n'.format(bad_totalsd, run, lumi))