import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
# customize
from data.prompt_reco.setting import REDUCED_FEATURES, FEATURES, SELECT_PD
import data.prompt_reco.utility as utility

from model.reco.autoencoder import ( VanillaAutoencoder, SparseAutoencoder,
                                     ContractiveAutoencoder, VariationalAutoencoder )

COLORS = ('green', 'blue')
GROUP_LABELS = ('A', 'B')
HUMAN_LABELS = ('Good', 'Bad')

def main():
    # Setting
    is_reduced_data = True
    Autoencoder = VanillaAutoencoder
    test_model = "Vanilla"
    number_model = 1
    BS = 256

    N_FEATURES = len(REDUCED_FEATURES*7) if is_reduced_data else 2807
    # data
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

    # training
    print("Preparing dataset...")
    split = int(0.8*len(data))
    # train set
    df_train = data.iloc[:split].copy()
    X_train = df_train.iloc[:, 0:N_FEATURES]
    y_train = df_train["label"]
    # test set
    df_test = data.iloc[split:].copy()
    X_test = df_test.iloc[:, 0:N_FEATURES]
    y_test = df_test["label"]
    X_test = pd.concat([X_train[y_train == 1], X_test])
    y_test = pd.concat([y_train[y_train == 1], y_test])

    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    print("Training KMeans")
    # standardize data
    # transformer = StandardScaler()
    transformer = MinMaxScaler(feature_range=(0,1))
    transformer.fit(X_train.values)
    X_train = transformer.transform(X_train.values)
    X_test = transformer.transform(X_test.values)
    # X_train = normalize(X_train, norm='l1')

    ## combine
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # training
    kmeans_model = KMeans(n_clusters=2).fit(X)
    y_pred = kmeans_model.predict(X)
    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    # visualzie K-means
    fig, ax = plt.subplots()
    for i, group_label in enumerate(GROUP_LABELS):
        scat_data = principal_components[y_pred == i]
        ax.scatter(
            scat_data[:, 0], scat_data[:, 1], alpha=0.8,
            c = COLORS[i if i == 0 else 1],
            label = GROUP_LABELS[i]
        )
    ax.legend()
    plt.title('Clustering by K-Means, visual in Principal Basis (JetHT)')
    plt.savefig('JetHT_kmeans.png')

    # visual labeld 
    fig, ax = plt.subplots()
    for i, group_label in enumerate(GROUP_LABELS):
        scat_data = principal_components[y == i]
        ax.scatter(
            scat_data[:, 0], scat_data[:, 1], alpha=0.8,
            c = COLORS[i],
            label = HUMAN_LABELS[i]
        )
    ax.legend()
    plt.title('Labeled by Human, visual in Principal Basis (JetHT)')
    plt.savefig('JetHT_label.png')
    # visual One-Class SVM cutoff
    svm_model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    svm_model.fit(X_train)
    sampling_svm_dvs = -svm_model.decision_function(X)[:, 0]
    min_sampling_svm_dvs, max_sampling_svm_dvs = min(sampling_svm_dvs), max(sampling_svm_dvs)
    colors_svm_dvs = list(map(lambda x: [0.2, 1.0-((x-min_sampling_svm_dvs)/(max_sampling_svm_dvs-min_sampling_svm_dvs)), (x-min_sampling_svm_dvs)/(max_sampling_svm_dvs-min_sampling_svm_dvs)], sampling_svm_dvs))
    colors_svm_cutoff = list(map(lambda x: [0, 0, 0.8] if x > 20.0 else [0, 1.0, 0], sampling_svm_dvs))
    fig, ax = plt.subplots()
    ax.scatter(
        principal_components[:, 0], principal_components[:, 1],
        alpha=0.8,
        c = colors_svm_dvs
    )
    plt.title('Decision Value from SVM, visual in Principal Basis (JetHT)')
    plt.savefig('SVM_DCs.png')
    fig, ax = plt.subplots()
    ax.scatter(
        principal_components[:, 0], principal_components[:, 1],
        alpha=0.8,
        c = colors_svm_cutoff
    )
    plt.title('Applying cutoff in SVM, visual in Principal Basis (JetHT)')
    plt.savefig('SVM_cutoff.png')

    # visual autoencoder loss
    autoencoder = Autoencoder(
        input_dim = [N_FEATURES],
        summary_dir = "model/reco/summary",
        model_name = "{} model {}".format(test_model, number_model),
        batch_size = BS
    )
    autoencoder.restore()
    sampling_totalsd = autoencoder.get_sd(X, scalar=True)
    max_totalsd = max(sampling_totalsd)
    min_totalsd = min(sampling_totalsd)
    colors_cutoff = list(map(lambda x: [0, 0, 0.8] if x > 10.0 else [0, 1.0, 0], sampling_totalsd))
    colors_loss = list(map(lambda x: [0.2, 1.0-((x-min_totalsd)/(max_totalsd-min_totalsd)), (x-min_totalsd)/(max_totalsd-min_totalsd)], sampling_totalsd))
    fig, ax = plt.subplots()
    ax.scatter(
        principal_components[:, 0], principal_components[:, 1],
        alpha=0.8,
        c = np.log10(sampling_totalsd)
    )
    plt.title('Loss from AE data, testing set visual in Principal Basis (JetHT)')
    plt.savefig('JetHT_AE_loss.png')
    fig, ax = plt.subplots()
    ax.scatter(
        principal_components[:, 0], principal_components[:, 1],
        alpha=0.8,
        c = colors_cutoff,
    )
    plt.title('Applying cutoff in AE, testing set visual in Principal Basis (JetHT)')
    plt.savefig('JetHT_AE_cutoff.png')