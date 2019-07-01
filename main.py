from training.reco.autoencoder import sparse
from training.reco.unsupervised import isolation_forest, oneclass_svm

if __name__ == "__main__":
    oneclass_svm.main()