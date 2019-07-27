# from training.new_reco import autoencoder
# from training.new_reco import oneclass_svm
# from training.new_reco import kmeans, random_feature_visual

from training.reco.autoencoder import sparse
from training.reco.k_means import k_means

if __name__ == "__main__":
    # kmeans.main()
    # random_feature_visual.main()
    # oneclass_svm.main()
    k_means.main()