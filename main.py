from training.new_reco import autoencoder
from training.new_reco import oneclass_svm
# from training.new_reco import kmeans, random_feature_visual

# from training.reco.autoencoder import sparse
# from training.reco.k_means import k_means

# from training.runregistry import decision_tree

if __name__ == "__main__":
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        autoencoder.main(selected_pd=selected_pd)
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        oneclass_svm.main(selected_pd=selected_pd)