from training.new_reco import autoencoder
from training.new_reco import oneclass_svm
from training.new_reco import kmeans, random_feature_visual

if __name__ == "__main__":
    # kmeans.main()
    # random_feature_visual.main()
    # oneclass_svm.main()
    autoencoder.compute_ms_dist(selected_pd="ZeroBias")
    autoencoder.compute_ms_dist(selected_pd="JetHT")
    autoencoder.compute_ms_dist(selected_pd="EGamma")
    autoencoder.compute_ms_dist(selected_pd="SingleMuon")