from training.new_reco import autoencoder
from training.new_reco import oneclass_svm
from training.new_reco import kmeans, random_feature_visual

if __name__ == "__main__":
    # kmeans.main()
    # random_feature_visual.main()
    # oneclass_svm.main()
    autoencoder.main(selected_pd="ZeroBias")
    autoencoder.main(selected_pd="JetHT")
    autoencoder.main(selected_pd="EGamma")
    autoencoder.main(selected_pd="SingleMuon")