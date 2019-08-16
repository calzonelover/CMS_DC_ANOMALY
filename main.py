from training.new_reco import autoencoder
from training.new_reco import oneclass_svm
# from training.new_reco import kmeans, random_feature_visual

# from training.reco.autoencoder import sparse
# from training.reco.k_means import k_means

# from training.runregistry import decision_tree

if __name__ == "__main__":
    '''
        Need to train EGamma when FailureScenario is arrived
    ''' 
    for selected_pd in ["ZeroBias", "JetHT", "SingleMuon"]:
        autoencoder.main(selected_pd=selected_pd, BS = 2**16, EPOCHS = 2200, include_bad_failure=True, cutoff_eventlumi=True)
    # for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
    #     oneclass_svm.main(selected_pd=selected_pd, cutoff_eventlumi=True)