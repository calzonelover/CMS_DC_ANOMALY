# from training.new_reco import autoencoder
# from training.new_reco import oneclass_svm
from training.new_reco import kmeans, random_feature_visual
from data.new_prompt_reco import pull_label as pull_label_prompt_reco

# from training.reco.autoencoder import sparse
# from training.reco.k_means import k_means

# from training.runregistry import decision_tree

if __name__ == "__main__":
    kmeans.main()
    # for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
    #     pull_label_prompt_reco.main(selected_pd=selected_pd)