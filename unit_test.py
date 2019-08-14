# from training.reco.k_means import k_means
from data.new_prompt_reco import unit_test as test_prompt_reco_data
# from data.new_prompt_reco import pull_label as pull_label_prompt_reco
# from training.new_reco import kmeans

if __name__ == "__main__":
    # apply filter 500
    # kmeans.plot_subsystem3d(selected_pd="JetHT")
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        test_prompt_reco_data.main(selected_pd=selected_pd)
        # pull_label_prompt_reco.main(selected_pd=selected_pd)
        # kmeans.plot_subsystem(selected_pd=selected_pd)
    # kmeans.plot_bad_good_separate_case()