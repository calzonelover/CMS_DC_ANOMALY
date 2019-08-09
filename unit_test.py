# from training.reco.k_means import k_means
# from data.new_prompt_reco import unit_test as test_data
# from data.new_prompt_reco import pull_label as pull_label_prompt_reco
from training.new_reco import kmeans

if __name__ == "__main__":
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        # pull_label_prompt_reco.main(selected_pd=selected_pd)
        kmeans.plot_subsystem(selected_pd=selected_pd)
    # kmeans.plot_bad_good_separate_case()