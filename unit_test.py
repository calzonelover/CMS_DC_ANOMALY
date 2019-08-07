# from training.reco.k_means import k_means
from data.new_prompt_reco import unit_test as test_data
# from data.runregistry.simplest import unit_test as test_simplest_rr
# from data.runregistry.simplest import extraction as extract_rr

# from data.prompt_reco import unit_test as prompt_reco_test
# from data.express_2017 import extract as express_2017_extract

if __name__ == "__main__":
    test_data.main(selected_pd="ZeroBias", include_failure=True)
    # test_data.main(selected_pd="JetHT", include_failure=True)
    # test_data.main(selected_pd="EGamma", include_failure=False)
    # test_data.main(selected_pd="SingleMuon", include_failure=False)