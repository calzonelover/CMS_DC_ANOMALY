from training.reco.k_means import k_means
from data.new_prompt_reco import unit_test as test_data
from data.runregistry.simplest import unit_test as test_simpleast_rr
from data.runregistry.simplest import extraction as extract_rr

from data.prompt_reco import unit_test as prompt_reco_test

if __name__ == "__main__":
    # extract_rr.main(selected_pd = "ZeroBias")
    # extract_rr.main(selected_pd = "JetHT")
    # extract_rr.main(selected_pd = "EGamma")
    # extract_rr.main(selected_pd = "SingleMuon")
    test_simpleast_rr.get_fraction()