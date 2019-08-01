from training.reco.k_means import k_means
from data.new_prompt_reco import unit_test as test_data
# from data.runregistry.simplest import unit_test as test_simpleast_rr
# from data.runregistry.simplest import extraction as extract_rr

# from data.prompt_reco import unit_test as prompt_reco_test
from data.express_2017 import unit_test as express_2017_test

if __name__ == "__main__":
    express_2017_test.main()