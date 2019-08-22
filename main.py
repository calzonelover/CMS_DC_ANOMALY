
from model.reco import new_autoencoder as Autoencoder
from training.new_reco import autoencoder
# from training.new_reco import oneclass_svm
# from training.new_reco import kmeans, random_feature_visual

# from training.reco.autoencoder import sparse
# from training.reco.k_means import k_means

# from training.runregistry import decision_tree

if __name__ == "__main__":
    '''
        Need to train EGamma when FailureScenario is arrived
    ''' 
    # for selected_pd in ["ZeroBias", "JetHT", "SingleMuon"]:
    #     autoencoder.main(
    #         selected_pd=selected_pd,
    #         BS = 2**16, EPOCHS = 1200,
    #         include_bad_failure=True,
    #         cutoff_eventlumi=True,
    #         DATA_SPLIT_TRAIN = [1.0 for i in range(5)],
    #     )
    ## Testing
    '''
    autoencoder.compute_ms_dist(
        selected_pd = "ZeroBias",
        Autoencoder = Autoencoder.VariationalAutoencoder,
        model_name = "Variational",
        number_model = 1,
        include_bad_failure = True,
        cutoff_eventlumi = True,
        is_dropna = True,
        data_preprocessing_mode = 'minmaxscalar',
        gpu_memory_growth = True,
    )
    autoencoder.compute_ms_dist(
        selected_pd = "JetHT",
        Autoencoder = Autoencoder.VariationalAutoencoder,
        model_name = "Variational",
        number_model = 1,
        include_bad_failure = True,
        cutoff_eventlumi = True,
        is_dropna = True,
        data_preprocessing_mode = 'minmaxscalar',
        gpu_memory_growth = True,
    )
    autoencoder.compute_ms_dist(
        selected_pd = "SingleMuon",
        Autoencoder = Autoencoder.VanillaAutoencoder,
        model_name = "Vanilla",
        number_model = 1,
        include_bad_failure = True,
        cutoff_eventlumi = True,
        is_dropna = True,
        data_preprocessing_mode = 'minmaxscalar',
        gpu_memory_growth = True,
    )
    '''
    # autoencoder.error_features(
    #     selected_pd = "ZeroBias",
    #     Autoencoder = Autoencoder.VariationalAutoencoder,
    #     model_name = "Variational",
    #     number_model = 1,
    #     include_bad_failure = True,
    #     cutoff_eventlumi = True,
    #     is_dropna = True,
    #     data_preprocessing_mode = 'minmaxscalar',
    #     gpu_memory_growth = True,
    # )
    # autoencoder.error_features(
    #     selected_pd = "JetHT",
    #     Autoencoder = Autoencoder.VariationalAutoencoder,
    #     model_name = "Variational",
    #     number_model = 1,
    #     include_bad_failure = True,
    #     cutoff_eventlumi = True,
    #     is_dropna = True,
    #     data_preprocessing_mode = 'minmaxscalar',
    #     gpu_memory_growth = True,
    # )
    autoencoder.error_features(
        selected_pd = "SingleMuon",
        Autoencoder = Autoencoder.VanillaAutoencoder,
        model_name = "Vanilla",
        number_model = 1,
        include_bad_failure = True,
        cutoff_eventlumi = True,
        is_dropna = True,
        data_preprocessing_mode = 'minmaxscalar',
        gpu_memory_growth = True,
    )