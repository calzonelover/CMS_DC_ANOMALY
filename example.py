from training.new_reco import autoencoder, oneclass_svm

if __name__ == "__main__":
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        autoencoder.main(
            selected_pd=selected_pd,
            BS=2e16,
            EPOCHS = 1200,
            DATA_SPLIT_TRAIN = [0.25, 0.5, 0.75, 1.0],
        )