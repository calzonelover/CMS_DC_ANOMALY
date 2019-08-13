from training.new_reco import autoencoder, oneclass_svm

if __name__ == "__main__":
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        autoencoder.main(selected_pd=selected_pd)