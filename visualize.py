from report.reco.new_data import global_plot

if __name__ == '__main__':
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]:
        global_plot.spectrum_component_weights(selected_pd=selected_pd)