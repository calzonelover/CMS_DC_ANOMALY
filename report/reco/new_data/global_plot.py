import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc
from math import sqrt, isnan
import os

def spectrum_component_weights(
        path_dat='report/reco/new_data/logs/minmaxscalar/',
        selected_pd = 'JetHT',
        n_first_component = 20,
    ):
    for axis in [1, 2]:
        df_pc = pd.read_csv(os.path.join(path_dat, "pc_{}.csv".format(selected_pd))).query("axis == {}".format(axis))
        plot_x1, plot_y1 = (df_pc['feature'][:n_first_component], df_pc['component'][:n_first_component])
        fig, ax = plt.subplots()
        plt.plot(plot_x1, plot_y1)
        plt.title('{} largest absolute weight in {} principal component ({})'.format(n_first_component, axis,selected_pd))    
        plt.yscale('log')
        fig.autofmt_xdate()
        plt.savefig(os.path.join(path_dat, '{}_pc{}.png'.format(selected_pd, axis)), bbox_inches='tight')