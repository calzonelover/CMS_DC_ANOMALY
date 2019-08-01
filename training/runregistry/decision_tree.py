import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc


def main(
        selected_pd = "JetHT",
        interested_statuses = "hcal_hcal"
    ):
    pass