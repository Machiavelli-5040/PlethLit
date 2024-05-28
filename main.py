import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from tools.pipeline import Pipeline
from upload_data.data import from_zip_dataset_to_numpy

st.title(
    "Unsupervised classification of plethysmography signals with advanced visual representations - A demonstration."
)

zip_file = st.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    file_array = from_zip_dataset_to_numpy(zipfile.ZipFile(zip_file, "r"))

labels_file = st.file_uploader("Upload a csv file of the labels", type="csv")


DOWN_SAMPFREQ = 250
PROMINENCE = 0.03
WLEN = 2
MIN_CYCLE = 0.1
MAX_CYCLE = 0.3
TRAINING_SIZE = 30
INTERVAL = 60
IN_D = 0.2
OUT_D = 0.2
N_ITER = 10
MAX_WARPING = 0.06
QUANTILE = 0.95
NJOBS = 1
VERBOSE = True

SAMPFREQ = st.sidebar.number_input("Sampling frequency", step=100, value=2000)
N_CLUSTER = st.sidebar.slider("Number of clusters", min_value=1, max_value=5, value=3)
N_IN_CLUSTER = N_OUT_CLUSTER = N_CLUSTER
if st.sidebar.checkbox("Advanced parameters"):
    DOWN_SAMPFREQ = st.sidebar.number_input(
        "Downsampling frequency", step=50, value=250
    )
    PROMINENCE = st.sidebar.number_input("Prominence", step=0.01, value=0.03)


if st.button("Run"):
    if zip_file is None:
        st.write("You have to enter a file first")
    else:
        pipe = Pipeline(
            SAMPFREQ,
            PROMINENCE,
            WLEN,
            MIN_CYCLE,
            MAX_CYCLE,
            TRAINING_SIZE,
            INTERVAL,
            N_IN_CLUSTER,
            IN_D,
            N_OUT_CLUSTER,
            OUT_D,
            DOWN_SAMPFREQ,
            MAX_WARPING,
            N_ITER,
            QUANTILE,
            NJOBS,
            VERBOSE,
        )
        pipe.fit(file_array)
        fig = pipe.plot_medoid()
        st.write(fig)
