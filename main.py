import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools.pipeline import Pipeline
from upload_data.data import from_zip_dataset_to_numpy

st.title(
    "Unsupervised classification of plethysmography signals with advanced visual representations - A demonstration."
)

# Inputs

zip_file = st.sidebar.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    zip_file = zipfile.ZipFile(zip_file, "r")
    file_array = from_zip_dataset_to_numpy(zip_file)

    labels_df = st.sidebar.file_uploader(
        "Upload a csv file with your labels to access visualization by label",
        type="csv",
    )
    if labels_df is not None:
        labels_df = pd.read_csv(labels_df)
        dict_labels = {}
        for col in labels_df.columns:
            dict_labels[col] = list(labels_df[col].unique())
        st.write(dict_labels)

# Parameters

NJOBS = 1
VERBOSE = True
form = st.sidebar.form("Choose parameters")
form.write("Choose Parameters")
SAMPFREQ = form.number_input("Sampling frequency", step=100, value=2000)
N_CLUSTER = form.slider("Number of clusters", min_value=1, max_value=5, value=3)
N_IN_CLUSTER = N_OUT_CLUSTER = N_CLUSTER
# if st.sidebar.checkbox("Advanced parameters"):

# advanced parameters
expander = form.expander("Advanced parameters")
DOWN_SAMPFREQ = expander.number_input("Downsampling frequency", step=50, value=250)
PROMINENCE = expander.number_input("Prominence", step=0.01, value=0.03)
WLEN = expander.number_input("Wlen", step=1, value=2)
MIN_CYCLE = expander.number_input("Min_cycle", step=0.1, value=0.1)
MAX_CYCLE = expander.number_input("Max_cycle", step=0.1, value=0.3)
TRAINING_SIZE = expander.number_input("Training size", step=1, value=30)
INTERVAL = expander.number_input("Interval", step=1, value=60)
IN_D = expander.number_input("IN_D", step=0.1, value=0.2)
OUT_D = expander.number_input("OUT_D", step=0.1, value=0.2)
N_ITER = expander.number_input("nb_iter", step=1, value=10)
MAX_WARPING = expander.number_input(
    "max_warping", step=0.01, value=0.06, min_value=0.0, max_value=1.0
)
QUANTILE = expander.number_input(
    "quantile", step=0.01, value=0.95, min_value=0.0, max_value=1.0
)
apply_params = form.form_submit_button("Apply parameters")
# Run experiment

st.write(
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
if zip_file is not None:
    if labels_df is None:
        st.subheader("Display datas")
        st.write("Enter a csv file to access advanced research parameters")
        number_of_files = file_array.size
        signal_idx = st.selectbox(
            "Choose the signal you want to display", np.arange(number_of_files)
        )
        # downsampling of the TS
        freq_ratio_ = SAMPFREQ // DOWN_SAMPFREQ
        considered_ts = file_array[signal_idx][::freq_ratio_]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(considered_ts.shape[0])), y=considered_ts)
        )

        # Set title
        fig.update_layout(title_text="Time series number " + " " + str(signal_idx))

        st.plotly_chart(fig)

if st.sidebar.button("Run"):
    if zip_file is None:
        st.sidebar.write("Please input a zip archive first")
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
        st.subheader("Clustering results")
        st.plotly_chart(pipe.plot_medoid(), theme=None)
