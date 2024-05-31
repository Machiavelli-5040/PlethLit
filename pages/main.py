import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools.pipeline import Pipeline
from tools.utils import qrcode_plot, symbolic_representation
from upload_data.data import from_zip_dataset_to_numpy

st.title("Results")

# Inputs
st.sidebar.subheader("Select files and parameters")
zip_file = st.sidebar.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    zip_file = zipfile.ZipFile(zip_file, "r")
    file_array = from_zip_dataset_to_numpy(zip_file)

    labels_df = st.sidebar.file_uploader(
        "Upload a csv file with your labels to access advanced visualization.",
        type="csv",
    )
    if labels_df is not None:
        labels_df = pd.read_csv(labels_df)
        # Remove eventual manual index column
        labels_df = labels_df.drop(["Unnamed: 0"], axis=1, errors="ignore")

        dict_labels = {}
        for col in labels_df.columns:
            dict_labels[col] = list(labels_df[col].unique())

        if set(zip_file.namelist()) != set(dict_labels["filename"]):
            st.write("WARNING: Filenames in the zip and in the csv do not match")

# Parameters

NJOBS = 1
VERBOSE = True
form = st.sidebar.form("Choose parameters")
form.write("Choose Parameters")
SAMPFREQ = form.number_input("Sampling frequency", step=100, value=2000)
DOWN_SAMPFREQ = form.number_input("Downsampling frequency", step=50, value=250)
N_CLUSTER = form.slider("Number of clusters", min_value=1, max_value=5, value=3)
N_IN_CLUSTER = N_OUT_CLUSTER = N_CLUSTER

# Advanced parameters
expander = form.expander("Advanced parameters")
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
run = st.sidebar.button("Run")

if zip_file is not None:
    tab_1, tab_2, tab_3 = st.tabs(
        [
            "Individual representation",
            "Collective representation",
            "Representative respiratory cycles",
        ]
    )
    if labels_df is None:
        tab_1.subheader("Indiviudal representation")
        signal_idx = tab_1.selectbox(
            "Choose the signal you want to display",
            list(range(len(file_array))),
            format_func=lambda i: zip_file.namelist()[i],
        )
        # Downsampling of the TS
        freq_ratio_ = SAMPFREQ // DOWN_SAMPFREQ
        considered_ts = file_array[signal_idx][::freq_ratio_]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(considered_ts.shape[0])), y=considered_ts)
        )
        tab_1.plotly_chart(fig)

    else:

        tab_1.subheader("Individual representation")
        filtering_expander = tab_1.expander("Advanced Research")
        filtering_columns = filtering_expander.multiselect(
            "Columns to filter",
            list(labels_df.columns[1:]),
            default=list(labels_df.columns[1:]),
        )
        filtering_params = {}
        for param_name in filtering_columns:
            filtering_params[param_name] = filtering_expander.multiselect(
                f"{param_name}",
                dict_labels[param_name],
                default=dict_labels[param_name],
            )
        labels_df_copy = labels_df.copy(deep=True)
        for key, value in filtering_params.items():
            labels_df_copy = labels_df_copy[labels_df_copy[key].isin(value)]

        signal_idx = tab_1.selectbox(
            "Choose the signal you want to display",
            labels_df_copy.index,
            format_func=lambda i: zip_file.namelist()[i],
        )

        # Downsampling of the TS
        freq_ratio_ = SAMPFREQ // DOWN_SAMPFREQ
        considered_ts = file_array[signal_idx][::freq_ratio_]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(considered_ts.shape[0])), y=considered_ts)
        )
        st.plotly_chart(fig)

if run:
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

        preds = json.loads(
            pipe.json_predictions_,
        )
        preds_df = pd.read_json(preds[signal_idx], orient="columns")
        total_duration = considered_ts.shape[0]
        duration_array = (
            np.concatenate(
                (preds_df["in_start_index"][1:].to_numpy(), [total_duration])
            )
            - preds_df["in_start_index"].to_numpy()
        )
        with tab_1:
            symbolic_representation(
                preds_df["in_cluster"],
                preds_df["out_cluster"],
                duration_array,
                total_duration,
                N_IN_CLUSTER,
                N_OUT_CLUSTER,
            )
            qrcode = np.zeros((N_IN_CLUSTER, N_OUT_CLUSTER))
            for idx_in, idx_out, duration in zip(
                preds_df["in_cluster"], preds_df["out_cluster"], duration_array
            ):
                qrcode[idx_in, idx_out] += duration
            qrcode /= np.sum(qrcode)

            in_labels = [chr(x) for x in range(ord("A"), ord("A") + N_CLUSTER)]
            out_labels = [x for x in range(N_CLUSTER)]
            qrcode_plot(in_labels, out_labels, qrcode)

        tab_2.subheader("Collective representation")
        with tab_3:
            st.subheader("Clustering results")
            st.plotly_chart(pipe.plot_medoid(), theme=None)
