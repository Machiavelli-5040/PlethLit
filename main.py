import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state

from tools.pipeline import Pipeline
from tools.utils import qrcode_plot, symbolic_representation
from upload_data.data import from_zip_dataset_to_numpy

st.set_page_config(layout="wide")
st.title(
    "Unsupervised classification of plethysmography signals with advanced visual representations - A demonstration"
)
tab_0, tab_1, tab_2, tab_3 = st.tabs(
    [
        "Guidelines",
        "Individual representation",
        "Collective representation",
        "Representative respiratory cycles",
    ]
)
st.sidebar.subheader("Select files and parameters")
tab_0.subheader("Guidelines")
tab_1.subheader("Individual representation")
tab_2.subheader("Collective representation")
tab_3.subheader("Representative respiratory cycles")

# Session state
if "pipe" not in state:
    state.pipe = None


# Sidebar ======================================================================


# Inputs
zip_file = st.sidebar.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    zip_file = zipfile.ZipFile(zip_file, "r")
    file_array = from_zip_dataset_to_numpy(zip_file)

    labels_df = st.sidebar.file_uploader(
        "Upload a csv file with your labels to access advanced visualization",
        type="csv",
    )
    if labels_df is not None:
        labels_df = pd.read_csv(labels_df)

        dict_labels = {}
        for col in labels_df.columns:
            dict_labels[col] = list(labels_df[col].unique())

        if set(zip_file.namelist()) != set(dict_labels["filename"]):
            st.write("WARNING: Filenames in the zip and in the csv do not match")

# Basic parameters
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

apply_params = form.form_submit_button("Apply parameters")  # Save parameters
run = st.sidebar.button("Run")  # Run experiment

# Run
if run:
    if zip_file is None:
        st.sidebar.write("Please input a zip archive first")
    else:
        state.pipe = Pipeline(
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
        state.pipe.fit(file_array)


# Guidelines ===================================================================


with tab_0:
    st.write("To do!")
    st.image("meme.jpg")


# Individual ===================================================================


with tab_1:
    if zip_file is None:
        st.write("Please input a zip file to visualize signals.")
    else:
        if labels_df is None:
            signal_idx = st.selectbox(
                "Choose the signal you want to display:",
                list(range(len(file_array))),
                format_func=lambda i: zip_file.namelist()[i],
            )
        else:
            filtering_expander = st.expander("Advanced Research")
            filtering_columns = filtering_expander.multiselect(
                "Columns to filter",
                list(labels_df.columns[1:]),
                default=list(labels_df.columns[1:]),
            )
            filtering_expander.write("Please select to features to keep below:")
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

            signal_idx = st.selectbox(
                "Choose the signal you want to display (the list of signals will be tuned according to the chosen parameters above):",
                labels_df_copy.index,
                format_func=lambda i: zip_file.namelist()[i],
            )
        # Downsampling
        freq_ratio_ = SAMPFREQ // DOWN_SAMPFREQ
        considered_ts = file_array[signal_idx][::freq_ratio_]
        fig_signal = go.Figure()
        fig_signal.add_trace(
            go.Scatter(x=list(range(len(considered_ts))), y=considered_ts)
        )

        # Plots
        if state.pipe is None:
            st.plotly_chart(fig_signal)
        else:
            preds = json.loads(
                state.pipe.json_predictions_,
            )
            preds_df = pd.read_json(preds[signal_idx], orient="columns")
            total_duration = len(considered_ts)
            duration_array = (
                np.concatenate(
                    (preds_df["in_start_index"][1:].to_numpy(), [total_duration])
                )
                - preds_df["in_start_index"].to_numpy()
            )

            fig_in, fig_out = symbolic_representation(
                preds_df["in_cluster"],
                preds_df["out_cluster"],
                duration_array,
                N_IN_CLUSTER,
                N_OUT_CLUSTER,
            )

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            for i in fig_signal.data:
                fig.add_trace(i, row=1, col=1)
            for i in fig_in.data:
                fig.add_trace(i, row=2, col=1)
            for i in fig_out.data:
                fig.add_trace(i, row=3, col=1)
            st.plotly_chart(fig)

            qrcode = np.zeros((N_IN_CLUSTER, N_OUT_CLUSTER))
            for idx_in, idx_out, duration in zip(
                preds_df["in_cluster"], preds_df["out_cluster"], duration_array
            ):
                qrcode[idx_in, idx_out] += duration
            qrcode /= np.sum(qrcode)

            in_labels = [chr(x) for x in range(ord("A"), ord("A") + N_CLUSTER)]
            out_labels = list(range(N_CLUSTER))
            qrcode_plot(in_labels, out_labels, qrcode)


# Collective ===================================================================


with tab_2:
    if state.pipe is None:
        st.write("Please run the demonstration first to output results here.")


# Representative cycles ========================================================


with tab_3:
    if state.pipe is None:
        st.write("Please run the demonstration first to output results here.")
    else:
        st.plotly_chart(state.pipe.plot_medoid(), theme=None)
