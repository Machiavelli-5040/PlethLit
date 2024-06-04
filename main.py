import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state

from tools.pipeline import Pipeline
from tools.utils import (
    custom_multiselect,
    custom_qrcode,
    qrcode_plot,
    symbolic_representation,
)
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
    ordered_filenames = sorted(zip_file.namelist())
    file_array = from_zip_dataset_to_numpy(zip_file)

    labels_df = st.sidebar.file_uploader(
        "Upload a csv file with your labels to access advanced visualization",
        type="csv",
    )
    if labels_df is not None:
        labels_df = (
            pd.read_csv(labels_df).sort_values(by=["filename"]).reset_index(drop=True)
        )

        dict_labels = {}
        for col in labels_df.columns:
            dict_labels[col] = list(labels_df[col].unique())

        if set(zip_file.namelist()) != set(dict_labels["filename"]):
            st.write("WARNING: Filenames in the zip and in the csv do not match")

# Parameters
params = {}
params["njobs"] = 1
params["verbose"] = True
form = st.sidebar.form("parameters")
form.write("Choose parameters")
# Basic parameters
params["sampfreq"] = form.number_input("Sampling frequency", step=100, value=2000)
params["down_sampfreq"] = form.number_input(
    "Downsampling frequency", step=50, value=250
)
params["in_ncluster"] = form.slider(
    "Number of inhalation clusters", min_value=1, max_value=5, value=3
)
params["out_ncluster"] = form.slider(
    "Number of exhalation clusters", min_value=1, max_value=5, value=3
)
# Advanced parameters
expander = form.expander("Advanced parameters")
params["prominence"] = expander.number_input("Prominence", step=0.01, value=0.03)
params["wlen"] = expander.number_input("Wlen", step=1, value=2)
params["cycle_minimum_duration"] = expander.number_input(
    "Cycle minimum duration", step=0.1, value=0.1
)
params["cycle_maximum_duration"] = expander.number_input(
    "Cycle maximum duration", step=0.1, value=0.3
)
params["training_size_per_interval"] = expander.number_input(
    "Training size per interval", step=1, value=30
)
params["interval"] = expander.number_input("Interval", step=1, value=60)
params["in_centroid_duration"] = expander.number_input(
    "In centroid duration", step=0.1, value=0.2
)
params["out_centroid_duration"] = expander.number_input(
    "Out centroid duration", step=0.1, value=0.2
)
params["n_iteration"] = expander.number_input("Number of iterations", step=1, value=10)
params["radius"] = expander.number_input(
    "Max warping/radius", step=0.01, value=0.06, min_value=0.0, max_value=1.0
)
params["quantile_threshold"] = expander.number_input(
    "Quantile threshold", step=0.01, value=0.95, min_value=0.0, max_value=1.0
)
form.form_submit_button("Apply parameters")  # Save parameters
run = st.sidebar.button("Run")  # Run experiment

# Run
if run:
    if zip_file is None:
        st.sidebar.write("Please input a zip archive first")
    else:
        state.pipe = Pipeline(**params)
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
                format_func=lambda i: ordered_filenames[i],
            )
        else:
            labels_df_indv = custom_multiselect(
                labels_df, dict_labels, "Advanced research", "indv"
            )
            signal_idx = st.selectbox(
                "Choose the signal you want to display (the list of signals will be tuned according to the chosen parameters above):",
                labels_df_indv.index,
                format_func=lambda i: ordered_filenames[i],
            )
        # Downsampling
        freq_ratio_ = params["sampfreq"] // params["down_sampfreq"]
        considered_ts = file_array[signal_idx][::freq_ratio_]
        fig_signal = go.Figure(
            go.Scatter(
                x=list(range(len(considered_ts))),
                y=considered_ts,
                showlegend=False,
                name="",
            )
        )

        # Plots
        if state.pipe is None:
            st.plotly_chart(fig_signal)
        else:
            preds = json.loads(
                state.pipe.json_predictions_,
            )
            try:
                preds_df = pd.read_json(preds[signal_idx], orient="columns")
            except TypeError:
                st.write("Please select al least 1 value for each parameter.")
            else:
                total_duration = len(considered_ts)
                duration_array = (
                    np.concatenate(
                        (preds_df["in_start_index"][1:].to_numpy(), [total_duration])
                    )
                    - preds_df["in_start_index"].to_numpy()
                )

                # Signal visualization
                fig_in, fig_out = symbolic_representation(
                    preds_df["in_cluster"],
                    preds_df["out_cluster"],
                    duration_array,
                    params["in_ncluster"],
                    params["out_ncluster"],
                )

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0
                )
                for idx, subfig in enumerate([fig_signal, fig_in, fig_out]):
                    for i in subfig.data:
                        fig.add_trace(i, row=idx + 1, col=1)
                for i, s in enumerate(["in", "out"]):
                    next(fig.select_yaxes(row=2 + i, col=1)).update(
                        labelalias={1: f"{s}"}
                    )
                fig.update_xaxes(range=[0, len(considered_ts)])
                st.plotly_chart(fig)

                # QR code
                qrcode = np.zeros((params["in_ncluster"], params["out_ncluster"]))
                for idx_in, idx_out, duration in zip(
                    preds_df["in_cluster"], preds_df["out_cluster"], duration_array
                ):
                    if idx_in != -1 and idx_out != -1:
                        qrcode[idx_in, idx_out] += duration
                qrcode /= np.sum(qrcode)
                in_labels = [
                    chr(x) for x in range(ord("A"), ord("A") + params["in_ncluster"])
                ]
                out_labels = list(range(params["out_ncluster"]))
                qrcode_plot(in_labels, out_labels, qrcode)


# Collective ===================================================================


with tab_2:
    if state.pipe is None or labels_df is None:
        st.write(
            "Please run the demonstration first and provide a csv file with the labels to output results here."
        )
    else:
        preds = json.loads(
            state.pipe.json_predictions_,
        )
        col1, col2 = st.columns(2)
        with col1:
            labels_df_coll_1 = custom_multiselect(
                labels_df, dict_labels, "Parameter select", "coll_1"
            )
            if not len(labels_df_coll_1.index):
                st.write("Please select al least 1 value for each parameter.")
            else:
                custom_qrcode(preds, params, file_array, labels_df_coll_1.index)

        with col2:
            labels_df_coll_2 = custom_multiselect(
                labels_df, dict_labels, "Parameter select", "coll_2"
            )
            if not len(labels_df_coll_2.index):
                st.write("Please select al least 1 value for each parameter.")
            else:
                custom_qrcode(preds, params, file_array, labels_df_coll_2.index)


# Representative cycles ========================================================


with tab_3:
    if state.pipe is None:
        st.write("Please run the demonstration first to output results here.")
    else:
        st.plotly_chart(state.pipe.plot_medoid(), theme=None)
