from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import Parallel, delayed
from matplotlib.colors import rgb2hex


def cdist(
    measure: Callable, dataset1: np.ndarray, dataset2=None, diagonal=False, njobs=1
) -> np.ndarray:
    """Compute cross distance matrix

    Args:
        measure (Callable): Distance measure to use
        dataset1 (np.ndarray): formated sequence dataset.
        dataset2 (np.ndarray, optional): formated sequence dataset. If None Dataset 1 is duplicated. Defaults to None.
        diagonal (bool, optional): True to compute diagonal element. Defaults to False.
        njobs (int, optional): number of core to use. Defaults to 1.

    Returns:
        np.ndarray: cross distance matrix
    """
    m = len(dataset1)
    if dataset2 is None:
        cdist_array = np.zeros((m, m))
        idx1 = np.triu_indices(m, 1 - int(diagonal))
        cdist_array[idx1] = Parallel(n_jobs=njobs, prefer="threads")(
            delayed(measure.distance)(dataset1[i], dataset1[j])
            for i in range(m)
            for j in range(i + 1 - int(diagonal), m)
        )
        idx2 = np.tril_indices(m, -1)
        cdist_array[idx2] = cdist_array.T[idx2]
        return cdist_array
    else:
        cdist_array = Parallel(n_jobs=njobs, prefer="threads")(
            delayed(measure.distance)(s1, s2) for s1 in dataset1 for s2 in dataset2
        )
        return np.array(cdist_array).reshape(m, -1)


def to_time_series_dataset(data: list) -> np.ndarray:
    """Transform a list of sequences to a format dataset

    Args:
        data (list): list of sequences [ts_1,ts_2,....] where ts_i is a np.ndarray of shape (N_i,) or (N_i,1)

    Returns:
        np.ndarray: object array of sequences of shape (N,1).
    """
    tdata = [to_time_series(ts) for ts in data]
    sizes = [ts.shape[0] for ts in data]
    if (len(tdata) > 1) and not (np.all(np.array(sizes) == sizes[0])):
        return np.array(tdata, dtype=object)
    else:
        return np.array(tdata, dtype=float)


def to_time_series(ts: np.ndarray) -> np.ndarray:
    if ts.ndim == 1:
        return ts.reshape(-1, 1).astype(float)
    else:
        return ts.astype(float)


def symbolic_representation(
    in_pred,
    out_pred,
    duration_arr,
    n_in_cluster,
    n_out_cluster,
    in_palette="autumn",
    out_palette="winter",
):
    # Get colors
    in_cmap = plt.cm.get_cmap(in_palette, n_in_cluster)
    in_color_seq = np.array(
        [rgb2hex(in_cmap(i)) for i in range(n_in_cluster)] + ["#bbbbbb"]
    )
    out_cmap = plt.cm.get_cmap(out_palette, n_out_cluster)
    out_color_seq = np.array(
        [rgb2hex(out_cmap(i)) for i in range(n_out_cluster)] + ["#bbbbbb"]
    )

    df_in = pd.DataFrame({"y": np.ones_like(in_pred), "time": duration_arr})
    fig_in = px.bar(
        df_in,
        x="time",
        y="y",
        orientation="h",
        height=200,
        hover_data={
            "y": False,
            "pred": [chr(65 + i) if i > -1 else "?" for i in in_pred],
        },
    )
    fig_in.update_traces(marker_color=in_color_seq[in_pred])

    df_out = pd.DataFrame({"y": np.ones_like(in_pred), "time": duration_arr})
    fig_out = px.bar(
        df_out,
        x="time",
        y="y",
        orientation="h",
        height=200,
        hover_data={"y": False, "pred": [i if i > -1 else "?" for i in out_pred]},
    )
    fig_out.update_traces(marker_color=out_color_seq[out_pred])

    return fig_in, fig_out


def custom_multiselect(
    df: pd.DataFrame,
    labels: dict,
    expander_name: str,
    key: str,
) -> pd.DataFrame:

    expander = st.expander(f"{expander_name}")

    form_col = expander.form(f"{key}_columns")
    columns = form_col.multiselect(
        "Columns to filter",
        list(df.columns[1:]),
        default=list(df.columns[1:]),
        key=f"{key}",
    )
    form_col.form_submit_button("Apply columns")

    expander.write("Please select the features to keep below:")

    params = {}
    form_params = expander.form(f"{key}_labels")
    for param_name in columns:
        params[param_name] = form_params.multiselect(
            f"{param_name}",
            labels[param_name],
            default=labels[param_name],
            key=f"{key}_{param_name}",
        )
    form_params.form_submit_button("Apply labels")

    df_copy = df.copy(deep=True)
    for key, value in params.items():
        df_copy = df_copy[df_copy[key].isin(value)]

    return df_copy


def get_duration_array(predictions, total_duration):
    return (
        np.concatenate((predictions["in_start_index"][1:].to_numpy(), [total_duration]))
        - predictions["in_start_index"].to_numpy()
    )


def get_qrcode_values(
    predictions,
    in_ncluster,
    out_ncluster,
    duration_array,
    idx_list,
):
    qrcode = np.zeros((in_ncluster, out_ncluster))
    n_outliers, n_values = 0, 0

    for idx in idx_list:
        preds_df = pd.read_json(predictions[idx], orient="columns")
        n_values += len(preds_df)
        for idx_in, idx_out, duration in zip(
            preds_df["in_cluster"], preds_df["out_cluster"], duration_array
        ):
            if idx_in != -1 and idx_out != -1:
                qrcode[idx_in, idx_out] += duration
            else:
                n_outliers += 1

    qrcode /= np.sum(qrcode)
    return qrcode, n_outliers, n_values


def get_qrcode_fig(
    in_ncluster,
    out_ncluster,
    qrcode,
    showaxis=True,
    showscale=True,
    margin=True,
    width=400,
):
    in_labels = [chr(x) for x in range(ord("A"), ord("A") + in_ncluster)]
    out_labels = list(range(out_ncluster))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=out_labels,
            y=in_labels,
            z=qrcode,
            zmin=0,
            showscale=showscale,
            colorscale="viridis",
            text=qrcode,
            texttemplate="%{text:.3f}",
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(autorange="reversed", visible=showaxis)
    fig.update_xaxes(side="top", visible=showaxis)
    fig.update_layout(
        hovermode="closest",
        width=width,
        height=width,
    )

    if not margin:
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
