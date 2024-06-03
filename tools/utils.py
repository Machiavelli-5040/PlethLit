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

    df_in = pd.DataFrame(
        {"y": np.ones_like(in_pred), "time": duration_arr, "pred": in_pred}
    )
    fig_in = px.bar(
        df_in, x="time", y="y", orientation="h", height=200, hover_data=["pred"]
    )
    fig_in.update_traces(marker_color=in_color_seq[in_pred])
    fig_in.update_layout(
        xaxis_visible=False,
        xaxis_showticklabels=False,
        yaxis_visible=False,
        yaxis_showticklabels=False,
        showlegend=False,
    )

    df_out = pd.DataFrame(
        {"y": np.ones_like(in_pred), "time": duration_arr, "pred": out_pred}
    )
    fig_out = px.bar(
        df_out, x="time", y="y", orientation="h", height=200, hover_data=["pred"]
    )
    fig_out.update_traces(marker_color=out_color_seq[in_pred])
    fig_out.update_layout(
        xaxis_visible=False,
        xaxis_showticklabels=False,
        yaxis_visible=False,
        yaxis_showticklabels=False,
        showlegend=False,
    )

    return fig_in, fig_out


def qrcode_plot(
    in_labels,
    out_labels,
    qrcode,
    showaxis=True,
    showscale=True,
    margin=True,
    width=400,
):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=out_labels,
            y=in_labels,
            z=qrcode,
            zmin=0,
            showscale=showscale,
            colorscale="viridis",
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
    st.plotly_chart(fig)
    return fig
