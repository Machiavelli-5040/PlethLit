import json as json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyedflib
import streamlit as st


@st.cache_data
def from_zip_dataset_to_numpy(_zp: ZipFile) -> np.ndarray:
    """Take a dataset

    Args:
        zp (ZipFile): dataset as a zip object

    Raises:
        ValueError: Not a 1D signal
        ValueError: One signal cannot be read

    Returns:
        np.ndarray: dataset as a np.ndarray
    """

    # Select filename with the right extention and take out the mac extension
    extract_dir = f"datasets/{_zp.filename[:-4]}/"
    extension_lst = [".txt", ".csv", ".EDF"]
    mac_sub_dir = "__MACOSX"
    file_to_check = [
        filename for filename in _zp.namelist() if filename[-4:] in extension_lst
    ]
    file_to_check = [
        filename for filename in file_to_check if mac_sub_dir not in filename
    ]
    # sort in alphabetic order
    file_to_check.sort()

    if not file_to_check:
        raise ValueError

    # Extract signals
    lst = []
    for filename in file_to_check:
        try:
            if filename[-4:] == ".EDF":
                signal = (
                    pyedflib.EdfReader(_zp.extract(filename, extract_dir))
                    .readSignal(0)
                    .reshape(-1)
                )
                lst.append(signal)
            else:
                signal = pd.read_csv(
                    _zp.extract(filename, extract_dir), header=None
                ).values
                if signal.ndim == 2:
                    signal = signal.reshape(-1)
                    lst.append(signal)
                else:
                    raise ValueError
        except:
            raise ValueError

    return np.array(lst, dtype=object)
