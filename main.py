import numpy as np
import pandas as pd
import streamlit as st

from upload_data.data import from_zip_dataset_to_numpy

zip_file = st.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    file_array = from_zip_dataset_to_numpy(zip_file)

DOWN_SAMPFREQ = 250
PROMINENCE = 0.03
WLEN = 2
MIN_CYCLE = 0.1
MAX_CYCLE = 0.3
TRAINING_SIZE = 30
INTERVAL = 60
N_IN_CLUSTER = 3
N_OUT_CLUSTER = 3
IN_D = 0.2
OUT_D = 0.2
N_ITER = 10
MAX_WARPING = 0.06
QUANTILE = 0.95
NJOBS = 1
VERBOSE = True

SAMPFREQ = st.sidebar.number_input("Sampling frequency", step=100, value=2000)
