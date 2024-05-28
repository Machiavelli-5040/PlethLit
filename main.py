import numpy as np
import pandas as pd
import streamlit as st

from upload_data.data import from_zip_dataset_to_numpy

zip_file = st.file_uploader("Upload a zip file", type="zip")
if zip_file is not None:
    file_array = from_zip_dataset_to_numpy(zip_file)
