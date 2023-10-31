# %%
import streamlit as st
import pandas as pd

from util.plotter_functions import class_dist_bar, usage_co2_scatter
from pathlib import Path

# %%
# TODO: read csv for month df and for class
BASE_DIR = "/home/therealmolf/model-board/data"
MONTH_FILE = "month.csv"
CLASS_DIST_FILE = "class_dist.csv"

path = Path(BASE_DIR)

month_df = pd.read_csv(path / MONTH_FILE)
class_dist = pd.read_csv(path / CLASS_DIST_FILE)


# %%
st.write("# Steel Industry Energy Consumption Dataset")
st.write("---")

st.markdown("""
    ### Some Details:
    - ***Type:*** Regression
    - ***Instances:*** 35,040
    - ***Features:*** 9
    - ***Source:*** UCI Machine Learning Repository
    - ***Note:*** Just basic analysis!
    ---
""")


page = st.sidebar.selectbox("Select Page", ["Data Analysis"])


if page == "Data Analysis":
    class_fig = class_dist_bar(class_dist)
    st.plotly_chart(class_fig)

    st.markdown("---")

    usage_fig = usage_co2_scatter(month_df)
    st.plotly_chart(usage_fig)
