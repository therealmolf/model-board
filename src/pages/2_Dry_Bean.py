# %%
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch as t

from models.mlp import MLP
from models.config import Config

from pathlib import Path


plt.style.use('ggplot')


# %%
path = Path("/home/therealmolf/model-board/data/" +
            "DryBeanDataset/Dry_Bean_Dataset.xlsx")
df = pd.read_excel(path)

# %%
st.write("# Multilayer Perceptron on Dry Bean Dataset")

st.write("---")
st.markdown("""
    ### Some Details:
    - ***Type:*** Multiclass Classification
    - ***Instances:*** 13,611
    - ***Features:*** 16
    - ***Source:*** UCI Machine Learning Repository
    - ***Note:*** This is an unreliable model that has only been trained twice
                  for demonstrative purposes.
    ---
""")


page = st.sidebar.selectbox("Select Page", [
                                            "Data Analysis",
                                            "Prediction"])


if page == "Data Analysis":
    st.write("#### First 5 Rows")
    st.table(df.head())

    st.markdown("---")

    st.write('#### Class Distribution')

    value_counts = df['Class'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots()
    ax.barh(value_counts.index, value_counts, align='center')
    plt.xlabel('Count')
    plt.ylabel('Value')
    plt.title('Class Distribution')

    st.pyplot(fig)

elif page == "Prediction":

    cfg = Config()
    model = MLP(cfg).eval()

    scaler = MinMaxScaler()

    st.write("### Inputs")

    col_dict = {}
    for column in df.columns[:-1]:
        col_dict[column] = st.number_input(column)

    pred_button = st.button("Click")

    if pred_button:
        # This is unideal since you should have a pickled scaler
        # fit_transformed on just the training dataset
        scaler = MinMaxScaler()
        scaler.fit_transform(df.drop(["Class"], axis='columns').values)

        instance = scaler.transform([list(col_dict.values())])

        # Pass scaled instance into the trained model
        instance = t.tensor(instance,
                            device=cfg.device,
                            dtype=t.float32,
                            requires_grad=False)

        with t.no_grad():
            logits = model(instance)
            output = logits.argmax(dim=1)

        # Convert number output to Class name
        _, uniques = pd.factorize(df['Class'])

        st.write("# Logits: ")
        st.table(logits)

        st.write("# Prediction: ")
        st.write(f"##### The num prediction is {int(output)}")
        st.write(f"### The class prediction is {uniques[int(output)]}")
        st.write("""Note: the output is most likely unreliable due to the
                    the way scaler is transforming the instance.
                """)

# %%
