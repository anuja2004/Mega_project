import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "training_log.json"

st.title("📊 Federated Learning Training Dashboard")

# Load logs
try:
    with open(LOG_FILE, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except:
    st.error("No training_log.json found.")
    st.stop()

st.subheader("📈 Metrics Table")
st.dataframe(df)

# ----- Plotting function -----
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(7,4))
    plt.plot(df["round"], df[metric_name], marker='o')
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Over Rounds")
    st.pyplot(plt)

# ----- Display all graphs -----
st.subheader("📉 Accuracy")
plot_metric("avg_acc", "Accuracy")

st.subheader("📉 Loss")
plot_metric("avg_loss", "Loss")

st.subheader("📉 Precision")
plot_metric("avg_precision", "Precision")

st.subheader("📉 Recall")
plot_metric("avg_recall", "Recall")

st.subheader("📉 F1 Score")
plot_metric("avg_f1", "F1 Score")
