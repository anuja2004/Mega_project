import streamlit as st
import json
import time
import os

# --- Page Config ---
st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #0d1117; color: white; }
        .card {
            background: linear-gradient(135deg, #1f2937, #111827);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
            text-align: center;
        }
        .divider {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #3b82f6, #06b6d4);
            margin: 25px 0;
        }
        .round-box {
            background: #1f2937;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("## 🏦 **Federated Learning Training Dashboard**")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# --- Load training log ---
LOG_FILE = "training_log.json"
data = []
if os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

# --- Display latest summary ---
if data:
    latest = data[-1]
    round_num = latest["round"]
    avg_acc = latest["avg_acc"]
    avg_loss = latest["avg_loss"]

    st.markdown(f"### 🌸 **Round {round_num} Summary**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average Accuracy**")
        st.markdown(f"<h1 style='color:#22c55e;'>{avg_acc:.4f}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Average Loss**")
        st.markdown(f"<h1 style='color:#ef4444;'>{avg_loss:.4f}</h1>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No training data available yet.")
    st.stop()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# --- Bank progress section ---
st.markdown("### 🏛️ **Client Bank Training Progress**")
banks = ["🏦 SBI", "🏦 HDFC"]
cols = st.columns(len(banks))
progress_bars = []

for i, bank in enumerate(banks):
    with cols[i]:
        st.markdown(f"<div class='card'><h3>{bank}</h3>", unsafe_allow_html=True)
        pb = st.progress(0)
        progress_bars.append(pb)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Visualization ---
if st.button("🚀 Start Training Visualization"):
    st.info("Replaying training visualization based on actual log results...")
    total_rounds = 10

    for i, entry in enumerate(data, start=1):
        acc = entry["avg_acc"]
        loss = entry["avg_loss"]

        for pb in progress_bars:
            pb.progress(i / total_rounds)

        st.markdown(
            f"<div class='round-box'>📊 <b>Round {i}</b> → "
            f"Accuracy: <span style='color:#22c55e;'>{acc:.4f}</span>, "
            f"Loss: <span style='color:#ef4444;'>{loss:.4f}</span></div>",
            unsafe_allow_html=True
        )
        time.sleep(1.2)

    st.success("✅ Visualization complete! Metrics shown are actual logged values.")
else:
    st.info("💡 Click *Start Training Visualization* to replay previous rounds.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# --- Full training history ---
st.markdown("### 🧾 **Training History**")
for entry in reversed(data):
    st.markdown(
        f"<div class='round-box'>🌸 <b>Round {entry['round']}</b> → "
        f"Accuracy: <span style='color:#22c55e;'>{entry['avg_acc']:.4f}</span>, "
        f"Loss: <span style='color:#ef4444;'>{entry['avg_loss']:.4f}</span></div>",
        unsafe_allow_html=True
    )
