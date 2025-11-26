# server.py
import flwr as fl
import json
from datetime import datetime

LOG_FILE = "training_log.json"

# --- Function to log metrics ---
def log_metrics(round_num, avg_loss, avg_acc):
    entry = {
        "round": round_num,
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"📊 Logged Round {round_num}: Acc={avg_acc:.4f}, Loss={avg_loss:.4f}")


# --- Aggregation function ---
def evaluate_metrics_aggregation_fn(results):
    # Each result = (client_proxy, metrics)
    accs = [r[1]["test_accuracy"] for r in results if "test_accuracy" in r[1]]
    losses = [r[1]["test_loss"] for r in results if "test_loss" in r[1]]

    if not accs or not losses:
        print("⚠️ No metrics received from clients.")
        return {}

    avg_acc = sum(accs) / len(accs)
    avg_loss = sum(losses) / len(losses)

    # --- Safe handling for training_log.json ---
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
        round_num = len(data) + 1
    except FileNotFoundError:
        data = []
        round_num = 1
    except json.JSONDecodeError:
        data = []
        round_num = 1

    # Log metrics for Streamlit
    log_metrics(round_num, avg_loss, avg_acc)

    print(f"📈 Global Round {round_num} — avg_loss: {avg_loss:.4f}, avg_acc: {avg_acc:.4f}")
    return {"avg_loss": avg_loss, "avg_acc": avg_acc}



# --- Main function ---
def main():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    print("🚀 Starting Flower server on 127.0.0.1:8080")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
