# server.py
import flwr as fl
import json
from datetime import datetime

LOG_FILE = "training_log.json"

# --- Function to log metrics ---
def log_metrics(round_num, avg_loss, avg_acc, avg_precision, avg_recall, avg_f1):
    entry = {
        "round": round_num,
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
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

    print(f"📊 Logged Round {round_num}: Acc={avg_acc:.4f}, Loss={avg_loss:.4f}, "
          f"P={avg_precision:.4f}, R={avg_recall:.4f}, F1={avg_f1:.4f}")


# --- Aggregation function ---
def evaluate_metrics_aggregation_fn(results):
    accs = [r[1].get("test_accuracy") for r in results if "test_accuracy" in r[1]]
    losses = [r[1].get("test_loss") for r in results if "test_loss" in r[1]]
    precisions = [r[1].get("precision") for r in results if "precision" in r[1]]
    recalls = [r[1].get("recall") for r in results if "recall" in r[1]]
    f1s = [r[1].get("f1_score") for r in results if "f1_score" in r[1]]

    avg_acc = sum(accs) / len(accs)
    avg_loss = sum(losses) / len(losses)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
        round_num = len(data) + 1
    except:
        round_num = 1

    log_metrics(round_num, avg_loss, avg_acc, avg_precision, avg_recall, avg_f1)

    print(
        f"📈 Global Round {round_num} — Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
        f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}"
    )

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
    }



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
