import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import flwr as fl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from model import FraudModel


# ------------------------- Data Loading -------------------------
def load_data(cid):
    df = pd.read_csv("data/fraud_data.csv")

    # Drop unnecessary columns
    drop_cols = ["trans_date_trans_time", "cc_num", "merchant", "first", "last",
                 "street", "city", "state", "job", "dob", "trans_num"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # ------------------ ✅ Balance and limit dataset ------------------
    fraud_df = df[df['is_fraud'] == 1]
    non_fraud_df = df[df['is_fraud'] == 0][:2500]
    df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ------------------ Split features and labels ------------------
    X = df.drop(columns=["is_fraud"]).values
    y = df["is_fraud"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Different split for each client
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(cid)
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # ✅ Use a larger batch size for faster training
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

    return train_loader, test_loader, X.shape[1]


# ------------------------- Training -------------------------
def train(model, loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_loss, correct = 0.0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


# ------------------------- Evaluation -------------------------
def test(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


# ------------------------- Flower Client -------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss, train_acc = train(self.model, self.train_loader, self.device)
        print(f"📊 Client training — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        return self.get_parameters(config), len(self.train_loader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, test_acc = test(self.model, self.test_loader, self.device)
        print(f"🧪 Client evaluation — loss: {test_loss:.4f}, acc: {test_acc:.4f}")
        return float(test_loss), len(self.test_loader.dataset), {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }


# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, input_dim = load_data(args.cid)
    model = FraudModel(input_dim=input_dim).to(device)
    client = FlowerClient(model, train_loader, test_loader, device)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
