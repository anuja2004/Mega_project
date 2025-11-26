import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data(partition=1, test_size=0.2, batch_size=32):
    df = pd.read_csv("dataset.csv")

    # Assuming the target column name is 'label'
    X = df.drop(columns=['label']).values
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=partition)

    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader
