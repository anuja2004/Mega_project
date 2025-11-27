import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ------------------ Load dataset ------------------
df = pd.read_csv("data/fraud_data.csv")

# ------------------ Balance dataset ------------------
fraud_df = df[df["is_fraud"] == 1]
non_fraud_df = df[df["is_fraud"] == 0][:2500]
df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------ Keep only numeric columns ------------------
df = df.select_dtypes(include=["number"])

# ------------------ Features & Labels ------------------
X = df.drop(columns=["is_fraud"]).values
y = df["is_fraud"].values

# ------------------ Scale features ------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------ Train-test split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ ANN Model ------------------
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ------------------ Train model ------------------
print("\n Starting ANN Training...\n")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

# ------------------ Evaluate ------------------
print("\n Training Complete!\n")
loss, acc = model.evaluate(X_test, y_test)
print(f"🔹 Test Accuracy: {acc * 100:.2f}%")
print(f"🔹 Test Loss: {loss:.4f}")

# ------------------ Predictions ------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

# ------------------ Classification Report ------------------
report = classification_report(y_test, y_pred)
print("\nTraditional model\n")
print("\n📊 Classification Report:\n")
print(report)

# Save to file
with open("classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {acc * 100:.4f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("\n💾 Report saved to 'classification_report.txt'")
