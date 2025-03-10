import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load test dataset (Make sure it follows the same preprocessing as training data)
CSV_FILE = "asl_landmarks.csv"  # Use a separate test dataset if available
df = pd.read_csv(CSV_FILE)

# Split features and labels
X_test = df.iloc[:, :-1].values  # Landmarks
y_test = df.iloc[:, -1].values   # ASL Letters

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Encode labels
label_encoder = np.load("label_encoder.npy", allow_pickle=True)
y_test_encoded = np.array([np.where(label_encoder == label)[0][0] for label in y_test])

# Predict the classes
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate Accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")
