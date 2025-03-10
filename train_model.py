import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Load the updated dataset
df = pd.read_csv("asl_landmarks1.csv")  # Updated filename

# ✅ Prepare data (63 features instead of 42)
X = df.iloc[:, :-1].values  # All 63 landmark features
y = df.iloc[:, -1].values   # Labels

# ✅ Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Optimized model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Now 63 input features
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes (A-Z)
])

# ✅ Tune optimizer learning rate
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train with optimized settings
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ✅ Save model and label encoder
model.save("asl_model.h5")
np.save("label_encoder.npy", label_encoder.classes_)

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n📊 Model Accuracy on Test Data: {test_acc * 100:.2f}%")
