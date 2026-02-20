# digit-recognizer

🧠 Handwritten Digit Classification with TensorFlow

This project builds a neural network model using TensorFlow/Keras to classify handwritten digits (0–9) from pixel data.
It trains on labeled images and predicts labels for unseen test images.

📌 What the Code Does
1️⃣ Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np

TensorFlow/Keras → builds & trains neural networks

Pandas → loads CSV datasets

NumPy → numerical operations

2️⃣ Build the Neural Network
model = Sequential([
    Dense(units=256, activation='relu', input_shape=(784,)),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax'),
])

Architecture:

Input: 784 pixels (28×28 image flattened)

Hidden Layer 1: 256 neurons (ReLU)

Hidden Layer 2: 128 neurons (ReLU)

Output Layer: 10 neurons (digits 0–9)

👉 ReLU helps learn patterns
👉 Softmax outputs probabilities

3️⃣ Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

Adam → efficient optimizer

Loss function → used for multi-class classification

Accuracy → performance metric

4️⃣ Load & Prepare Training Data
df = pd.read_csv('train.csv')

x_train = df[df.columns[1:]].values / 255.0
y_train = df[df.columns[0]].values

✔ Reads training dataset
✔ Separates labels & pixels
✔ Normalizes pixel values (0–255 → 0–1)

👉 Normalization improves training speed & accuracy.

5️⃣ Train the Model
model.fit(x_train, y_train, epochs=100, batch_size=32)

Epochs = 100 → number of training cycles

Batch size = 32 → samples processed at once

6️⃣ Load Test Data
test_df = pd.read_csv('test.csv')
x_test = test_df.values / 255.0
7️⃣ Make Predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

✔ Predict probabilities
✔ Select digit with highest probability

8️⃣ Create Submission File
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

submission.to_csv('submission.csv', index=False)

Outputs predictions in CSV format.

📂 Expected Dataset Format
train.csv
label	pixel1	pixel2	...
test.csv

| pixel1 | pixel2 | ... |

This format is commonly used in the MNIST/Kaggle digit recognizer dataset.

▶️ How to Run
1️⃣ Install dependencies
pip install tensorflow pandas numpy
2️⃣ Place datasets

Put train.csv and test.csv in the project folder.

3️⃣ Run script
python script.py
4️⃣ Output

A file named:

submission.csv

will contain predicted digit labels.
