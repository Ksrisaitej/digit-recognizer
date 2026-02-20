import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np

model = Sequential(
    [
        Dense(units=256, activation='relu', input_shape=(784,)),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax'),
    ]
)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
df= pd.read_csv('train.csv')
x_train = df[df.columns[1:]].values/255.0
y_train = df[df.columns[0]].values
model.fit(x_train, y_train, epochs=100, batch_size=32)

test_df = pd.read_csv('test.csv')
x_test = test_df.values / 255.0

predictions = model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

submission.to_csv('submission.csv', index=False)
