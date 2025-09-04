<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow%20Library-Deep%20Learning%20%26%20AI-FF6F00?style=for-the-badge&logo=python&logoColor=white" alt="TensorFlow" />
</p>

<h1 align="center">🧠 TensorFlow – Scalable Deep Learning Framework</h1>

<p align="center">
  Build • Train • Deploy
</p>


---

## **1. Introduction**

* **TensorFlow:** An open-source deep learning library by Google.

* Used for:

  * Neural networks (ANN, CNN, RNN, LSTM)
  * Image processing
  * Natural Language Processing (NLP)
  * Time series forecasting

* **Installation:**

  ```bash
  pip install tensorflow
  ```

* **Key Components:**

  * `tf.constant`, `tf.Variable`
  * `tf.keras` (high-level API)
  * GPU/TPU support for acceleration

---

## **2. Importing TensorFlow**

```python
import tensorflow as tf
print(tf.__version__)
```

---

## **3. Tensors (Core Data Structure)**

```python
# Creating tensors
scalar = tf.constant(5)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]])
tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(scalar.shape, vector.shape, matrix.shape)
```

---

## **4. Basic Operations**

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

add = tf.add(a, b)
mul = tf.multiply(a, b)
dot = tf.tensordot(a, b, axes=1)
```

---

## **5. Building a Neural Network with Keras**

### 5.1 Basic Sequential Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dummy dataset
import numpy as np
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3])

model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)
predictions = model.predict(X)
```

---

### 5.2 Classification Example (MNIST)

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.reshape(-1, 784), y_train, epochs=5, batch_size=32)
```

---

## **6. Convolutional Neural Networks (CNN)**

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## **7. Recurrent Neural Networks (RNN/LSTM)**

```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(100, 1)),
    LSTM(50),
    Dense(1)
])
```

---

## **8. Saving & Loading Models**

```python
model.save("my_model.h5")  # Save
loaded_model = tf.keras.models.load_model("my_model.h5")  # Load
```

---

## **9. Callbacks**

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, callbacks=[early_stop])
```

---

## **10. TensorFlow vs PyTorch**

* **TensorFlow:** Production-ready, supports TensorFlow Serving & TFLite
* **PyTorch:** Research-friendly, more Pythonic

---

# **TensorFlow Practice Questions**

### Beginner

1. Create a constant tensor and perform addition & multiplication.
2. Build a simple Sequential model with one hidden layer.
3. Train a model to predict y = 2x + 1.

### Intermediate

1. Train a model on MNIST dataset (basic ANN).
2. Implement EarlyStopping callback in training.
3. Visualize model accuracy using Matplotlib.

### Advanced

1. Build a CNN for CIFAR-10 dataset classification.
2. Implement LSTM for stock price prediction.
3. Save a trained model and load it for inference.

---

# **Mini Projects (TensorFlow)**

### 1. **Handwritten Digit Recognition (MNIST)**

* Dataset: MNIST
* Model: ANN & CNN comparison
* Output: Predict digit from image

---

### 2. **House Price Prediction**

* Input: Features like area, rooms, location
* Model: Regression (ANN)
* Evaluate: MAE, MSE

---

### 3. **Sentiment Analysis on Movie Reviews**

* Dataset: IMDB reviews
* Preprocessing: Tokenization, Embedding
* Model: LSTM
* Output: Positive/Negative sentiment

---

### 4. **Stock Price Forecasting**

* Data: Historical stock prices
* Model: LSTM
* Output: Predict next 30 days

---

### 5. **Image Classifier (Cats vs Dogs)**

* Dataset: Kaggle Cats vs Dogs
* Model: CNN
* Output: Classify image as cat or dog

---

### 6. **AI Chatbot using TensorFlow**

* Dataset: Predefined intents.json
* Model: NLP + Dense Layers
* Output: Interactive chatbot

