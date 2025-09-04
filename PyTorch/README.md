<p align="center">
  <img src="https://img.shields.io/badge/PyTorch%20Library-Deep%20Learning%20%26%20AI-E65100?style=for-the-badge&logo=python&logoColor=white" alt="PyTorch" />
</p>

<h1 align="center">🔥 PyTorch – Deep Learning Framework for Python</h1>

<p align="center">
  Train • Deploy • Innovate
</p>
 

---

## **1. Introduction**

* **PyTorch:** An open-source deep learning library developed by Facebook.
* Known for:

  * Dynamic computation graph
  * Easy debugging and flexibility
  * Widely used in AI research, NLP, and computer vision
* Installation:

  ```bash
  pip install torch torchvision torchaudio
  ```

---

## **2. Importing PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

---

## **3. Tensors (Core Data Structure)**

```python
# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.rand(2, 3)
z = torch.zeros(3, 3)
ones = torch.ones(3, 3)
```

* **Basic operations:**

```python
a = torch.tensor([2, 4, 6])
b = torch.tensor([1, 3, 5])
print(a + b)        # Addition
print(a * b)        # Element-wise multiplication
print(torch.dot(a, b))
```

* **Check GPU:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

---

## **4. Autograd (Automatic Differentiation)**

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 4
y.backward()
print(x.grad)  # Derivative dy/dx
```

---

## **5. Building Neural Networks (PyTorch)**

### 5.1 Define a Model

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

---

### 5.2 Training a Simple Model

```python
# Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## **6. Datasets & Dataloaders**

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

---

## **7. Classification Example (MNIST)**

```python
from torchvision import datasets, transforms

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

---

## **8. Convolutional Neural Networks (CNN)**

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32*13*13, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32*13*13)
        return self.fc(x)
```

---

## **9. Recurrent Neural Networks (RNN/LSTM)**

```python
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

---

## **10. Saving & Loading Models**

```python
torch.save(model.state_dict(), 'model.pth')  # Save
model.load_state_dict(torch.load('model.pth'))  # Load
```

---

## **11. Optimizers**

* `torch.optim.SGD`
* `torch.optim.Adam`
* `torch.optim.RMSprop`

---

## **12. Loss Functions**

* `nn.MSELoss()` – Regression
* `nn.CrossEntropyLoss()` – Classification
* `nn.BCELoss()` – Binary classification

---

## **13. PyTorch vs TensorFlow**

* **PyTorch:** Dynamic graphs, more Pythonic, research-friendly
* **TensorFlow:** Static graphs (with eager mode), better production tools

---

# **PyTorch Practice Questions**

### Beginner

1. Create a tensor of shape (3,3) filled with random numbers.
2. Perform element-wise multiplication between two tensors.
3. Calculate gradients for a simple function: y = x² + 3x.

### Intermediate

1. Train a basic regression model to predict y = 2x + 1.
2. Implement a DataLoader with batch size = 4.
3. Build a simple feedforward neural network with one hidden layer.

### Advanced

1. Train a CNN on MNIST dataset.
2. Implement an LSTM for stock price prediction.
3. Save and reload a trained model for inference.

---

# **Mini Projects (PyTorch)**

### 1. **Handwritten Digit Recognition (MNIST)**

* Dataset: MNIST
* Model: CNN
* Output: Predict digit from image

---

### 2. **House Price Prediction**

* Dataset: Housing dataset (e.g., Boston housing)
* Model: Regression (Feedforward Neural Network)
* Evaluate: MAE, MSE, R² score

---

### 3. **Image Classifier (Cats vs Dogs)**

* Dataset: Kaggle Cats vs Dogs
* Model: CNN
* Output: Classify as cat or dog

---

### 4. **Sentiment Analysis**

* Dataset: IMDB reviews
* Model: LSTM
* Output: Positive/Negative sentiment

---

### 5. **Stock Price Forecasting**

* Data: Time series data
* Model: LSTM
* Output: Predict next 30 days prices

---

### 6. **AI Image Colorizer**

* Input: Black & white images
* Model: CNN
* Output: Colored images

