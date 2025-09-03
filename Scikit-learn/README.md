
# Scikit-learn 

---

## **1. Introduction**

* **Scikit-learn:**
  A Python library for **Machine Learning and Data Analysis**.

* Built on **NumPy, Pandas, SciPy, and Matplotlib**.

* Provides tools for:

  * Data preprocessing
  * Classification
  * Regression
  * Clustering
  * Dimensionality reduction
  * Model evaluation

* **Installation:**

  ```bash
  pip install scikit-learn
  ```

---

## **2. Importing**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
```

---

## **3. Data Preprocessing**

### 3.1 Loading Dataset

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
```

---

### 3.2 Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 3.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## **4. Regression Example – Linear Regression**

```python
# Dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)
```

---

## **5. Classification Example – Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## **6. Clustering Example – KMeans**

```python
from sklearn.cluster import KMeans

X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Cluster Centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
```

---

## **7. Dimensionality Reduction – PCA**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
print("Reduced Shape:", X_pca.shape)
```

---

## **8. Model Evaluation**

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## **9. Common Algorithms in Scikit-learn**

* **Regression:** LinearRegression, Ridge, Lasso, SVR
* **Classification:** LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, KNeighborsClassifier
* **Clustering:** KMeans, DBSCAN, AgglomerativeClustering
* **Dimensionality Reduction:** PCA, t-SNE
* **Model Selection:** GridSearchCV, RandomizedSearchCV

---

## **10. Saving & Loading Models**

```python
import joblib

joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
```

---

# **Scikit-learn Practice Questions**

### Beginner

1. Load the Iris dataset and print its shape.
2. Split data into 80% training and 20% testing.
3. Fit a Linear Regression model on sample data and predict new values.

### Intermediate

1. Train a Logistic Regression model on the Iris dataset and calculate accuracy.
2. Apply StandardScaler to a dataset and verify mean & variance.
3. Perform KMeans clustering on a random dataset with 3 clusters.

### Advanced

1. Use PCA to reduce features of the Iris dataset and visualize using Matplotlib.
2. Perform hyperparameter tuning using GridSearchCV for a RandomForest model.
3. Build a confusion matrix and classification report for a trained model.

---

# **Mini Projects (Scikit-learn)**

### 1. **Student Score Predictor**

* Input: Study hours vs Exam scores dataset
* Use: Linear Regression
* Output: Predict marks for a given study time

---

### 2. **Iris Flower Classifier**

* Use Logistic Regression or Random Forest
* Train on Iris dataset
* Predict species for new input

---

### 3. **Customer Segmentation (Clustering)**

* Dataset: Customer age & spending score
* Algorithm: KMeans
* Visualize clusters using Matplotlib or Seaborn

---

### 4. **Email Spam Classifier**

* Dataset: Email content (spam/ham)
* Steps:

  * Text preprocessing (CountVectorizer/Tfidf)
  * Train Naive Bayes classifier
  * Test on new emails

---

### 5. **House Price Prediction**

* Dataset: Features (area, rooms, location) → price
* Algorithm: Linear Regression or Random Forest
* Evaluate using RMSE & R² score

---

### 6. **Diabetes Prediction**

* Dataset: PIMA Diabetes dataset
* Algorithm: Logistic Regression or SVC
* Output: Predict whether a person is diabetic

