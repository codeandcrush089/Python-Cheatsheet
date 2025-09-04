
<p align="center">
  <img src="https://img.shields.io/badge/NumPy%20Library-High%20Performance%20Computing-007ACC?style=for-the-badge&logo=python&logoColor=white" alt="NumPy" />
</p>

<h1 align="center">🔢 NumPy – Fast Numerical Computing for Python</h1>

<p align="center">
  Arrays • Math • Efficiency
</p>

---

## **1. Introduction to NumPy**

* **What is NumPy?**
  NumPy (Numerical Python) is a Python library for **numerical computations, arrays, and matrix operations**.

* **Key Features:**

  * Provides the `ndarray` object (fast, memory-efficient array)
  * Supports element-wise operations
  * Mathematical, statistical, and linear algebra functions
  * Easy integration with Pandas, Matplotlib, and Scikit-learn

* **Installation:**

  ```bash
  pip install numpy
  ```

---

## **2. Importing NumPy**

```python
import numpy as np
```

---

## **3. Creating NumPy Arrays**

### 3.1 From Python List

```python
arr = np.array([1, 2, 3, 4])
print(arr)
```

### 3.2 Multi-dimensional Arrays

```python
arr2d = np.array([[1,2,3], [4,5,6]])
```

### 3.3 Using Built-in Functions

```python
np.zeros((2,3))       # 2x3 matrix of zeros
np.ones((3,3))        # 3x3 matrix of ones
np.eye(3)             # Identity matrix
np.arange(0,10,2)     # Even numbers from 0 to 8
np.linspace(0,1,5)    # 5 equally spaced numbers between 0 and 1
```

---

## **4. Array Attributes**

```python
arr = np.array([[1,2,3],[4,5,6]])
print(arr.shape)   # (2,3)
print(arr.ndim)    # 2 (dimensions)
print(arr.size)    # 6 (total elements)
print(arr.dtype)   # data type
```

---

## **5. Indexing & Slicing**

```python
arr = np.array([10,20,30,40,50])
print(arr[0])      # First element
print(arr[-1])     # Last element
print(arr[1:4])    # Slicing

arr2d = np.array([[1,2,3],[4,5,6]])
print(arr2d[0,1])  # 2
print(arr2d[:,1])  # Second column
```

---

## **6. Array Operations**

### Element-wise Operations

```python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a + b)
print(a * b)
print(a / b)
print(a ** 2)
```

### Scalar Operations

```python
print(a + 10)
print(a * 2)
```

---

## **7. Mathematical Functions**

```python
arr = np.array([1,2,3,4])
print(np.sqrt(arr))
print(np.exp(arr))
print(np.log(arr))
print(np.sum(arr))
print(np.mean(arr))
print(np.std(arr))
```

---

## **8. Reshaping & Transposing**

```python
arr = np.arange(12)        # 0 to 11
new_arr = arr.reshape(3,4) # 3x4 matrix
print(new_arr.T)           # Transpose
```

---

## **9. Stacking & Splitting**

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

# Stacking
np.hstack((a,b))   # Horizontal
np.vstack((a,b))   # Vertical

# Splitting
np.hsplit(a,2)     # Split columns
```

---

## **10. Broadcasting**

```python
a = np.array([1,2,3])
b = 5
print(a + b)  # Adds 5 to each element
```

---

## **11. Random Module in NumPy**

```python
np.random.rand(3,3)          # Random floats (0 to 1)
np.random.randint(1,100,5)   # Random integers
np.random.seed(42)           # Reproducibility
```

---

## **12. Linear Algebra**

```python
arr = np.array([[1,2],[3,4]])
print(np.linalg.det(arr))       # Determinant
print(np.linalg.inv(arr))       # Inverse
print(np.dot(arr, arr))         # Matrix multiplication
```

---

## **13. Saving & Loading Arrays**

```python
np.save('array.npy', arr)        # Save
loaded_arr = np.load('array.npy')# Load
```

---

# **NumPy Practice Questions**

### Basic Level

1. Create a NumPy array from a list `[10, 20, 30, 40]`.
2. Generate an array of even numbers between 1–20.
3. Create a 3×3 matrix filled with ones.

### Intermediate Level

1. Create an array of numbers from 1 to 12 and reshape it to 3×4.
2. Find the mean, median, and standard deviation of an array.
3. Extract all elements greater than 10 from an array.

### Advanced Level

1. Create two matrices and perform matrix multiplication.
2. Generate a random 5×5 matrix and find its determinant.
3. Implement broadcasting to add a 1D array to a 2D array.

---

# **NumPy Mini Projects**

1. **Matrix Calculator**

   * Perform addition, subtraction, and multiplication of two matrices.

2. **Random Data Generator**

   * Generate a dataset of 1000 random numbers and find mean, median, and standard deviation.

3. **Image to Grayscale Converter**

   * Use NumPy to manipulate pixel arrays (with PIL/Matplotlib).

4. **Linear Equation Solver**

   * Solve a system of linear equations using `np.linalg.solve()`.

5. **Monte Carlo Simulation**

   * Estimate the value of π using random numbers.

