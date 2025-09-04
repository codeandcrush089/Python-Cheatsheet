<p align="center">
  <img src="https://img.shields.io/badge/Dask%20Library-Data%20Processing%20Made%20Easy-FF6F61?style=for-the-badge&logo=python&logoColor=white" alt="Dask Logo" />
</p>

<h1 align="center">🚀 Dask – Scalable Python for Big Data</h1>

<p align="center">
  Efficient • Parallel • Powerful
</p>


## **1. Introduction**

* **Dask:** A parallel computing library for large datasets.

* Similar to Pandas but can handle data larger than memory.

* Supports:

  * Parallel & distributed computing
  * Arrays, DataFrames, Delayed computations
  * Integration with NumPy, Pandas, Scikit-learn

* **Installation:**

```bash
pip install dask[complete]
```

---

## **2. Importing Dask**

```python
import dask.array as da
import dask.dataframe as dd
```

---

## **3. Dask Arrays (NumPy-like)**

```python
# Create a random Dask array
x = da.random.random((10000, 10000), chunks=(1000, 1000))
mean = x.mean().compute()  # compute() executes the task
print(mean)
```

---

## **4. Dask DataFrames (Pandas-like)**

```python
# Load large CSV
df = dd.read_csv("large_dataset.csv")

# Basic operations
print(df.head())
print(df.describe().compute())
```

---

## **5. Parallel Computing Example**

```python
from dask import delayed

@delayed
def square(x):
    return x * x

@delayed
def add(x, y):
    return x + y

result = add(square(10), square(20))
print(result.compute())
```

---

## **6. Persisting Data**

```python
df = df.persist()  # Keep data in memory for faster operations
```

---

## **7. Saving Data**

```python
df.to_csv("output/*.csv", index=False)
```

---

## **8. Integration with Scikit-learn**

```python
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression

df = dd.read_csv("large_dataset.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

# **Dask Practice Questions**

### Beginner

1. Create a Dask array and find its mean and sum.
2. Load a large CSV file and display its first 10 rows.
3. Convert a Pandas DataFrame to a Dask DataFrame.

### Intermediate

1. Filter rows from a large dataset based on a condition.
2. Perform groupby and aggregation using Dask.
3. Save filtered data to multiple CSV files.

### Advanced

1. Implement a delayed function to perform addition and multiplication.
2. Build a Dask pipeline to clean large data.
3. Train a machine learning model on a dataset larger than RAM.

---

# **Dask Mini Projects**

### 1. **Large CSV Cleaner**

* Input: Large dataset (1GB+)
* Features:

  * Missing value handling
  * Filter unwanted rows
  * Save cleaned dataset in chunks

---

### 2. **Parallel Data Processor**

* Input: Multiple CSV files
* Features:

  * Read all files in parallel
  * Aggregate and merge data
  * Save final result

---

### 3. **Big Data E-commerce Analysis**

* Dataset: E-commerce sales
* Features:

  * Total revenue per region
  * Top 10 products by sales
  * Save report as CSV

---

