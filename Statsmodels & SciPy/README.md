<p align="center">
  <img src="https://img.shields.io/badge/Statsmodels%20%26%20SciPy-Statistical%20%26%20Scientific%20Computing-0288D1?style=for-the-badge&logo=python&logoColor=white" alt="Statsmodels & SciPy" />
</p>

<h1 align="center">📐 Statsmodels & SciPy – Statistical & Scientific Computing</h1>

<p align="center">
  Model • Test • Optimize
</p>

---
## **1. Introduction**

### **Statsmodels**

* Focus: **Statistical models & hypothesis testing**
* Used for:

  * Regression analysis
  * ANOVA, T-tests, Chi-square tests
  * Time series analysis (ARIMA, SARIMAX)
* Installation:

  ```bash
  pip install statsmodels
  ```

### **SciPy**

* Focus: **Scientific & numerical computing**
* Built on top of NumPy
* Used for:

  * Linear algebra
  * Statistics & probability
  * Optimization
  * Signal processing
* Installation:

  ```bash
  pip install scipy
  ```

---

## **2. Importing**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats, optimize, linalg
```

---

## **3. Statsmodels – Basics**

### 3.1 Linear Regression

```python
# Example dataset
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + np.random.randn(100)

X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())
```

---

### 3.2 Logistic Regression

```python
from statsmodels.discrete.discrete_model import Logit

# Binary classification dataset
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

X = sm.add_constant(X)
model = Logit(y, X).fit()
print(model.summary())
```

---

### 3.3 ANOVA (Analysis of Variance)

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.DataFrame({
    'group': ['A']*5 + ['B']*5 + ['C']*5,
    'value': [23, 21, 19, 22, 20, 30, 29, 31, 28, 32, 40, 42, 41, 39, 38]
})

model = ols('value ~ C(group)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

---

### 3.4 Time Series (ARIMA)

```python
from statsmodels.tsa.arima.model import ARIMA

data = np.random.randn(100)
model = ARIMA(data, order=(1,1,1))
result = model.fit()
print(result.summary())
```

---

## **4. SciPy – Basics**

### 4.1 Descriptive Statistics

```python
data = np.random.randn(100)
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

print("Mean:", mean, "Median:", median, "Std Dev:", std_dev)
```

---

### 4.2 Hypothesis Testing

#### **T-test**

```python
sample1 = np.random.randn(50)
sample2 = np.random.randn(50) + 0.5

t_stat, p_value = stats.ttest_ind(sample1, sample2)
print("T-statistic:", t_stat, "P-value:", p_value)
```

#### **Chi-square Test**

```python
observed = np.array([[50, 30], [20, 100]])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print("Chi2:", chi2, "P-value:", p)
```

#### **ANOVA**

```python
f_stat, p_value = stats.f_oneway(sample1, sample2, np.random.randn(50))
print("F-statistic:", f_stat, "P-value:", p_value)
```

---

### 4.3 Probability Distributions

```python
from scipy.stats import norm

# Probability density function
x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, 0, 1)

# Cumulative distribution function
cdf = norm.cdf(x, 0, 1)
```

---

### 4.4 Optimization Example

```python
def func(x):
    return x**2 + 4*x + 5

result = optimize.minimize(func, x0=0)
print("Minimum at:", result.x)
```

---

### 4.5 Linear Algebra

```python
A = np.array([[3, 2], [1, 4]])
b = np.array([7, 10])
solution = linalg.solve(A, b)
print("Solution:", solution)
```

---

## **5. Statsmodels vs SciPy**

* **Statsmodels** = Modeling + detailed statistical reports
* **SciPy** = General scientific computing + quick tests

---

# **Practice Questions**

### Beginner

1. Calculate mean, median, and standard deviation of 100 random numbers.
2. Perform a one-sample t-test using SciPy.
3. Create a simple OLS linear regression using Statsmodels.

### Intermediate

1. Perform an independent two-sample t-test.
2. Run a Chi-square test on a contingency table.
3. Fit an ARIMA model on random time series data.

### Advanced

1. Perform ANOVA with multiple groups.
2. Build a logistic regression model with Statsmodels.
3. Use SciPy to optimize a quadratic function.

---

# **Mini Projects**

### 1. **A/B Testing for Website Conversion**

* Data: Conversion rates for two designs
* Test: T-test (SciPy)
* Visualize: Boxplot with Seaborn

---

### 2. **Sales Forecasting using ARIMA**

* Dataset: Monthly sales data
* Tool: ARIMA (Statsmodels)
* Output: Future predictions

---

### 3. **Customer Satisfaction Survey Analysis**

* Data: Survey responses (Likert scale)
* Test: ANOVA & Chi-square
* Goal: Identify factors affecting satisfaction

---

### 4. **Hypothesis Testing for Marketing Campaign**

* Data: Click-through rates before/after campaign
* Test: Paired T-test
* Report: P-value & effect size

---

### 5. **Optimizing Production Cost**

* Function: Cost vs production units
* Tool: SciPy optimize
* Goal: Find minimum cost point

---

## **Saving Results**

```python
model.save('model_results.pickle')  # Statsmodels
# For SciPy: Save arrays or DataFrames using numpy or pandas
```

