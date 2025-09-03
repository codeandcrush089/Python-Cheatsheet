
# Matplotlib & Seaborn 
---

## **1. Introduction**

* **Matplotlib:**
  A Python library for creating static, animated, and interactive visualizations.

  * Primary module: `matplotlib.pyplot`

* **Seaborn:**
  Built on top of Matplotlib, provides **high-level interface** for attractive and informative statistical graphics.

* **Installation:**

  ```bash
  pip install matplotlib seaborn
  ```

---

## **2. Importing Libraries**

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## **3. Matplotlib Basics**

### 3.1 Simple Line Plot

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, color='blue', marker='o', linestyle='--')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.grid(True)
plt.show()
```

---

### 3.2 Figure & Subplots

```python
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax[0, 0].plot(x, y)
ax[0, 1].bar(x, y)
ax[1, 0].scatter(x, y)
ax[1, 1].hist(y)
plt.tight_layout()
plt.show()
```

---

### 3.3 Bar Chart

```python
categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

plt.bar(categories, values, color='orange')
plt.title('Bar Chart')
plt.show()
```

---

### 3.4 Histogram

```python
import numpy as np
data = np.random.randn(1000)

plt.hist(data, bins=20, color='green', edgecolor='black')
plt.title('Histogram')
plt.show()
```

---

### 3.5 Scatter Plot

```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='red')
plt.title('Scatter Plot')
plt.show()
```

---

### 3.6 Pie Chart

```python
sizes = [30, 25, 20, 25]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart')
plt.show()
```

---

## **4. Seaborn Basics**

### 4.1 Sample Dataset

```python
tips = sns.load_dataset("tips")
tips.head()
```

---

### 4.2 Distribution Plot

```python
sns.histplot(tips['total_bill'], kde=True, color='blue')
plt.title('Distribution of Total Bill')
plt.show()
```

---

### 4.3 Count Plot

```python
sns.countplot(x='day', data=tips, palette='Set2')
plt.title('Count of Customers by Day')
plt.show()
```

---

### 4.4 Box Plot

```python
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Boxplot of Total Bill by Day')
plt.show()
```

---

### 4.5 Violin Plot

```python
sns.violinplot(x='day', y='total_bill', data=tips)
plt.title('Violin Plot')
plt.show()
```

---

### 4.6 Pair Plot

```python
sns.pairplot(tips, hue='sex')
plt.show()
```

---

### 4.7 Heatmap

```python
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

---

## **5. Customization & Styling**

```python
sns.set_style('darkgrid')       # Other: white, whitegrid, ticks
sns.set_palette('pastel')       # Set color palette
plt.figure(figsize=(8, 5))      # Adjust figure size
```

---

## **6. Saving Plots**

```python
plt.savefig('plot.png', dpi=300)
```

---

# **Matplotlib & Seaborn Practice Questions**

### Basic Level

1. Create a line chart showing `x = [1,2,3,4]` and `y = [2,4,6,8]`.
2. Draw a bar chart of marks of 5 students.
3. Create a histogram of 100 random numbers between 1–100.

### Intermediate Level

1. Use Seaborn to create a count plot of `tips` dataset showing customers by day.
2. Plot a scatter plot of two random variables using Matplotlib.
3. Create a box plot of `total_bill` vs `day` using Seaborn.

### Advanced Level

1. Visualize correlation matrix of the `tips` dataset using a heatmap.
2. Create a subplot with a histogram and a scatter plot in one figure.
3. Customize a Seaborn plot with a new color palette and style.

---

# **Mini Projects (Matplotlib & Seaborn)**

### 1. **Sales Data Visualization Dashboard**

* Load CSV file of monthly sales
* Show:

  * Line chart for sales over time
  * Bar chart for top-selling products
  * Pie chart for sales distribution by region

### 2. **COVID-19 Data Visualization**

* Visualize confirmed cases, recoveries, and deaths over time
* Create a heatmap for cases per country

### 3. **Student Performance Analysis**

* Plot marks distribution using histogram
* Use boxplot to analyze marks by subject
* Create a scatter plot for marks vs study hours

### 4. **E-commerce Customer Insights**

* Analyze spending patterns by day and gender
* Visualize average purchase using bar chart & violin plot

### 5. **Weather Data Visualization**

* Plot temperature trends (line chart)
* Show rainfall distribution (histogram)
* Heatmap for seasonal variations

