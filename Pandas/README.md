# Pandas Library 

## **1. Introduction to Pandas**

* **What is Pandas?**
  Pandas is a powerful, open-source Python library used for **data analysis and manipulation**.
* **Key Features:**

  * Fast and efficient data structures (`Series` & `DataFrame`)
  * Handling of missing data
  * Data filtering, grouping, and aggregation
  * Integration with NumPy, Matplotlib, and Scikit-learn
* **Installation:**

  ```bash
  pip install pandas
  ```

---

## **2. Pandas Data Structures**

### 2.1 Series

* A **one-dimensional** labeled array.

```python
import pandas as pd
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
```

### 2.2 DataFrame

* A **two-dimensional** labeled data structure.

```python
data = {'Name': ['John', 'Emma', 'Alex'],
        'Age': [25, 30, 28]}
df = pd.DataFrame(data)
print(df)
```

---

## **3. Creating DataFrames**

```python
# From dictionary
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

# From list of lists
df = pd.DataFrame([[1,2],[3,4],[5,6]], columns=['X','Y'])

# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')
```

---

## **4. Viewing & Inspecting Data**

```python
df.head()      # First 5 rows
df.tail(3)     # Last 3 rows
df.info()      # Summary
df.describe()  # Statistical summary
df.shape       # Rows and columns
df.columns     # Column names
```

---

## **5. Selecting & Indexing**

```python
# Select a column
df['Name']

# Select multiple columns
df[['Name','Age']]

# Select rows by index
df.iloc[0:2]

# Select rows by labels
df.loc[0:2, ['Name','Age']]
```

---

## **6. Filtering Data**

```python
# Conditional filtering
df[df['Age'] > 25]

# Multiple conditions
df[(df['Age'] > 25) & (df['Name'] == 'Emma')]
```

---

## **7. Adding, Modifying & Deleting Columns**

```python
# Add a new column
df['Country'] = ['USA','UK','Canada']

# Modify column
df['Age'] = df['Age'] + 1

# Delete column
df.drop('Country', axis=1, inplace=True)
```

---

## **8. Handling Missing Data**

```python
df.isnull().sum()       # Count missing values
df.dropna()             # Drop missing rows
df.fillna(0, inplace=True)  # Fill missing values with 0
```

---

## **9. Sorting Data**

```python
df.sort_values('Age')                  # Ascending
df.sort_values('Age', ascending=False) # Descending
```

---

## **10. Grouping & Aggregation**

```python
# Group by column
grouped = df.groupby('Name')['Age'].mean()

# Multiple aggregations
df.groupby('Name').agg({'Age': ['mean','max','min']})
```

---

## **11. Merging, Joining & Concatenating**

```python
# Concatenate
pd.concat([df1, df2])

# Merge (similar to SQL joins)
pd.merge(df1, df2, on='ID', how='inner')
```

---

## **12. Exporting Data**

```python
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
```

---

## **13. Advanced Pandas Features**

* **Apply function:**

```python
df['Age'] = df['Age'].apply(lambda x: x + 2)
```

* **Pivot tables:**

```python
df.pivot_table(values='Age', index='Name', aggfunc='mean')
```

* **Handling duplicates:**

```python
df.drop_duplicates(inplace=True)
```

---

# **Pandas Practice Questions**

### Basic Level

1. Create a Pandas Series of numbers from 1–10.
2. Read a CSV file and print its first 10 rows.
3. Display the number of rows and columns in a DataFrame.
4. Retrieve only the "Name" and "Age" columns.

### Intermediate Level

1. Filter rows where salary is greater than 50,000.
2. Replace missing values in "Age" column with mean age.
3. Sort the dataset by "Salary" in descending order.

### Advanced Level

1. Group a dataset by "Department" and find average salary.
2. Merge two DataFrames using a common column.
3. Create a pivot table showing sales by region and product.

---

# **Pandas Mini Projects**

1. **Student Records Analysis**

   * Input: CSV file with student names, marks, and grades
   * Task: Find top 3 scorers, average marks, and pass/fail count.

2. **Sales Data Dashboard**

   * Analyze monthly sales data (CSV)
   * Calculate total sales, highest-selling product, and visualize trends.

3. **COVID-19 Data Analysis**

   * Download a COVID dataset
   * Show total cases, recoveries, and deaths by country.

4. **Bank Customer Data Cleaning**

   * Handle missing values, duplicates
   * Create a cleaned version of dataset.

5. **Employee Performance Report**

   * Group by department
   * Find average performance score and highest performer.

