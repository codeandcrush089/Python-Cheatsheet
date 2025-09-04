<p align="center">
  <img src="https://img.shields.io/badge/Plotly%20Library-Interactive%20Data%20Visualization-00BFA5?style=for-the-badge&logo=python&logoColor=white" alt="Plotly" />
</p>

<h1 align="center">📈 Plotly – Interactive & Dynamic Visualizations</h1>

<p align="center">
  Interactive • Analytical • Engaging
</p>


---

## **1. Introduction**

* **Plotly:**
  An interactive visualization library for Python.

  * Great for dashboards, reports, and web-based visualizations.
  * Works seamlessly with **Jupyter Notebook, Dash, and web apps**.

* **Installation:**

  ```bash
  pip install plotly
  ```

* **Two main interfaces:**

  1. `plotly.graph_objects` – Low-level, flexible
  2. `plotly.express` – High-level, easy-to-use

---

## **2. Importing Plotly**

```python
import plotly.express as px
import plotly.graph_objects as go
```

---

## **3. Basic Plotly Express Examples**

### 3.1 Line Chart

```python
import pandas as pd

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 15, 25, 30]
})

fig = px.line(data, x='x', y='y', title='Simple Line Chart')
fig.show()
```

---

### 3.2 Scatter Plot

```python
df = px.data.iris()

fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species',
                 title='Iris Dataset Scatter Plot')
fig.show()
```

---

### 3.3 Bar Chart

```python
data = {'Category': ['A', 'B', 'C', 'D'], 'Values': [10, 25, 17, 30]}
df = pd.DataFrame(data)

fig = px.bar(df, x='Category', y='Values', color='Category', title='Bar Chart')
fig.show()
```

---

### 3.4 Histogram

```python
df = px.data.tips()
fig = px.histogram(df, x='total_bill', nbins=20, color='sex',
                   title='Histogram of Total Bill')
fig.show()
```

---

### 3.5 Pie Chart

```python
fig = px.pie(df, names='day', values='total_bill', title='Sales by Day')
fig.show()
```

---

### 3.6 Box Plot

```python
fig = px.box(df, x='day', y='total_bill', color='sex', title='Boxplot of Total Bill')
fig.show()
```

---

## **4. Plotly Graph Objects Examples**

### 4.1 Custom Line Plot

```python
fig = go.Figure()

fig.add_trace(go.Scatter(x=[1,2,3,4,5],
                         y=[10,15,13,17,22],
                         mode='lines+markers',
                         name='Sales'))

fig.update_layout(title='Custom Line Plot',
                  xaxis_title='X-Axis',
                  yaxis_title='Y-Axis')
fig.show()
```

---

### 4.2 Multiple Traces

```python
fig = go.Figure()

fig.add_trace(go.Bar(x=['A','B','C'], y=[10,20,30], name='2023'))
fig.add_trace(go.Bar(x=['A','B','C'], y=[15,25,20], name='2024'))

fig.update_layout(barmode='group', title='Grouped Bar Chart')
fig.show()
```

---

## **5. 3D & Advanced Visualizations**

### 5.1 3D Scatter

```python
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species', title='3D Scatter Plot')
fig.show()
```

---

### 5.2 Choropleth Map

```python
gapminder = px.data.gapminder().query("year==2007")
fig = px.choropleth(gapminder, locations="iso_alpha",
                    color="gdpPercap",
                    hover_name="country",
                    title='World GDP Per Capita (2007)')
fig.show()
```

---

## **6. Customization**

* **Themes:**

```python
px.defaults.template = "plotly_dark"
```

* **Size:**

```python
fig.update_layout(width=800, height=500)
```

* **Save as HTML/PNG:**

```python
fig.write_html("plot.html")
fig.write_image("plot.png")  # requires kaleido: pip install -U kaleido
```

---

# **Plotly Practice Questions**

### Beginner

1. Create a simple line chart showing monthly revenue.
2. Generate a histogram for 100 random numbers.
3. Make a pie chart showing market share of 4 companies.

### Intermediate

1. Create a scatter plot with hover tooltips using the Iris dataset.
2. Build a grouped bar chart showing 2 years of sales.
3. Make a box plot for `total_bill` grouped by `day` using `tips` dataset.

### Advanced

1. Build a 3D scatter plot using random data.
2. Create a choropleth map showing world population.
3. Add multiple traces (line + bar) in a single figure with custom titles.

---

# **Mini Projects (Plotly)**

### 1. **Interactive Sales Dashboard**

* Data: Monthly sales by region
* Visuals:

  * Line chart (sales over time)
  * Bar chart (region-wise comparison)
  * Pie chart (market share)
* Export as **interactive HTML dashboard**

---

### 2. **COVID-19 Data Visualizer**

* Fetch COVID-19 data (cases, deaths, recoveries)
* Use line charts for trends
* Choropleth map for global cases

---

### 3. **Stock Market Visualizer**

* Fetch stock prices using Yahoo Finance API
* Plot interactive candlestick chart
* Add moving average overlays

---

### 4. **Customer Purchase Analysis**

* Dataset: E-commerce purchase history
* Visuals:

  * Histogram of purchase amounts
  * Boxplot for customer segments
  * Scatter plot for age vs spending

---

### 5. **Weather Dashboard**

* Data: Temperature, humidity, rainfall
* Line chart for temperature trends
* Heatmap for seasonal rainfall patterns
* Pie chart for weather type distribution

