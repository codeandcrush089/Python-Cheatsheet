<p align="center">
  <img src="https://img.shields.io/badge/BeautifulSoup%20Library-HTML%20%26%20XML%20Parsing-4CAF50?style=for-the-badge&logo=python&logoColor=white" alt="BeautifulSoup" />
</p>

<h1 align="center">🍵 BeautifulSoup – Powerful Web Scraping Made Easy</h1>

<p align="center">
  Parse • Extract • Automate
</p>


## **1. Introduction**

* **BeautifulSoup:** A Python library to parse and extract data from HTML and XML documents.

* Often used with:

  * **`requests`** (to fetch web pages)
  * **`lxml`** or **`html.parser`** (parsing engines)

* **Installation:**

```bash
pip install beautifulsoup4 requests lxml
```

---

## **2. Importing BeautifulSoup**

```python
from bs4 import BeautifulSoup
import requests
```

---

## **3. Fetching a Web Page**

```python
url = "https://example.com"
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, "lxml")
```

---

## **4. Parsing HTML**

```python
print(soup.title)          # Get <title> tag
print(soup.title.text)     # Get text of <title>
print(soup.p)              # First <p> tag
```

---

## **5. Finding Elements**

### Find by Tag

```python
soup.find("h1")
soup.find_all("p")
```

### Find by Class & ID

```python
soup.find("div", class_="content")
soup.find(id="main-section")
```

### CSS Selectors

```python
soup.select(".class-name")      # Class selector
soup.select("#id-name")         # ID selector
soup.select("div > p")          # Child selector
```

---

## **6. Extracting Attributes**

```python
link = soup.find("a")
print(link["href"])             # Get href attribute
```

---

## **7. Extracting Text**

```python
for p in soup.find_all("p"):
    print(p.get_text())
```

---

## **8. Navigating the HTML Tree**

```python
div = soup.find("div")
print(div.parent)       # Parent tag
print(div.contents)     # All child elements
print(div.next_sibling) # Next element
```

---

## **9. Modifying the HTML**

```python
tag = soup.find("h1")
tag.string = "New Title"
```

---

## **10. Removing Tags**

```python
for script in soup(["script", "style"]):
    script.decompose()
```

---

## **11. Saving Extracted Data**

```python
data = [p.get_text() for p in soup.find_all("p")]
with open("output.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(line + "\n")
```

---

## **12. Handling Pagination**

```python
for i in range(1, 6):
    url = f"https://example.com/page/{i}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    # Extract data
```

---

## **13. Best Practices**

* Use **`time.sleep()`** between requests (avoid blocking).
* Respect **robots.txt**.
* Avoid scraping sensitive or copyrighted content without permission.

---

# **BeautifulSoup Practice Questions**

### Beginner

1. Fetch a web page and print its title.
2. Extract all links (`<a>` tags) from a page.
3. Find all paragraphs and print their text.

### Intermediate

1. Extract all product names and prices from a sample e-commerce page.
2. Get all image URLs from a web page.
3. Remove all `<script>` and `<style>` tags before saving HTML.

### Advanced

1. Scrape multiple pages with pagination.
2. Extract job listings (title, company, location) from a job portal.
3. Save scraped data to a CSV or Excel file.

---

# **Mini Projects (BeautifulSoup)**

### 1. **News Headlines Scraper**

* Input: News website
* Features:

  * Scrape latest headlines & URLs
  * Save to CSV
  * Update daily using scheduler

---

### 2. **E-commerce Price Tracker**

* Input: Product URL
* Features:

  * Extract product name & price
  * Track price history in Excel
  * Send alert if price drops

---

### 3. **IMDb Top Movies Scraper**

* Dataset: IMDb Top 250
* Features:

  * Extract movie name, year, rating
  * Save as CSV/JSON
  * Sort by rating

---

### 4. **Job Listings Extractor**

* Input: Job portal (Indeed/LinkedIn)
* Features:

  * Scrape job title, company, location
  * Filter by keywords
  * Export to Excel

---

### 5. **Wikipedia Info Extractor**

* Input: Topic (e.g., "Python programming")
* Features:

  * Extract first paragraph
  * Save summary as `.txt` file

