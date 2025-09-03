
# OpenPyXL

---

## **1. Introduction**

* **OpenPyXL:** Python library for Excel automation.

* Supports:

  * Reading & Writing `.xlsx` files
  * Editing existing spreadsheets
  * Adding styles, charts, formulas
  * Automating Excel reports

* **Installation:**

```bash
pip install openpyxl
```

---

## **2. Importing OpenPyXL**

```python
import openpyxl
from openpyxl import Workbook, load_workbook
```

---

## **3. Creating a New Workbook**

```python
wb = Workbook()               # Create a new workbook
ws = wb.active                # Get active sheet
ws.title = "DataSheet"        # Rename sheet
wb.save("example.xlsx")       # Save workbook
```

---

## **4. Loading an Existing Workbook**

```python
wb = load_workbook("example.xlsx")
ws = wb.active
```

---

## **5. Writing Data to Cells**

```python
ws['A1'] = "Name"
ws['B1'] = "Age"
ws.append(["John", 25])  # Append row
ws.append(["Emma", 30])
wb.save("example.xlsx")
```

---

## **6. Reading Data from Cells**

```python
print(ws['A1'].value)      # Read single cell
for row in ws.iter_rows(values_only=True):
    print(row)
```

---

## **7. Modifying Data**

```python
ws['B2'] = 28   # Update cell
wb.save("example.xlsx")
```

---

## **8. Adding Multiple Sheets**

```python
sheet1 = wb.create_sheet("Sheet1")
sheet2 = wb.create_sheet("Sheet2", 0)  # Insert at first position
wb.save("example.xlsx")
```

---

## **9. Deleting Sheets**

```python
del wb['Sheet1']
wb.save("example.xlsx")
```

---

## **10. Styling Cells**

```python
from openpyxl.styles import Font, PatternFill, Alignment

ws['A1'].font = Font(bold=True, color="FFFFFF")
ws['A1'].fill = PatternFill(start_color="0000FF", fill_type="solid")
ws['A1'].alignment = Alignment(horizontal="center")
wb.save("example.xlsx")
```

---

## **11. Merging & Unmerging Cells**

```python
ws.merge_cells('A1:C1')
ws['A1'] = "Merged Title"
ws.unmerge_cells('A1:C1')
wb.save("example.xlsx")
```

---

## **12. Adding Formulas**

```python
ws['C2'] = "=SUM(B2:B5)"
wb.save("example.xlsx")
```

---

## **13. Working with Rows & Columns**

```python
ws.insert_rows(2)         # Insert row
ws.delete_rows(4)         # Delete row
ws.insert_cols(2)         # Insert column
ws.delete_cols(3)         # Delete column
wb.save("example.xlsx")
```

---

## **14. Adding Charts**

```python
from openpyxl.chart import BarChart, Reference

values = Reference(ws, min_col=2, max_col=2, min_row=2, max_row=5)
chart = BarChart()
chart.add_data(values)
ws.add_chart(chart, "E2")
wb.save("example.xlsx")
```

---

## **15. Data Validation (Dropdowns)**

```python
from openpyxl.worksheet.datavalidation import DataValidation

dv = DataValidation(type="list", formula1='"Yes,No,Maybe"', allow_blank=True)
ws.add_data_validation(dv)
dv.add(ws["C2"])
wb.save("example.xlsx")
```

---

## **16. Protecting Sheets**

```python
ws.protection.sheet = True
ws.protection.password = "1234"
wb.save("example.xlsx")
```

---

## **17. Reading Large Excel Files Efficiently**

```python
wb = load_workbook("large_file.xlsx", read_only=True)
for row in wb.active.iter_rows(values_only=True):
    print(row)
```

---

# **OpenPyXL Practice Questions**

### Beginner

1. Create a new Excel file with a sheet named “Students”.
2. Write names and marks for 5 students.
3. Save and verify the data.

### Intermediate

1. Add a column for “Grade” and calculate it using a formula.
2. Apply styles to header cells.
3. Merge the top row for a title.

### Advanced

1. Insert a bar chart for student marks.
2. Add a dropdown for “Pass/Fail” status.
3. Protect the sheet with a password.

---

# **Mini Projects (OpenPyXL)**

### 1. **Student Marks Management**

* Input: Names, Marks
* Features:

  * Auto-grade calculation
  * Pass/Fail dropdown
  * Save as `marks.xlsx`

---

### 2. **Monthly Expense Tracker**

* Input: Categories (Food, Rent, Travel, etc.)
* Features:

  * Auto total calculation with formulas
  * Bar chart visualization
  * Saves `expenses.xlsx`

---

### 3. **Attendance Tracker**

* Input: Names of employees/students
* Features:

  * Present/Absent dropdown
  * Auto-count attendance percentage
  * Highlight below 75%

---

### 4. **Sales Report Generator**

* Input: Product, Quantity, Price
* Features:

  * Auto-calculate total revenue
  * Apply styles & charts
  * Export as `sales_report.xlsx`

---

### 5. **Invoice Generator**

* Features:

  * Auto-fill product details
  * Calculate total price
  * Generate PDF (with `openpyxl` + `reportlab`)

