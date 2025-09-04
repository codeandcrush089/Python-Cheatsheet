<p align="center">
  <img src="https://img.shields.io/badge/Python-Powerful%20%26%20Versatile%20Programming%20Language-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">🐍 Python – The Backbone of Modern Data & AI</h1>

<p align="center">
  Simple • Powerful • Scalable
</p>

---
## **1. Introduction to Python**

* **What is Python?**

  * Python is a high-level, interpreted, and versatile programming language.
  * Created by **Guido van Rossum in 1991**.
  * Emphasizes **readability, simplicity, and productivity**.

* **Key Features:**

  * Easy to learn & use
  * Open-source & free
  * Cross-platform
  * Object-Oriented & Procedural
  * Huge standard library & strong community

* **Applications:**

  * Web Development
  * Data Science & Machine Learning
  * Automation & Scripting
  * Game Development
  * APIs, IoT, Desktop Apps

---

## **2. Installing Python**

* Download from: [python.org/downloads](https://www.python.org/downloads/)
* Install Python 3.x (latest stable version).
* **Add to PATH during installation.**
* Verify installation:

  ```bash
  python --version
  ```

---

## **3. Python Basics**

### 3.1 Syntax

* Python uses **indentation** (4 spaces) instead of braces.
* Example:

  ```python
  if True:
      print("Hello, Python!")
  ```

### 3.2 Comments

```python
# Single-line comment

"""
Multi-line
comment
"""
```

### 3.3 Running Python

* **Interactive Mode:** Type `python` in terminal.
* **Script Mode:** Create a `.py` file and run:

  ```bash
  python myscript.py
  ```

---

## **4. Variables & Data Types**

### 4.1 Variables

* Dynamically typed: no need to declare type.

```python
x = 10
name = "John"
```

### 4.2 Data Types

* int
* float
* str
* bool
* None

### 4.3 Type Casting

```python
x = int("5")
y = float(10)
z = str(100)
```

---

## **5. Input & Output**

```python
# Output
print("Hello World")

# Input
name = input("Enter your name: ")
print("Welcome", name)
```

---

## **6. Operators**

* Arithmetic: `+ - * / // % **`
* Comparison: `== != > < >= <=`
* Logical: `and, or, not`
* Assignment: `=, +=, -=, *=, /=`
* Identity: `is, is not`
* Membership: `in, not in`

---

## **7. Control Flow**

### 7.1 Conditional Statements

```python
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")
```

### 7.2 Loops

**For Loop:**

```python
for i in range(5):
    print(i)
```

**While Loop:**

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

**Break & Continue:**

```python
for i in range(10):
    if i == 5:
        break
    print(i)
```

---

## **8. Functions**

```python
def greet(name):
    return f"Hello, {name}"

print(greet("John"))
```

* **Default Arguments:**

```python
def greet(name="Guest"):
    print("Hello", name)
```

* **Lambda Functions:**
* A lambda function in Python is a small, anonymous function defined with the lambda keyword. It can take any number of arguments but can only have one expression. The expression is evaluated and returned.
```python
add = lambda x, y: x + y
print(add(5, 3))
```

---

## **9. Data Structures**

### 9.1 List

```python
my_list = [1, 2, 3]
my_list.append(4)
print(my_list)
```

### 9.2 Tuple

```python
my_tuple = (1, 2, 3)
```

### 9.3 Set

```python
my_set = {1, 2, 3}
my_set.add(4)
```

### 9.4 Dictionary

```python
my_dict = {"name": "John", "age": 25}
print(my_dict["name"])
```

---

## **10. Strings**

```python
text = "Hello, Python"
print(text.upper())
print(text.lower())
print(text[0:5])
print(len(text))
```

* f-Strings:

```python
name = "John"
print(f"Hello, {name}")
```

---

## **11. File Handling**

```python
# Writing
with open("file.txt", "w") as f:
    f.write("Hello, Python")

# Reading
with open("file.txt", "r") as f:
    print(f.read())
```

---

## **12. Exception Handling**

```python
try:
    x = int("abc")
except ValueError:
    print("Invalid number")
finally:
    print("Done")
```

---

## **13. Object-Oriented Programming (OOP)**

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name}")

p = Person("John", 25)
p.greet()
```

* Inheritance:

```python
class Student(Person):
    pass
```

---

## **14. Modules & Packages**

```python
import math
print(math.sqrt(16))
```

Create your own module:

```python
# mymodule.py
def greet():
    print("Hello from module!")
```

```python
import mymodule
mymodule.greet()
```

---

# **Python Practice Questions + Mini Projects**

---

## **1. Introduction to Python**

### Practice Questions:

1. What is Python? List its key features.
2. Name 5 real-world applications of Python.
3. Explain why Python is called an interpreted language.

---

## **2. Variables & Data Types**

### Practice Questions:

1. Create a variable for each data type (int, float, string, bool).
2. Convert a string `"25"` into an integer and multiply by 2.
3. What is dynamic typing? Give an example.

---

## **3. Operators**

### Practice Questions:

1. Write a program to calculate the area of a rectangle.
2. Demonstrate the difference between `/` and `//`.
3. Use logical operators to check if a number is between 10 and 50.

---

## **4. Control Flow (if-else)**

### Practice Questions:

1. Write a program to check if a number is positive, negative, or zero.
2. Accept a user's age and decide if they are eligible to vote.
3. Find the largest among three numbers using if-elif-else.

---

## **5. Loops**

### Practice Questions:

1. Print numbers from 1 to 20 using a `for` loop.
2. Create a multiplication table for a number.
3. Write a program to find the sum of all even numbers between 1–50.
4. Reverse a string using a loop.

---

## **6. Functions**

### Practice Questions:

1. Write a function to find the square of a number.
2. Create a function that checks if a number is prime.
3. Write a function to calculate factorial of a number.

---

## **7. Data Structures**

### Practice Questions:

1. Create a list of fruits and:

   * Add a new fruit
   * Remove one fruit
   * Print its length
2. Create a dictionary with student names as keys and marks as values. Retrieve marks of a specific student.
3. Write a program to remove duplicates from a list.

---

## **8. Strings**

### Practice Questions:

1. Reverse a string.
2. Count vowels in a string.
3. Check if a string is a palindrome.

---

## **9. File Handling**

### Practice Questions:

1. Write a program to create a text file and write “Hello Python”.
2. Read a file and count how many lines it has.
3. Append a new line to an existing file.

---

## **10. Exception Handling**

### Practice Questions:

1. Handle `ZeroDivisionError` when dividing two numbers.
2. Write a program that keeps asking for an integer until a valid number is entered.

---

## **11. Object-Oriented Programming**

### Practice Questions:

1. Create a class `Car` with attributes brand and model. Create an object and display its details.
2. Implement inheritance with a base class `Animal` and a derived class `Dog`.
3. Add a method in the class to calculate the age of a person from their birth year.

---

# **Mini Projects (Beginner-Friendly)**

1. **Calculator App**

   * Input: two numbers + operation (+, -, \*, /)
   * Output: result

2. **Guess the Number Game**

   * Computer selects a random number between 1–100.
   * User keeps guessing until correct.

3. **Simple To-Do List (Console-based)**

   * Add, delete, and view tasks.

4. **Student Grade Calculator**

   * Input: marks of subjects.
   * Output: average and grade.

5. **Basic Password Generator**

   * Generate a random strong password with letters, numbers, and symbols.



