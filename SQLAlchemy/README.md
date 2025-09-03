# SQLAlchemy – Complete Notes

---

## **1. Introduction**

* **SQLAlchemy:** A Python library to interact with databases using:

  * **Core (SQL Expression Language)** – Write raw SQL-like queries
  * **ORM (Object Relational Mapper)** – Work with Python classes instead of SQL queries

* Supported Databases:

  * SQLite
  * MySQL
  * PostgreSQL
  * SQL Server
  * Oracle

* **Installation:**

```bash
pip install sqlalchemy
```

For MySQL or PostgreSQL:

```bash
pip install pymysql psycopg2
```

---

## **2. Importing SQLAlchemy**

```python
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
```

---

## **3. Connecting to a Database**

```python
# SQLite
engine = create_engine("sqlite:///example.db")

# MySQL
# engine = create_engine("mysql+pymysql://user:password@localhost:3306/dbname")

# PostgreSQL
# engine = create_engine("postgresql://user:password@localhost:5432/dbname")

connection = engine.connect()
```

---

## **4. Creating a Table (Core Method)**

```python
from sqlalchemy import MetaData, Table, Column

metadata = MetaData()

users = Table(
    'users', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(50)),
    Column('age', Integer)
)

metadata.create_all(engine)
```

---

## **5. Inserting Data (Core)**

```python
insert_query = users.insert().values(name="John", age=30)
connection.execute(insert_query)
```

---

## **6. Querying Data (Core)**

```python
from sqlalchemy import select

query = select(users)
result = connection.execute(query)
for row in result:
    print(row)
```

---

## **7. ORM Approach**

### Defining a Model

```python
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    age = Column(Integer)

Base.metadata.create_all(engine)
```

---

### Creating a Session

```python
Session = sessionmaker(bind=engine)
session = Session()
```

---

### Inserting Data (ORM)

```python
new_user = User(name="Emma", age=28)
session.add(new_user)
session.commit()
```

---

### Querying Data (ORM)

```python
users = session.query(User).all()
for user in users:
    print(user.name, user.age)
```

---

### Filtering Data

```python
young_users = session.query(User).filter(User.age < 30).all()
```

---

### Updating Data

```python
user = session.query(User).filter_by(name="Emma").first()
user.age = 29
session.commit()
```

---

### Deleting Data

```python
user = session.query(User).filter_by(name="John").first()
session.delete(user)
session.commit()
```

---

## **8. Relationships (One-to-Many)**

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class Address(Base):
    __tablename__ = 'addresses'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    city = Column(String(50))
    user = relationship("User", back_populates="addresses")

User.addresses = relationship("Address", order_by=Address.id, back_populates="user")
```

---

## **9. Transactions**

```python
from sqlalchemy.exc import SQLAlchemyError

try:
    session.add(User(name="Alice", age=23))
    session.commit()
except SQLAlchemyError:
    session.rollback()
```

---

## **10. Raw SQL Execution**

```python
result = session.execute("SELECT * FROM users")
for row in result:
    print(row)
```

---

## **11. Dropping Tables**

```python
Base.metadata.drop_all(engine)
```

---

## **12. Best Practices**

* Always use **sessions** for ORM.
* Handle **exceptions with rollback()**.
* Use **environment variables for database credentials**.

---

# **SQLAlchemy Practice Questions**

### Beginner

1. Create an SQLite database and a `students` table with `name` and `marks`.
2. Insert 5 records using Core.
3. Fetch all records and display them.

### Intermediate

1. Create a table with `users` and `addresses` (one-to-many relationship).
2. Update a user’s name.
3. Delete users with age below 18.

### Advanced

1. Build an ORM model with relationships for an e-commerce database.
2. Use filter, order\_by, and limit to extract top 5 customers.
3. Implement transactions with rollback on failure.

---

# **Mini Projects (SQLAlchemy)**

### 1. **Student Management System**

* Tables: Students, Courses
* Features:

  * Add students & enroll them in courses
  * Fetch all enrolled students
  * Update student details

---

### 2. **E-commerce Inventory Manager**

* Tables: Products, Orders
* Features:

  * Add/update products
  * Track stock
  * Fetch top-selling items

---

### 3. **Library Management System**

* Tables: Books, Members, BorrowedBooks
* Features:

  * Borrow/Return system
  * Track due dates
  * View all borrowed books

---

### 4. **Employee Payroll System**

* Tables: Employees, Salaries
* Features:

  * Calculate monthly salary
  * Fetch highest-paid employees
  * Generate salary report

---

### 5. **Blogging Platform**

* Tables: Users, Posts, Comments
* Features:

  * Add blog posts
  * Manage comments
  * Fetch posts by user


