# Pandas: Complete Guide for Data Analysis

Pandas is the essential library for data manipulation and analysis in Python. Built on top of NumPy, it provides powerful, flexible data structures and data analysis tools for working with structured data.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Data Structures: Series & DataFrame](#data-structures-series--dataframe)
- [Data Creation & I/O](#data-creation--io)
- [Indexing and Selection](#indexing-and-selection)
- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [GroupBy Operations](#groupby-operations)
- [Merging, Joining & Concatenation](#merging-joining--concatenation)
- [Reshaping Data](#reshaping-data)
- [Time Series](#time-series)
- [String Operations](#string-operations)
- [Categorical Data](#categorical-data)
- [Window Functions](#window-functions)
- [Performance Optimization](#performance-optimization)
- [ML/Data Science Patterns](#mldata-science-patterns)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)

---

## Core Concepts

### Why Pandas?

**Labeled Data**: Unlike NumPy, pandas provides explicit index/column labels, making data self-documenting and easier to manipulate.

**Heterogeneous Data**: DataFrames can hold different data types in different columns (strings, integers, floats, dates, etc.).

**Missing Data Handling**: Built-in support for missing data (NaN, None) with powerful tools for detection and handling.

**Relational Operations**: SQL-like operations (join, merge, group by) built into the API.

**Time Series**: Powerful date/time functionality with frequency-based indexing.

```python
import pandas as pd
import numpy as np

# The pandas way
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Labeled access
df['name']  # Get column by name
df.loc[0]   # Get row by label

# vs NumPy (unlabeled)
arr = np.array([[25, 50000], [30, 60000], [35, 70000]])
arr[:, 0]   # Get column by position (what does column 0 mean?)
```

### Key Design Principles

1. **Explicit is better than implicit**: Use `.loc[]` and `.iloc[]` for clarity
2. **Chaining**: Operations can be chained for readable data pipelines
3. **Copy vs View**: Be aware of when operations create copies vs views
4. **Vectorization**: Avoid loops, use vectorized operations
5. **Index matters**: The index is a first-class citizen in pandas

---

## Data Structures: Series & DataFrame

### Series: 1D Labeled Array

```python
# Creating Series
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# Custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s['a'])  # 1

# From dictionary
data = {'a': 1, 'b': 2, 'c': 3}
s = pd.Series(data)

# Series attributes
s.values    # Underlying NumPy array
s.index     # Index object
s.dtype     # Data type
s.shape     # Shape tuple
s.size      # Number of elements
s.name      # Series name
```

### Series Operations

```python
s = pd.Series([1, 2, 3, 4, 5])

# Vectorized operations
s + 10      # Add 10 to all elements
s * 2       # Multiply by 2
s ** 2      # Square all elements
np.sqrt(s)  # NumPy functions work

# Statistical methods
s.mean()
s.std()
s.median()
s.quantile(0.75)
s.sum()
s.cumsum()  # Cumulative sum
s.min(), s.max()

# Boolean operations
s[s > 2]         # Filter
s.between(2, 4)  # Values between 2 and 4
s.isin([1, 3, 5])  # Check membership
```

### DataFrame: 2D Labeled Array

```python
# Creating DataFrames
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': ['w', 'x', 'y', 'z']
})

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob', 'age': 30, 'city': 'LA'}
]
df = pd.DataFrame(data)

# From NumPy array
arr = np.random.randn(4, 3)
df = pd.DataFrame(arr,
                  columns=['A', 'B', 'C'],
                  index=['row1', 'row2', 'row3', 'row4'])

# From dict of Series
df = pd.DataFrame({
    'A': pd.Series([1, 2, 3]),
    'B': pd.Series([4, 5, 6])
})
```

### DataFrame Attributes and Methods

```python
# Basic information
df.shape        # (rows, columns)
df.size         # Total elements
df.columns      # Column names
df.index        # Row index
df.dtypes       # Data types of each column
df.info()       # Summary information
df.describe()   # Statistical summary

# Quick views
df.head(n=5)    # First n rows
df.tail(n=5)    # Last n rows
df.sample(n=5)  # Random n rows

# Column access
df['A']         # Returns Series
df[['A', 'B']]  # Returns DataFrame
df.A            # Attribute access (only if valid Python identifier)
```

### Index Object

```python
# Creating custom index
df = pd.DataFrame({'A': [1, 2, 3]},
                  index=['row1', 'row2', 'row3'])

# Index operations
df.index = ['a', 'b', 'c']  # Set new index
df.reset_index()             # Reset to default integer index
df.reset_index(drop=True)    # Reset without keeping old index
df.set_index('column_name')  # Set column as index

# Multi-level index (hierarchical)
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

# Accessing multi-index
df.loc['A']           # All rows where first level is 'A'
df.loc[('A', 1)]      # Specific row
df.xs('A', level=0)   # Cross-section
```

---

## Data Creation & I/O

### Reading Data from Files

```python
# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv',
                 sep=';',              # Delimiter
                 header=0,             # Row for column names
                 index_col=0,          # Column to use as index
                 usecols=['A', 'B'],   # Columns to read
                 dtype={'A': int},     # Specify dtypes
                 parse_dates=['date'], # Parse dates
                 na_values=['?', 'N/A'],  # Additional NA values
                 encoding='utf-8',     # Character encoding
                 nrows=1000)           # Read first 1000 rows

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # First sheet

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')  # List of dicts
df = pd.read_json('data.json', orient='index')    # Dict of dicts

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df = pd.read_sql_query('SELECT * FROM table WHERE id > 10', conn)
df = pd.read_sql_table('table_name', conn)

# Parquet (efficient columnar format)
df = pd.read_parquet('data.parquet')

# HTML tables
dfs = pd.read_html('https://example.com/page.html')  # Returns list

# Clipboard
df = pd.read_clipboard()  # Read from clipboard
```

### Writing Data to Files

```python
# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv',
          sep='\t',              # Tab-separated
          columns=['A', 'B'],    # Select columns
          header=True,           # Include header
          index=True,            # Include index
          encoding='utf-8')

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# JSON
df.to_json('output.json', orient='records', indent=2)

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)
# if_exists: 'fail', 'replace', 'append'

# Parquet
df.to_parquet('output.parquet', compression='gzip')

# HTML
df.to_html('output.html', index=False)

# Pickle (preserves all pandas types)
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')
```

### Creating DataFrames from Scratch

```python
# From dictionary
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# From list of lists
data = [[1, 4], [2, 5], [3, 6]]
df = pd.DataFrame(data, columns=['A', 'B'])

# From NumPy array
arr = np.random.randn(100, 5)
df = pd.DataFrame(arr, columns=list('ABCDE'))

# Empty DataFrame with schema
df = pd.DataFrame(columns=['name', 'age', 'city'])

# Date range DataFrame
dates = pd.date_range('2023-01-01', periods=100, freq='D')
df = pd.DataFrame({'date': dates, 'value': np.random.randn(100)})

# From records
records = [
    (1, 'Alice', 25),
    (2, 'Bob', 30),
    (3, 'Charlie', 35)
]
df = pd.DataFrame.from_records(records, columns=['id', 'name', 'age'])
```

---

## Indexing and Selection

### Column Selection

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
})

# Single column (returns Series)
df['A']
df.A  # Only if column name is valid Python identifier

# Multiple columns (returns DataFrame)
df[['A', 'B']]

# Column slicing (by position, not label!)
df.iloc[:, 0:2]  # First two columns

# Select columns by type
df.select_dtypes(include=['int64'])
df.select_dtypes(exclude=['object'])
df.select_dtypes(include=[np.number])  # All numeric

# Filter columns by name pattern
df.filter(like='col')       # Contains 'col'
df.filter(regex='^col')     # Starts with 'col'
df.filter(items=['A', 'B']) # Exact names
```

### Row Selection

```python
# By position (integer-location based)
df.iloc[0]        # First row
df.iloc[-1]       # Last row
df.iloc[0:3]      # First three rows
df.iloc[[0, 2]]   # First and third rows

# By label (label-based)
df.loc[0]         # Row with index label 0
df.loc[0:2]       # INCLUSIVE slicing!
df.loc[['a', 'c']]  # Specific labels

# Boolean indexing
df[df['A'] > 2]
df[df['B'].isin([10, 30])]
df[(df['A'] > 2) & (df['B'] < 40)]  # Multiple conditions
df[df['A'].between(2, 3)]

# Query method (SQL-like)
df.query('A > 2')
df.query('A > 2 and B < 40')
df.query('A in [1, 3]')
```

### Combined Selection: loc and iloc

```python
# loc: [rows, columns] by label
df.loc[0, 'A']              # Single value
df.loc[0:2, ['A', 'B']]     # Rows 0-2, columns A and B
df.loc[:, 'A':'C']          # All rows, columns A through C
df.loc[df['A'] > 2, 'B']    # Boolean row selection, column B

# iloc: [rows, columns] by integer position
df.iloc[0, 0]               # First row, first column
df.iloc[0:2, 0:2]           # First 2 rows, first 2 columns
df.iloc[:, [0, 2]]          # All rows, first and third columns
df.iloc[df['A'].values > 2, 1]  # Boolean with iloc (convert to bool array)

# at and iat: Fast scalar access
df.at[0, 'A']    # By label (faster than loc for scalars)
df.iat[0, 0]     # By position (faster than iloc for scalars)
```

### Boolean Indexing Deep Dive

```python
# Simple boolean masks
mask = df['A'] > 2
df[mask]

# Compound conditions (use & | ~ not 'and' 'or' 'not')
df[(df['A'] > 2) & (df['B'] < 40)]   # AND
df[(df['A'] > 2) | (df['B'] < 40)]   # OR
df[~(df['A'] > 2)]                    # NOT

# Using isin for membership
df[df['A'].isin([1, 3, 5])]

# String methods
df[df['name'].str.contains('Alice')]
df[df['name'].str.startswith('A')]

# Null checks
df[df['A'].isna()]
df[df['A'].notna()]

# Multiple column conditions
df[df[['A', 'B']].apply(lambda x: x.sum() > 50, axis=1)]

# Query with variables
threshold = 2
df.query('A > @threshold')  # @ for external variables
```

### Advanced Indexing

```python
# Fancy indexing with lists
rows = [0, 2, 4]
cols = ['A', 'C']
df.loc[rows, cols]

# Boolean indexing with assignment
df.loc[df['A'] > 2, 'B'] = 999  # Set values where condition is True

# MultiIndex selection
df.loc[('A', 1), :]  # Tuple for multi-level index
df.xs('A', level=0)   # Cross-section

# IndexSlice for complex multi-index selection
idx = pd.IndexSlice
df.loc[idx[:, 'value'], :]  # All first level, 'value' in second level
```

---

## Data Cleaning

### Handling Missing Data

```python
# Detecting missing values
df.isna()         # Boolean DataFrame
df.isna().sum()   # Count per column
df.isna().sum().sum()  # Total missing values
df.isnull()       # Alias for isna()

# Visualize missing patterns
df.isna().sum().plot(kind='bar')

# Check if any/all missing
df.isna().any()   # Any missing per column
df.isna().all()   # All missing per column

# Drop missing values
df.dropna()                    # Drop rows with any NA
df.dropna(how='all')          # Drop rows where all values are NA
df.dropna(subset=['A', 'B'])  # Drop rows with NA in specific columns
df.dropna(axis=1)             # Drop columns with any NA
df.dropna(thresh=2)           # Keep rows with at least 2 non-NA values

# Fill missing values
df.fillna(0)                  # Fill with constant
df.fillna({'A': 0, 'B': 99})  # Different values per column
df.fillna(method='ffill')     # Forward fill
df.fillna(method='bfill')     # Backward fill
df.fillna(df.mean())          # Fill with column mean
df.fillna(df.median())        # Fill with column median
df.fillna(df.mode().iloc[0])  # Fill with mode

# Interpolate missing values
df.interpolate()                    # Linear interpolation
df.interpolate(method='polynomial', order=2)  # Polynomial
df.interpolate(method='time')       # Time-based interpolation

# Replace specific values
df.replace(0, np.nan)         # Replace 0 with NaN
df.replace([0, 1], [100, 200])  # Multiple replacements
df.replace({'A': 0}, 100)     # Column-specific replacement
```

### Handling Duplicates

```python
# Detect duplicates
df.duplicated()                    # Boolean Series
df.duplicated().sum()              # Count duplicates
df.duplicated(subset=['A'])        # Check specific columns
df.duplicated(keep='first')        # Mark all but first as duplicate
df.duplicated(keep='last')         # Mark all but last as duplicate
df.duplicated(keep=False)          # Mark all duplicates (including first)

# Get duplicate rows
df[df.duplicated()]
df[df.duplicated(subset=['name'], keep=False)]

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['name'])  # Based on specific columns
df.drop_duplicates(keep='last')      # Keep last occurrence
df.drop_duplicates(inplace=True)     # Modify in place
```

### Data Type Conversion

```python
# Check types
df.dtypes
df['A'].dtype

# Convert types
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(float)
df['C'] = df['C'].astype(str)

# Convert to categorical
df['category'] = df['category'].astype('category')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Invalid -> NaT

# Convert to numeric (handles errors)
df['number'] = pd.to_numeric(df['number'], errors='coerce')  # Invalid -> NaN
df['number'] = pd.to_numeric(df['number'], errors='ignore')  # Leave as-is

# Infer better dtypes
df = df.infer_objects()
df = df.convert_dtypes()  # Use nullable dtypes (pd.Int64, pd.StringDtype)
```

### String Cleaning

```python
# Strip whitespace
df['name'] = df['name'].str.strip()
df['name'] = df['name'].str.lstrip()
df['name'] = df['name'].str.rstrip()

# Case conversion
df['name'] = df['name'].str.lower()
df['name'] = df['name'].str.upper()
df['name'] = df['name'].str.title()

# Replace patterns
df['text'] = df['text'].str.replace('old', 'new')
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)  # Remove digits

# Remove special characters
df['clean'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Extract patterns
df['number'] = df['text'].str.extract(r'(\d+)')
df[['area', 'phone']] = df['full_phone'].str.extract(r'(\d{3})-(\d{7})')
```

### Outlier Detection and Handling

```python
# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
df_no_outliers = df[z_scores < 3]

# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# Clip outliers
df['value_clipped'] = df['value'].clip(lower_bound, upper_bound)

# Winsorize
from scipy.stats.mstats import winsorize
df['value_winsorized'] = winsorize(df['value'], limits=[0.05, 0.05])
```

---

## Data Transformation

### Sorting

```python
# Sort by values
df.sort_values('A')                          # Ascending
df.sort_values('A', ascending=False)         # Descending
df.sort_values(['A', 'B'])                   # Multiple columns
df.sort_values(['A', 'B'], ascending=[True, False])  # Mixed order

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Sort with custom key
df.sort_values('name', key=lambda x: x.str.lower())
```

### Filtering

```python
# Boolean filtering
df[df['age'] > 25]
df[df['name'].str.contains('Alice')]

# Query method
df.query('age > 25')
df.query('age > 25 and city == "NYC"')
df.query('name.str.contains("Alice")', engine='python')

# Using where (keeps shape, fills non-matching with NaN)
df.where(df['age'] > 25)

# Using mask (opposite of where)
df.mask(df['age'] <= 25)

# Filter by callable
df[lambda x: x['age'] > 25]
```

### Adding and Removing Columns

```python
# Add new column
df['new_col'] = 0
df['sum'] = df['A'] + df['B']
df['ratio'] = df['A'] / df['B']

# Add column from function
df['squared'] = df['A'].apply(lambda x: x ** 2)

# Add column with assign (returns new DataFrame)
df = df.assign(
    new1=lambda x: x['A'] * 2,
    new2=lambda x: x['new1'] + 10
)

# Insert column at specific position
df.insert(1, 'new_col', [1, 2, 3, 4])

# Remove columns
df.drop('col_name', axis=1)
df.drop(['col1', 'col2'], axis=1)
df.drop(columns=['col1', 'col2'])

# Remove columns in place
df.drop('col_name', axis=1, inplace=True)

# Select all except certain columns
df.drop(columns=df.columns.difference(['keep1', 'keep2']))
```

### Renaming

```python
# Rename columns
df.rename(columns={'old': 'new'})
df.rename(columns={'A': 'col_A', 'B': 'col_B'})

# Rename with function
df.rename(columns=str.lower)
df.rename(columns=lambda x: x.replace(' ', '_'))

# Set column names directly
df.columns = ['col1', 'col2', 'col3']

# Rename index
df.rename(index={0: 'row1', 1: 'row2'})

# Add prefix/suffix
df.add_prefix('col_')
df.add_suffix('_x')
```

### Apply, Map, and Applymap

```python
# apply: Apply function along axis
df['A'].apply(lambda x: x ** 2)              # Series apply
df.apply(lambda x: x.max() - x.min())        # Column-wise (axis=0, default)
df.apply(lambda x: x.max() - x.min(), axis=1)  # Row-wise

# apply with multiple return values
df.apply(lambda x: pd.Series([x.min(), x.max()]), axis=1)

# map: Element-wise transformation (Series only)
df['category'].map({'A': 1, 'B': 2, 'C': 3})
df['value'].map(lambda x: x * 2)

# applymap: Element-wise transformation (entire DataFrame) - DEPRECATED
# Use .map() instead
df.map(lambda x: x * 2)

# replace: Value replacement
df.replace({'A': {'old': 'new'}})
df['status'].replace({'active': 1, 'inactive': 0})

# Vectorized string operations (preferred over apply)
df['text'].str.upper()  # Better than df['text'].apply(str.upper)

# Vectorized operations (always prefer these)
df['A'] * 2  # Better than df['A'].apply(lambda x: x * 2)
```

### Binning and Discretization

```python
# Cut: Bin continuous data into discrete intervals
ages = [1, 5, 10, 15, 20, 25, 30, 35, 40]
bins = [0, 12, 18, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']
pd.cut(ages, bins=bins, labels=labels)

# Equal-width bins
pd.cut(df['value'], bins=5)  # 5 equal-width bins

# qcut: Quantile-based discretization
pd.qcut(df['value'], q=4)  # Quartiles
pd.qcut(df['value'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Custom function for binning
def age_group(age):
    if age < 18: return 'Minor'
    elif age < 65: return 'Adult'
    else: return 'Senior'

df['age_group'] = df['age'].apply(age_group)

# np.select for multiple conditions
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70,
    df['score'] >= 60
]
choices = ['A', 'B', 'C', 'D']
df['grade'] = np.select(conditions, choices, default='F')
```

### Rank and Quantile

```python
# Rank
df['rank'] = df['score'].rank()
df['rank'] = df['score'].rank(ascending=False)  # Higher score = rank 1
df['rank'] = df['score'].rank(method='dense')    # No gaps in ranking
df['rank'] = df['score'].rank(method='min')      # Ties get minimum rank
df['rank'] = df['score'].rank(pct=True)          # Percentile ranks

# Quantile
df['score'].quantile(0.25)  # 25th percentile
df['score'].quantile([0.25, 0.5, 0.75])  # Multiple quantiles
df.quantile(0.5)  # Median of all numeric columns
```

---

## GroupBy Operations

### Basic GroupBy

```python
# Group by single column
grouped = df.groupby('category')

# Group by multiple columns
grouped = df.groupby(['category', 'region'])

# Aggregation
df.groupby('category')['value'].sum()
df.groupby('category')['value'].mean()
df.groupby('category')['value'].count()
df.groupby('category').size()  # Count including NaN

# Multiple aggregations
df.groupby('category').agg({
    'value': 'sum',
    'quantity': 'mean',
    'price': ['min', 'max']
})

# Named aggregations (pandas 0.25+)
df.groupby('category').agg(
    total_value=('value', 'sum'),
    avg_quantity=('quantity', 'mean'),
    max_price=('price', 'max')
)

# Apply same aggregation to all columns
df.groupby('category').sum()
df.groupby('category').mean()
```

### Advanced GroupBy

```python
# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

df.groupby('category')['value'].agg(range_func)
df.groupby('category')['value'].agg(['sum', 'mean', range_func])

# agg with lambda
df.groupby('category')['value'].agg(lambda x: x.max() - x.min())

# Multiple columns, multiple functions
df.groupby('category').agg({
    'value': ['sum', 'mean', 'std'],
    'quantity': ['min', 'max'],
    'price': lambda x: x.median()
})

# transform: Return same-shaped object with group-wise operations
df['value_norm'] = df.groupby('category')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Group-wise centering
df['centered'] = df.groupby('category')['value'].transform(lambda x: x - x.mean())

# filter: Filter groups based on group properties
df.groupby('category').filter(lambda x: x['value'].sum() > 100)
df.groupby('category').filter(lambda x: len(x) >= 5)  # Groups with >=5 members
```

### Iteration and Group-wise Operations

```python
# Iterate over groups
for name, group in df.groupby('category'):
    print(f"Group: {name}")
    print(group)
    print()

# Get specific group
df.groupby('category').get_group('A')

# Apply custom function to each group
def process_group(group):
    group['normalized'] = (group['value'] - group['value'].mean()) / group['value'].std()
    return group

df = df.groupby('category').apply(process_group)

# apply vs transform
# apply: Can change shape, returns DataFrame
# transform: Must return same shape, returns Series/DataFrame aligned with input

# Cumulative operations within groups
df['cumsum'] = df.groupby('category')['value'].cumsum()
df['cummax'] = df.groupby('category')['value'].cummax()
df['rank'] = df.groupby('category')['value'].rank()
```

### Grouping with Bins and Time

```python
# Group by bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100])
df.groupby('age_group')['income'].mean()

# Group by time period
df['date'] = pd.to_datetime(df['date'])
df.set_index('date').groupby(pd.Grouper(freq='M')).sum()  # Monthly
df.set_index('date').groupby(pd.Grouper(freq='W')).sum()  # Weekly
df.set_index('date').groupby(pd.Grouper(freq='Q')).sum()  # Quarterly
```

### Pivot Table (GroupBy + Reshape)

```python
# Pivot table: GroupBy + Reshape in one operation
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum'
)

# Multiple aggregations
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc=['sum', 'mean', 'count']
)

# Multiple values
df.pivot_table(
    values=['sales', 'profit'],
    index='region',
    columns='product',
    aggfunc='sum'
)

# With margins (totals)
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)

# Fill missing values
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    fill_value=0
)
```

---

## Merging, Joining & Concatenation

### Concatenation

```python
# Vertical concatenation (stacking rows)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
pd.concat([df1, df2])  # axis=0 is default

# Reset index after concat
pd.concat([df1, df2], ignore_index=True)

# Horizontal concatenation (side by side)
pd.concat([df1, df2], axis=1)

# Concatenate with keys (multi-level index)
pd.concat([df1, df2], keys=['first', 'second'])

# Only keep matching columns
pd.concat([df1, df2], join='inner')

# Keep all columns, fill with NaN
pd.concat([df1, df2], join='outer')  # Default
```

### Merge (SQL-like joins)

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# Inner join (intersection)
pd.merge(df1, df2, on='key', how='inner')
# Result: B, C

# Left join (keep all from left)
pd.merge(df1, df2, on='key', how='left')
# Result: A, B, C (value2 is NaN for A)

# Right join (keep all from right)
pd.merge(df1, df2, on='key', how='right')
# Result: B, C, D (value1 is NaN for D)

# Outer join (union)
pd.merge(df1, df2, on='key', how='outer')
# Result: A, B, C, D (with NaN where no match)

# Merge on multiple columns
pd.merge(df1, df2, on=['key1', 'key2'])

# Merge with different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Merge on index
pd.merge(df1, df2, left_index=True, right_index=True)

# Suffixes for overlapping columns
pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))

# Validate merge type
pd.merge(df1, df2, on='key', validate='one_to_one')
# Options: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
```

### Join (Index-based merge)

```python
# Join using index
df1.join(df2)  # Left join by default
df1.join(df2, how='inner')
df1.join(df2, how='outer')

# Join on column
df1.join(df2, on='key')

# Join with suffix
df1.join(df2, lsuffix='_left', rsuffix='_right')
```

### Merge Strategies for Large Data

```python
# Indicator column (shows merge source)
result = pd.merge(df1, df2, on='key', how='outer', indicator=True)
# _merge column: 'left_only', 'right_only', 'both'

# Check for merge issues
result[result['_merge'] == 'left_only']  # Rows only in left
result[result['_merge'] == 'right_only']  # Rows only in right

# Merge with validation
try:
    pd.merge(df1, df2, on='key', validate='one_to_one')
except ValueError as e:
    print(f"Merge validation failed: {e}")
```

---

## Reshaping Data

### Pivot and Melt

```python
# Wide to long: melt
df_wide = pd.DataFrame({
    'id': [1, 2, 3],
    '2020': [100, 200, 300],
    '2021': [110, 220, 330],
    '2022': [120, 240, 360]
})

df_long = df_wide.melt(
    id_vars=['id'],
    value_vars=['2020', '2021', '2022'],
    var_name='year',
    value_name='value'
)

# Long to wide: pivot
df_wide_again = df_long.pivot(
    index='id',
    columns='year',
    values='value'
)

# pivot_table (handles duplicates with aggregation)
df_long.pivot_table(
    index='id',
    columns='year',
    values='value',
    aggfunc='mean'
)
```

### Stack and Unstack

```python
# Stack: Column labels → innermost row index
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
}, index=['row1', 'row2'])

stacked = df.stack()
# Multi-index Series:
# row1  A    1
#       B    3
# row2  A    2
#       B    4

# Unstack: Row index → column labels
unstacked = stacked.unstack()  # Back to original

# Unstack specific level
multi_index_df.unstack(level=0)
multi_index_df.unstack(level='level_name')

# Fill NaN after unstack
df.unstack(fill_value=0)
```

### Crosstab

```python
# Frequency table
df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'smoker': ['Y', 'N', 'Y', 'Y', 'N'],
    'age_group': ['Adult', 'Adult', 'Senior', 'Adult', 'Senior']
})

# Simple cross-tabulation
pd.crosstab(df['gender'], df['smoker'])

# With margins
pd.crosstab(df['gender'], df['smoker'], margins=True)

# Normalize (proportions)
pd.crosstab(df['gender'], df['smoker'], normalize='index')  # Row percentages
pd.crosstab(df['gender'], df['smoker'], normalize='columns')  # Column percentages
pd.crosstab(df['gender'], df['smoker'], normalize='all')  # Overall percentages

# With aggregation
pd.crosstab(df['gender'], df['smoker'],
            values=df['age'], aggfunc='mean')

# Multiple row/column variables
pd.crosstab([df['gender'], df['age_group']], df['smoker'])
```

### Transpose

```python
# Swap rows and columns
df.T
df.transpose()
```

### Explode (Unnest lists)

```python
# Expand lists in column to separate rows
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'hobbies': [['reading', 'swimming'], ['gaming', 'cooking', 'travel']]
})

df.explode('hobbies')
# name    hobbies
# Alice   reading
# Alice   swimming
# Bob     gaming
# Bob     cooking
# Bob     travel

# Explode multiple columns (pandas 1.3+)
df.explode(['hobbies', 'other_col'])
```

---

## Time Series

### Datetime Basics

```python
# Create datetime objects
pd.Timestamp('2023-01-01')
pd.Timestamp('2023-01-01 12:30:45')
pd.Timestamp(2023, 1, 1)

# Date range
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')  # Daily
dates = pd.date_range('2023-01-01', periods=100, freq='H')   # 100 hours
dates = pd.date_range('2023-01-01', periods=12, freq='MS')   # Month start

# Frequencies: 'D' (day), 'H' (hour), 'T' or 'min' (minute),
#              'S' (second), 'W' (week), 'M' (month end),
#              'MS' (month start), 'Q' (quarter), 'Y' (year)

# Business day frequencies
pd.date_range('2023-01-01', periods=10, freq='B')   # Business days
pd.date_range('2023-01-01', periods=10, freq='BM')  # Business month end

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Invalid → NaT
```

### DateTime Properties and Methods

```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['dayofyear'] = df['date'].dt.dayofyear
df['week'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Day name and month name
df['day_name'] = df['date'].dt.day_name()  # 'Monday', etc.
df['month_name'] = df['date'].dt.month_name()

# Boolean checks
df['is_month_start'] = df['date'].dt.is_month_start
df['is_month_end'] = df['date'].dt.is_month_end
df['is_quarter_start'] = df['date'].dt.is_quarter_start
df['is_year_start'] = df['date'].dt.is_year_start
df['is_leap_year'] = df['date'].dt.is_leap_year

# Time deltas
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

### Time Series Indexing

```python
# Set datetime as index
df = df.set_index('date')

# Sort by datetime index
df = df.sort_index()

# Select by date
df.loc['2023-01-01']
df.loc['2023-01']  # Entire month
df.loc['2023']     # Entire year
df.loc['2023-01':'2023-06']  # Date range

# Boolean filtering
df[df.index.year == 2023]
df[df.index.month == 1]
df[(df.index >= '2023-01-01') & (df.index < '2023-02-01')]

# Truncate
df.truncate(before='2023-01-01', after='2023-12-31')
```

### Resampling

```python
# Downsampling (high freq → low freq)
df_daily = df.resample('D').sum()     # Daily sum
df_weekly = df.resample('W').mean()   # Weekly mean
df_monthly = df.resample('M').agg({
    'value': 'sum',
    'quantity': 'mean'
})

# Upsampling (low freq → high freq)
df_hourly = df.resample('H').ffill()  # Forward fill
df_hourly = df.resample('H').interpolate()  # Interpolate

# Custom aggregation
df.resample('M').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# OHLC resampling (financial data)
df.resample('D').ohlc()
```

### Rolling Windows

```python
# Simple moving average
df['MA_7'] = df['value'].rolling(window=7).mean()

# Multiple aggregations
df['rolling_stats'] = df['value'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])

# Centered window
df['MA_centered'] = df['value'].rolling(window=7, center=True).mean()

# Minimum periods (requires at least N non-NaN values)
df['MA'] = df['value'].rolling(window=7, min_periods=1).mean()

# Custom function
df['custom'] = df['value'].rolling(window=7).apply(lambda x: x.max() - x.min())

# Exponentially weighted moving average
df['EMA'] = df['value'].ewm(span=7).mean()
df['EMA'] = df['value'].ewm(alpha=0.3).mean()  # Decay factor
```

### Shifting and Lagging

```python
# Shift forward (lag)
df['value_lag1'] = df['value'].shift(1)
df['value_lag7'] = df['value'].shift(7)

# Shift backward (lead)
df['value_lead1'] = df['value'].shift(-1)

# Shift with custom frequency (datetime index)
df['value_prev_month'] = df['value'].shift(1, freq='M')

# Percent change
df['pct_change'] = df['value'].pct_change()
df['pct_change_7d'] = df['value'].pct_change(periods=7)

# Difference
df['diff'] = df['value'].diff()
df['diff_7d'] = df['value'].diff(periods=7)
```

### Time Zones

```python
# Localize (add timezone to naive datetime)
df.index = df.index.tz_localize('UTC')
df.index = df.index.tz_localize('US/Eastern')

# Convert timezone
df.index = df.index.tz_convert('Asia/Tokyo')

# Check timezone
df.index.tz

# Remove timezone
df.index = df.index.tz_localize(None)
```

---

## String Operations

### Basic String Methods

```python
# All string methods accessed via .str
df['text'].str.lower()
df['text'].str.upper()
df['text'].str.title()
df['text'].str.capitalize()
df['text'].str.swapcase()

# Strip whitespace
df['text'].str.strip()
df['text'].str.lstrip()
df['text'].str.rstrip()

# Length
df['text'].str.len()

# Concatenation
df['full_name'] = df['first_name'].str.cat(df['last_name'], sep=' ')

# Repeat
df['repeated'] = df['text'].str.repeat(3)
```

### String Searching and Matching

```python
# Contains
df[df['text'].str.contains('keyword')]
df[df['text'].str.contains('keyword', case=False)]  # Case-insensitive
df[df['text'].str.contains('key|word', regex=True)]  # OR pattern

# Startswith / Endswith
df[df['text'].str.startswith('prefix')]
df[df['text'].str.endswith('suffix')]

# Match (requires pattern at start)
df['text'].str.match(r'^\d+')  # Starts with digits

# Find position
df['text'].str.find('substring')  # Returns index or -1
df['text'].str.rfind('substring')  # Find from right

# Count occurrences
df['text'].str.count('pattern')
```

### String Replacement

```python
# Simple replacement
df['text'].str.replace('old', 'new')
df['text'].str.replace('old', 'new', case=False)

# Regex replacement
df['text'].str.replace(r'\d+', '', regex=True)  # Remove all digits
df['text'].str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace

# Multiple replacements
df['text'].str.replace({'old1': 'new1', 'old2': 'new2'})
```

### String Extraction

```python
# Extract with regex groups
df['text'].str.extract(r'(\d+)')  # First group
df[['area', 'phone']] = df['phone'].str.extract(r'(\d{3})-(\d{7})')

# Extract all occurrences
df['text'].str.extractall(r'(\d+)')

# Split
df['text'].str.split()  # Returns list
df['text'].str.split(expand=True)  # Returns DataFrame
df[['first', 'last']] = df['name'].str.split(' ', n=1, expand=True)

# Split from right
df['text'].str.rsplit(n=1, expand=True)

# Partition (split into 3 parts)
df['email'].str.partition('@')  # [before, separator, after]
```

### String Transformation

```python
# Pad
df['text'].str.pad(width=10, side='left', fillchar='0')  # Zero-padding
df['text'].str.pad(width=10, side='right')
df['text'].str.pad(width=10, side='both')

# Center
df['text'].str.center(width=20, fillchar='*')

# Slice
df['text'].str.slice(start=0, stop=5)
df['text'].str[:5]  # First 5 characters

# Get specific character
df['text'].str.get(0)  # First character
df['text'].str[0]     # Alternative

# Wrap text
df['text'].str.wrap(width=40)

# Normalize (Unicode)
df['text'].str.normalize('NFC')
```

### Advanced String Operations

```python
# Regular expression methods
df['text'].str.findall(r'\b\w+\b')  # All words
df['text'].str.replace(r'(\w+) (\w+)', r'\2 \1', regex=True)  # Swap words

# String contains any/all
keywords = ['python', 'pandas', 'numpy']
df['has_keyword'] = df['text'].str.contains('|'.join(keywords), case=False)

# Remove special characters
df['clean'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Decode/Encode
df['text'].str.encode('utf-8')
df['bytes'].str.decode('utf-8')

# String formatting
df['formatted'] = df['value'].apply(lambda x: f'{x:.2f}')
```

---

## Categorical Data

### Creating Categorical

```python
# Convert to categorical
df['category'] = df['category'].astype('category')
df['category'] = pd.Categorical(df['category'])

# Create with specific categories and order
df['size'] = pd.Categorical(
    df['size'],
    categories=['small', 'medium', 'large'],
    ordered=True
)

# Check if categorical
df['category'].dtype == 'category'
```

### Categorical Properties

```python
# Get categories
df['category'].cat.categories

# Get codes (integer representation)
df['category'].cat.codes

# Check if ordered
df['category'].cat.ordered
```

### Categorical Operations

```python
# Add categories
df['category'] = df['category'].cat.add_categories(['new_cat'])

# Remove categories
df['category'] = df['category'].cat.remove_categories(['old_cat'])

# Rename categories
df['category'] = df['category'].cat.rename_categories({
    'old': 'new'
})

# Reorder categories
df['size'] = df['size'].cat.reorder_categories(
    ['small', 'medium', 'large', 'xl'],
    ordered=True
)

# Set ordered
df['category'] = df['category'].cat.as_ordered()
df['category'] = df['category'].cat.as_unordered()

# Remove unused categories
df['category'] = df['category'].cat.remove_unused_categories()
```

### Benefits of Categorical

```python
# Memory savings
df['category'] = df['category'].astype('category')
# Can reduce memory usage by 50-90% for low-cardinality columns

# Faster groupby
df.groupby('category').sum()  # Faster with categorical

# Ordered comparisons
df['size'] = pd.Categorical(
    df['size'],
    categories=['S', 'M', 'L', 'XL'],
    ordered=True
)
df[df['size'] > 'M']  # Returns L and XL

# Preserved category order in plots
df['size'].value_counts().plot(kind='bar')  # Bars in category order
```

---

## Window Functions

### Rolling Windows

```python
# Simple rolling
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_sum'] = df['value'].rolling(window=7).sum()
df['rolling_std'] = df['value'].rolling(window=7).std()

# Multiple aggregations
df[['mean', 'std', 'min', 'max']] = df['value'].rolling(window=7).agg(
    ['mean', 'std', 'min', 'max']
)

# Custom function
df['range'] = df['value'].rolling(window=7).apply(
    lambda x: x.max() - x.min()
)

# With min_periods
df['rolling_mean'] = df['value'].rolling(window=7, min_periods=1).mean()

# Centered window
df['centered_ma'] = df['value'].rolling(window=7, center=True).mean()
```

### Expanding Windows

```python
# Cumulative operations
df['expanding_mean'] = df['value'].expanding().mean()
df['expanding_sum'] = df['value'].expanding().sum()
df['expanding_max'] = df['value'].expanding().max()

# Same as cumsum, cummax, etc.
df['cumsum'] = df['value'].cumsum()
df['cummax'] = df['value'].cummax()
df['cummin'] = df['value'].cummin()
df['cumprod'] = df['value'].cumprod()
```

### Exponentially Weighted Windows

```python
# EWMA - Exponentially Weighted Moving Average
df['ewma'] = df['value'].ewm(span=7).mean()
df['ewma'] = df['value'].ewm(alpha=0.3).mean()
df['ewma'] = df['value'].ewm(halflife=7).mean()

# Other EW functions
df['ewm_std'] = df['value'].ewm(span=7).std()
df['ewm_var'] = df['value'].ewm(span=7).var()

# Adjust parameter
df['ewma'] = df['value'].ewm(span=7, adjust=False).mean()
```

### Window Functions for Finance

```python
# Bollinger Bands
window = 20
df['MA'] = df['close'].rolling(window).mean()
df['std'] = df['close'].rolling(window).std()
df['upper_band'] = df['MA'] + 2 * df['std']
df['lower_band'] = df['MA'] - 2 * df['std']

# RSI (Relative Strength Index)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
df['ema12'] = df['close'].ewm(span=12).mean()
df['ema26'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['ema12'] - df['ema26']
df['signal'] = df['MACD'].ewm(span=9).mean()
```

---

## Performance Optimization

### Memory Optimization

```python
# Check memory usage
df.info(memory_usage='deep')
df.memory_usage(deep=True)

# Optimize dtypes
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Low cardinality
            df[col] = df[col].astype('category')

    return df

df = optimize_dtypes(df)

# Use specific dtypes when reading
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',
    'value': 'float32',
    'category': 'category'
})

# Chunking for large files
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed_chunk = process(chunk)
    chunks.append(processed_chunk)

df = pd.concat(chunks, ignore_index=True)
```

### Vectorization

```python
# BAD: Iterating rows
result = []
for idx, row in df.iterrows():  # SLOW!
    result.append(row['A'] + row['B'])
df['sum'] = result

# GOOD: Vectorized operation
df['sum'] = df['A'] + df['B']  # FAST!

# BAD: Apply with simple operation
df['squared'] = df['value'].apply(lambda x: x ** 2)

# GOOD: Direct operation
df['squared'] = df['value'] ** 2

# When apply is needed
df['complex'] = df.apply(lambda row: complex_function(row['A'], row['B']), axis=1)

# Try vectorization with numpy
df['result'] = np.where(df['value'] > 0, df['value'], 0)  # Vectorized ReLU
```

### Efficient Operations

```python
# Use query for complex filtering
df.query('A > 10 and B < 20')  # Can be faster than boolean indexing

# Use eval for expressions
df.eval('C = A + B')  # More memory efficient

# Avoid chained indexing
# BAD
df[df['A'] > 0]['B'] = 999  # SettingWithCopyWarning

# GOOD
df.loc[df['A'] > 0, 'B'] = 999

# Use categories for low-cardinality strings
df['category'] = df['category'].astype('category')

# Avoid loops, use groupby + transform
# BAD
for group in df['category'].unique():
    mask = df['category'] == group
    df.loc[mask, 'normalized'] = (
        df.loc[mask, 'value'] - df.loc[mask, 'value'].mean()
    )

# GOOD
df['normalized'] = df.groupby('category')['value'].transform(
    lambda x: x - x.mean()
)
```

### Index Optimization

```python
# Set index for frequent lookups
df = df.set_index('id')  # O(1) lookups

# Sort index for range queries
df = df.sort_index()

# Multi-index for hierarchical data
df = df.set_index(['category', 'subcategory'])

# Reset when not needed
df = df.reset_index(drop=True)
```

### Parallel Processing

```python
# Use swifter for automatic parallelization
# pip install swifter
import swifter
df['result'] = df['value'].swifter.apply(complex_function)

# Manual multiprocessing
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    return chunk.apply(complex_function)

# Split dataframe
chunks = np.array_split(df, 4)  # 4 chunks

# Process in parallel
with Pool(4) as pool:
    results = pool.map(process_chunk, chunks)

df = pd.concat(results)

# Dask for out-of-core computation
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=4)
result = ddf.groupby('category').value.mean().compute()
```

---

## ML/Data Science Patterns

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Basic split
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Time series split (no shuffle)
split_point = int(len(df) * 0.8)
train = df[:split_point]
test = df[split_point:]
```

### Feature Engineering

```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Or with scikit-learn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['category']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# Label encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['young', 'adult', 'middle', 'senior'])

# Interaction features
df['interaction'] = df['feature1'] * df['feature2']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())

# Log transform (for skewed data)
df['log_value'] = np.log1p(df['value'])  # log(1 + x)

# Date features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (z-score normalization)
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Min-Max scaling (0-1 range)
scaler = MinMaxScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Robust scaling (less sensitive to outliers)
scaler = RobustScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Manual standardization
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()

# Group-wise scaling
df['scaled'] = df.groupby('category')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### Handling Imbalanced Data

```python
# Check class distribution
df['target'].value_counts()
df['target'].value_counts(normalize=True)

# Undersampling majority class
from sklearn.utils import resample

# Separate classes
df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]

# Downsample majority
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Oversampling minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# SMOTE (Synthetic Minority Over-sampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Cross-Validation Folds

```python
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(df):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    # Train and evaluate

# Stratified K-Fold (for classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(df):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
```

### Feature Selection

```python
# Correlation with target
correlations = df.corr()['target'].sort_values(ascending=False)
top_features = correlations[1:11].index.tolist()  # Top 10

# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

# Univariate feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

# Feature importance from tree models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## Integration Patterns

### With NumPy

```python
# DataFrame to NumPy array
array = df.values
array = df.to_numpy()  # Preferred

# Specific columns
array = df[['A', 'B']].values

# NumPy array to DataFrame
df = pd.DataFrame(array, columns=['A', 'B', 'C'])

# Apply NumPy functions
df['result'] = np.sqrt(df['value'])
df['log'] = np.log1p(df['value'])

# NumPy operations on DataFrames
df_normalized = (df - df.mean()) / df.std()
```

### With Scikit-learn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
X_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
df['predictions'] = predictions

# Probabilities
probabilities = model.predict_proba(X_test)
df_probs = pd.DataFrame(probabilities, columns=model.classes_)
```

### With Matplotlib/Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas built-in plotting
df['value'].plot(kind='hist', bins=30)
df.plot(x='date', y='value', kind='line')
df.plot(kind='scatter', x='A', y='B', c='C', colormap='viridis')

# Bar plot
df.groupby('category')['value'].sum().plot(kind='bar')

# Box plot
df.boxplot(column='value', by='category')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Pairplot
sns.pairplot(df, hue='category')

# Distribution plot
sns.histplot(data=df, x='value', hue='category', kde=True)

# Time series
df.set_index('date')['value'].plot(figsize=(12, 6))
```

### With SQL

```python
import sqlite3
from sqlalchemy import create_engine

# SQLite
conn = sqlite3.connect('database.db')

# Read
df = pd.read_sql('SELECT * FROM table', conn)
df = pd.read_sql_query('SELECT * FROM table WHERE id > 100', conn)

# Write
df.to_sql('table_name', conn, if_exists='replace', index=False)

# SQLAlchemy (more databases)
engine = create_engine('postgresql://user:pass@localhost/dbname')
df = pd.read_sql('SELECT * FROM table', engine)
df.to_sql('table_name', engine, if_exists='append', index=False)

# Chunked reading for large tables
for chunk in pd.read_sql('SELECT * FROM large_table', conn, chunksize=10000):
    process(chunk)
```

### With Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('example').getOrCreate()

# Pandas to Spark
spark_df = spark.createDataFrame(df)

# Spark to Pandas
df = spark_df.toPandas()

# Distributed operations
spark_df.groupBy('category').agg({'value': 'sum'}).show()
```

---

## Best Practices

### Code Style

```python
# Use meaningful variable names
user_data = pd.read_csv('users.csv')  # Good
df = pd.read_csv('users.csv')        # Acceptable for exploratory

# Method chaining (when appropriate)
result = (df
    .query('age > 18')
    .groupby('category')
    ['value'].sum()
    .sort_values(ascending=False)
    .head(10)
)

# Avoid chained assignment
# BAD
df[df['A'] > 0]['B'] = 999  # SettingWithCopyWarning

# GOOD
df.loc[df['A'] > 0, 'B'] = 999

# Use .copy() when needed
df_subset = df[df['A'] > 0].copy()
df_subset['B'] = 999  # Safe
```

### Error Handling

```python
# Check for missing values before operations
if df['column'].isna().any():
    df['column'].fillna(0, inplace=True)

# Validate data types
assert df['age'].dtype == 'int64', "Age must be integer"

# Handle divisions by zero
df['ratio'] = df['A'] / df['B'].replace(0, np.nan)

# Try-except for robust code
try:
    df['date'] = pd.to_datetime(df['date'])
except ValueError:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
```

### Performance Tips

```python
# 1. Use vectorized operations
df['result'] = df['A'] + df['B']  # Fast

# 2. Avoid iterrows()
# for idx, row in df.iterrows():  # SLOW!

# 3. Use categorical for low-cardinality strings
df['category'] = df['category'].astype('category')

# 4. Read only needed columns
df = pd.read_csv('file.csv', usecols=['A', 'B', 'C'])

# 5. Use appropriate dtypes
df = pd.read_csv('file.csv', dtype={'id': 'int32', 'value': 'float32'})

# 6. Filter early, aggregate late
result = df[df['year'] == 2023].groupby('category').sum()  # Good

# 7. Use query() for complex conditions
df.query('A > 10 and B < 20')

# 8. Profile your code
import pandas as pd
pd.set_option('mode.chained_assignment', 'warn')  # Catch issues
```

### Data Validation

```python
# Schema validation
expected_columns = ['id', 'name', 'age', 'email']
assert set(expected_columns).issubset(df.columns), "Missing columns"

# Type validation
assert df['age'].dtype in ['int64', 'int32'], "Age must be integer"

# Range validation
assert df['age'].between(0, 120).all(), "Invalid age values"

# No duplicates
assert not df['id'].duplicated().any(), "Duplicate IDs found"

# No missing values in required columns
required = ['id', 'name']
assert df[required].notna().all().all(), "Missing required values"

# Using pandera (schema validation library)
# import pandera as pa
# schema = pa.DataFrameSchema({
#     'id': pa.Column(int, pa.Check.greater_than(0)),
#     'age': pa.Column(int, pa.Check.in_range(0, 120)),
#     'name': pa.Column(str, nullable=False)
# })
# validated_df = schema.validate(df)
```

### Documentation

```python
def process_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw sales data.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw sales data with columns: date, product, quantity, price

    Returns:
    --------
    pd.DataFrame
        Processed data with additional columns: revenue, month, year

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'date': ['2023-01-01', '2023-01-02'],
    ...     'product': ['A', 'B'],
    ...     'quantity': [10, 20],
    ...     'price': [100, 200]
    ... })
    >>> processed = process_sales_data(df)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['quantity'] * df['price']
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df
```

---

## Summary

Pandas is the cornerstone of data analysis in Python. Key takeaways:

1. **Master the fundamentals**: Series, DataFrame, indexing (loc/iloc)
2. **Think vectorized**: Avoid loops, use pandas operations
3. **Leverage groupby**: Most data analysis involves split-apply-combine
4. **Clean your data**: Handle missing values, duplicates, and types properly
5. **Optimize for performance**: Use appropriate dtypes, categories, and chunking
6. **Chain operations**: Build readable data pipelines
7. **Integrate well**: Pandas works seamlessly with NumPy, scikit-learn, matplotlib
8. **Validate your data**: Check assumptions, handle errors gracefully

**Common Patterns to Remember:**
- Filter: `df[df['col'] > value]` or `df.query('col > value')`
- Group & Aggregate: `df.groupby('col').agg({'val': 'sum'})`
- Join: `pd.merge(df1, df2, on='key', how='inner')`
- Reshape: `df.pivot()`, `df.melt()`, `df.stack()`
- Time Series: `df.resample('D').sum()`, `df.rolling(7).mean()`

**Resources:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)
- [Effective Pandas](https://github.com/TomAugspurger/effective-pandas)
