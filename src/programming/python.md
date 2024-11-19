# Python

## Basic Syntax Examples 

### Print
```python
print("Hello, World!")
```

### Lists
```python
my_list = [1, 2, 3, 4, 5]
print(my_list)
print(len(my_list))

my_list.append(6)
print(my_list)
```
    - Lists are mutable
    - Lists are ordered
    - Lists can contain duplicates
    - Lists can be nested
    - Lists are dynamic

### Tuples
```python
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple)
```



Here's a practical approach using pandas (which is great for handling Excel files) and dataclasses for clean class definition.

```python
from dataclasses import dataclass
import pandas as pd
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str

def load_people_from_excel(file_path: str) -> List[Person]:
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Convert DataFrame rows to Person objects
    people = [
        Person(
            name=row['name'],
            age=row['age'],
            email=row['email']
        ) for _, row in df.iterrows()
    ]
    
    return people

# Usage example
if __name__ == "__main__":
    # Assuming your Excel file has columns: name, age, email
    people = load_people_from_excel("your_file.xlsx")
    
    # Access the data
    for person in people:
        print(f"{person.name} is {person.age} years old")
```        

This solution offers several benefits:
Type Safety: Using @dataclass and type hints makes the code more maintainable and helps catch errors early
Clean Mapping: The mapping between Excel columns and class members is explicit and easy to understand
Easy to Extend: You can easily add more fields to the class and corresponding Excel columns
Alternative approach if you need more control over the mapping:

```python
class Person:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
    
    @classmethod
    def from_excel_row(cls, row):
        # Custom mapping logic here
        return cls(
            name=row['name'].strip(),  # Add data cleaning
            age=int(row['age']),       # Type conversion
            email=row['email'].lower() # Data normalization
        )

def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    return [Person.from_excel_row(row) for _, row in df.iterrows()]
```    
    
If your Excel columns have different names than your class members, you can add a mapping dictionary:

```python
EXCEL_TO_CLASS_MAPPING = {
    'Full Name': 'name',
    'Person Age': 'age',
    'E-mail Address': 'email'
}

def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    # Rename columns according to mapping
    df = df.rename(columns=EXCEL_TO_CLASS_MAPPING)
    return [Person.from_excel_row(row) for _, row in df.iterrows()]
```    
    
Here are a few more patterns for mapping Excel data to class members:
Factory Pattern with Schema Validation:
```python
from pydantic import BaseModel
from typing import Dict, Any

class PersonSchema(BaseModel):
    name: str
    age: int
    email: str
    
    class Config:
        extra = "forbid"  # Prevents additional fields

class PersonFactory:
    @staticmethod
    def create_from_row(row: Dict[str, Any]) -> 'Person':
        # Validate data using schema
        validated_data = PersonSchema(**row).dict()
        return Person(**validated_data)

def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    return [PersonFactory.create_from_row(row) for _, row in df.iterrows()]
    
```

Builder Pattern (useful for complex object construction):

```python
class PersonBuilder:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._person = Person()
        return self
    
    def set_name(self, name: str):
        self._person.name = name.strip()
        return self
    
    def set_age(self, age: int):
        self._person.age = int(age)
        return self
    
    def set_email(self, email: str):
        self._person.email = email.lower()
        return self
    
    def build(self) -> Person:
        person = self._person
        self.reset()
        return person

def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    builder = PersonBuilder()
    
    return [
        builder
            .set_name(row['name'])
            .set_age(row['age'])
            .set_email(row['email'])
            .build()
        for _, row in df.iterrows()
    ]
```

Serializer Pattern (similar to Django/REST framework approach):

```python   
class PersonSerializer:
    @staticmethod
    def validate_age(age: int) -> int:
        if not isinstance(age, (int, float)) or age < 0:
            raise ValueError("Age must be a positive number")
        return int(age)
    
    @staticmethod
    def validate_email(email: str) -> str:
        email = email.lower().strip()
        if '@' not in email:
            raise ValueError("Invalid email format")
        return email
    
    @classmethod
    def from_excel(cls, data: Dict[str, Any]) -> Person:
        return Person(
            name=data['name'].strip(),
            age=cls.validate_age(data['age']),
            email=cls.validate_email(data['email'])
        )

def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    return [PersonSerializer.from_excel(row) for _, row in df.iterrows()]    
``` 
    
Descriptor Pattern (for automatic data validation):

```python
class ValidatedField:
    def __init__(self, field_type, validator=None):
        self.field_type = field_type
        self.validator = validator
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        value = self.field_type(value)
        if self.validator:
            value = self.validator(value)
        instance.__dict__[self.name] = value

class Person:
    name = ValidatedField(str, lambda x: x.strip())
    age = ValidatedField(int, lambda x: x if x >= 0 else ValueError("Age must be positive"))
    email = ValidatedField(str, lambda x: x.lower())

    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
```

Strategy Pattern (for different data sources):

```python
from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, source: str) -> List[Person]:
        pass

class ExcelDataLoader(DataLoader):
    def load_data(self, source: str) -> List[Person]:
        df = pd.read_excel(source)
        return [Person(**row) for _, row in df.iterrows()]

class CSVDataLoader(DataLoader):
    def load_data(self, source: str) -> List[Person]:
        df = pd.read_csv(source)
        return [Person(**row) for _, row in df.iterrows()]

# Usage
loader = ExcelDataLoader()
people = loader.load_data("data.xlsx")
```

Each pattern has its use cases:
Use Factory when you need centralized object creation with validation
Use Builder for complex objects with many optional parameters
Use Serializer when you need strong data validation and transformation
Use Descriptor for automatic attribute-level validation
Use Strategy when you need to support multiple data sources
