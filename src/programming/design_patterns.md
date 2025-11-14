# Design Patterns

 // Start Generation Here

## Introduction to Design Patterns

Design patterns are proven solutions to common problems encountered in software development. They provide a standardized approach to solving issues related to object creation, structure, and behavior, promoting code reusability, scalability, and maintainability. Understanding design patterns is essential for building robust and efficient software systems.

### Categories of Design Patterns

Design patterns are typically categorized into three main types:

1. **Creational Patterns**: Focus on object creation mechanisms, aiming to create objects in a manner suitable to the situation.
2. **Structural Patterns**: Deal with object composition, identifying simple ways to realize relationships between objects.
3. **Behavioral Patterns**: Concerned with communication between objects, highlighting patterns of interaction.


## List of Design Patterns and Their Uses

### Creational Patterns

1. **Singleton**: Ensures a class has only one instance and provides a global point of access to it. Useful for managing shared resources like logging or configuration settings.

2. **Factory Method**: Defines an interface for creating an object but lets subclasses alter the type of objects that will be created. Useful for creating objects without specifying the exact class of the object to be created.

3. **Abstract Factory**: Provides an interface for creating families of related or dependent objects without specifying their concrete classes. Useful when the system needs to be independent of how its objects are created.

4. **Builder**: Separates the construction of a complex object from its representation, allowing the same construction process to create various representations. Useful for constructing complex objects step by step.

5. **Prototype**: Specifies the kinds of objects to create using a prototypical instance and creates new objects by copying this prototype. Useful when object creation is expensive or complex.

### Structural Patterns

1. **Adapter**: Allows incompatible interfaces to work together by converting the interface of one class into another expected by the clients. Useful when integrating legacy systems or third-party libraries.

2. **Bridge**: Decouples an abstraction from its implementation so that the two can vary independently. Useful for handling multiple implementations of an abstraction.

3. **Composite**: Composes objects into tree structures to represent part-whole hierarchies, allowing clients to treat individual objects and compositions uniformly. Useful for representing hierarchical structures like file systems.

4. **Decorator**: Adds additional responsibilities to an object dynamically without altering its structure. Useful for enhancing functionalities of objects without subclassing.

5. **Facade**: Provides a simplified interface to a complex subsystem, making the subsystem easier to use. Useful for reducing dependencies and simplifying client interaction.

6. **Flyweight**: Reduces the cost of creating and maintaining a large number of similar objects by sharing as much data as possible. Useful for handling large numbers of objects efficiently.

7. **Proxy**: Provides a surrogate or placeholder for another object to control access to it. Useful for lazy initialization, access control, or logging.

### Behavioral Patterns

1. **Chain of Responsibility**: Passes a request along a chain of handlers, allowing each handler to process or pass it along. Useful for decoupling senders and receivers of requests.

2. **Command**: Encapsulates a request as an object, thereby allowing for parameterization and queuing of requests. Useful for implementing undoable operations or task scheduling.

3. **Interpreter**: Defines a representation for a language's grammar and interprets sentences in the language. Useful for parsing and interpreting expressions or languages.

4. **Iterator**: Provides a way to access elements of an aggregate object sequentially without exposing its underlying representation. Useful for traversing collections.

5. **Mediator**: Defines an object that encapsulates how a set of objects interact, promoting loose coupling by keeping objects from referring to each other explicitly. Useful for reducing complexity in object interactions.

6. **Memento**: Captures and externalizes an object's internal state without violating encapsulation, allowing the object to be restored to this state later. Useful for implementing undo functionality.

7. **Observer**: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. Useful for event handling and implementing distributed event systems.

8. **State**: Allows an object to alter its behavior when its internal state changes, appearing as if it has changed its class. Useful for managing state-dependent behavior.

9. **Strategy**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable, allowing the algorithm to vary independently from clients that use it. Useful for selecting algorithms dynamically at runtime.

10. **Template Method**: Defines the skeleton of an algorithm in a method, deferring some steps to subclasses, allowing subclasses to redefine certain steps without changing the algorithm's structure. Useful for implementing invariant parts of an algorithm and varying certain steps.

11. **Visitor**: Represents an operation to be performed on elements of an object structure, allowing new operations to be added without modifying the classes of the elements on which it operates. Useful for separating algorithms from object structures.


### Creational Patterns

#### Singleton Pattern

**Intent**: Ensure a class has only one instance and provide a global point of access to it.

**Problem**: Sometimes you need exactly one instance of a class (e.g., database connection pool, thread pool, cache, configuration manager). Creating multiple instances wastes resources and can cause inconsistent state.

**Solution**: Make the class responsible for keeping track of its sole instance. The class can ensure that no other instance can be created (by intercepting requests to create new objects) and provide a way to access the instance.

**When to Use**:
- Exactly one instance of a class is required
- Controlled access to the sole instance is needed
- The instance should be extensible by subclassing

**Real-World Examples**:
- Database connection managers
- Logger systems
- Configuration managers
- Thread pools
- Cache systems
- Device drivers

**Implementation in C++ (Thread-Safe)**:

```cpp
#include <iostream>
#include <mutex>
#include <memory>

class DatabaseConnection {
public:
    // Get the singleton instance
    static DatabaseConnection& getInstance() {
        // C++11 guarantees thread-safe initialization of static local variables
        static DatabaseConnection instance;
        return instance;
    }

    // Delete copy constructor and assignment operator
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;

    void connect(const std::string& connectionString) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!connected_) {
            std::cout << "Connecting to database: " << connectionString << std::endl;
            connected_ = true;
        }
    }

    void query(const std::string& sql) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (connected_) {
            std::cout << "Executing: " << sql << std::endl;
        } else {
            std::cout << "Not connected!" << std::endl;
        }
    }

private:
    DatabaseConnection() : connected_(false) {
        std::cout << "DatabaseConnection instance created" << std::endl;
    }

    ~DatabaseConnection() {
        std::cout << "DatabaseConnection instance destroyed" << std::endl;
    }

    bool connected_;
    std::mutex mutex_;
};

// Usage
int main() {
    // All these calls return the same instance
    DatabaseConnection::getInstance().connect("server=localhost;db=mydb");
    DatabaseConnection::getInstance().query("SELECT * FROM users");

    DatabaseConnection& db1 = DatabaseConnection::getInstance();
    DatabaseConnection& db2 = DatabaseConnection::getInstance();

    std::cout << "Same instance? " << (&db1 == &db2 ? "Yes" : "No") << std::endl;

    return 0;
}
```

**Implementation in Python**:

```python
import threading

class DatabaseConnection:
    """Thread-safe singleton using double-checked locking"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.connected = False
            self._initialized = True
            print("DatabaseConnection instance created")

    def connect(self, connection_string):
        if not self.connected:
            print(f"Connecting to database: {connection_string}")
            self.connected = True

    def query(self, sql):
        if self.connected:
            print(f"Executing: {sql}")
        else:
            print("Not connected!")

# Python decorator approach (cleaner)
def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Logger:
    def __init__(self):
        self.log_file = "app.log"
        print("Logger initialized")

    def log(self, message):
        print(f"[LOG] {message}")

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(f"Same instance? {db1 is db2}")  # True

logger1 = Logger()
logger2 = Logger()
print(f"Same logger? {logger1 is logger2}")  # True
```

**Advantages**:
- Controlled access to sole instance
- Reduced memory footprint
- Permits refinement of operations and representation
- Lazy initialization possible

**Disadvantages**:
- Can be difficult to test (global state)
- Violates Single Responsibility Principle
- Can mask bad design (tight coupling)
- Requires special treatment in multi-threaded environments

---

#### Factory Method Pattern

**Intent**: Define an interface for creating an object, but let subclasses decide which class to instantiate.

**Problem**: A framework needs to standardize the architectural model for a range of applications, but allow for individual applications to define their own domain objects and provide for their instantiation.

**Solution**: Define a factory method that returns objects of a common interface. Subclasses implement the factory method to create specific product types.

**When to Use**:
- A class can't anticipate the class of objects it must create
- A class wants its subclasses to specify the objects it creates
- Classes delegate responsibility to one of several helper subclasses

**Real-World Examples**:
- GUI frameworks creating platform-specific buttons/windows
- Document editors creating different document types
- Logistics apps creating different transport types
- Database connectors for different DBMS systems

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>

// Product interface
class Transport {
public:
    virtual ~Transport() = default;
    virtual void deliver() = 0;
    virtual std::string getType() const = 0;
};

// Concrete Products
class Truck : public Transport {
public:
    void deliver() override {
        std::cout << "Delivering by land in a box" << std::endl;
    }

    std::string getType() const override {
        return "Truck";
    }
};

class Ship : public Transport {
public:
    void deliver() override {
        std::cout << "Delivering by sea in a container" << std::endl;
    }

    std::string getType() const override {
        return "Ship";
    }
};

class Airplane : public Transport {
public:
    void deliver() override {
        std::cout << "Delivering by air in a cargo hold" << std::endl;
    }

    std::string getType() const override {
        return "Airplane";
    }
};

// Creator (Factory)
class Logistics {
public:
    virtual ~Logistics() = default;

    // Factory method
    virtual std::unique_ptr<Transport> createTransport() = 0;

    void planDelivery() {
        auto transport = createTransport();
        std::cout << "Planning delivery using " << transport->getType() << std::endl;
        transport->deliver();
    }
};

// Concrete Creators
class RoadLogistics : public Logistics {
public:
    std::unique_ptr<Transport> createTransport() override {
        return std::make_unique<Truck>();
    }
};

class SeaLogistics : public Logistics {
public:
    std::unique_ptr<Transport> createTransport() override {
        return std::make_unique<Ship>();
    }
};

class AirLogistics : public Logistics {
public:
    std::unique_ptr<Transport> createTransport() override {
        return std::make_unique<Airplane>();
    }
};

// Usage
int main() {
    std::unique_ptr<Logistics> logistics;

    logistics = std::make_unique<RoadLogistics>();
    logistics->planDelivery();

    logistics = std::make_unique<SeaLogistics>();
    logistics->planDelivery();

    logistics = std::make_unique<AirLogistics>();
    logistics->planDelivery();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import Protocol

# Product interface
class Transport(ABC):
    @abstractmethod
    def deliver(self) -> None:
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass

# Concrete Products
class Truck(Transport):
    def deliver(self) -> None:
        print("Delivering by land in a box")

    def get_type(self) -> str:
        return "Truck"

class Ship(Transport):
    def deliver(self) -> None:
        print("Delivering by sea in a container")

    def get_type(self) -> str:
        return "Ship"

class Airplane(Transport):
    def deliver(self) -> None:
        print("Delivering by air in a cargo hold")

    def get_type(self) -> str:
        return "Airplane"

# Creator (Factory)
class Logistics(ABC):
    @abstractmethod
    def create_transport(self) -> Transport:
        """Factory method"""
        pass

    def plan_delivery(self) -> None:
        transport = self.create_transport()
        print(f"Planning delivery using {transport.get_type()}")
        transport.deliver()

# Concrete Creators
class RoadLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Truck()

class SeaLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Ship()

class AirLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Airplane()

# Usage
if __name__ == "__main__":
    logistics = RoadLogistics()
    logistics.plan_delivery()

    logistics = SeaLogistics()
    logistics.plan_delivery()

    logistics = AirLogistics()
    logistics.plan_delivery()
```

**Advantages**:
- Avoids tight coupling between creator and concrete products
- Single Responsibility Principle: product creation code in one place
- Open/Closed Principle: introduce new product types without breaking existing code

**Disadvantages**:
- Code can become more complicated with many new subclasses
- Requires subclassing just to create objects

---

#### Abstract Factory Pattern

**Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

**Problem**: You need to create families of related objects that must be used together, and you want to ensure compatibility between these objects.

**Solution**: Declare interfaces for creating each distinct product. Then create concrete factory classes that implement these interfaces for each product variant.

**When to Use**:
- System should be independent of how its products are created
- System should be configured with one of multiple families of products
- Family of related product objects must be used together
- You want to provide a class library of products without revealing implementations

**Real-World Examples**:
- GUI toolkits with different themes (Windows, Mac, Linux)
- Cross-platform UI libraries
- Database access libraries for different DBMS
- Document converters for different formats

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>

// Abstract Products
class Button {
public:
    virtual ~Button() = default;
    virtual void paint() = 0;
    virtual std::string getStyle() const = 0;
};

class Checkbox {
public:
    virtual ~Checkbox() = default;
    virtual void paint() = 0;
    virtual std::string getStyle() const = 0;
};

class TextField {
public:
    virtual ~TextField() = default;
    virtual void paint() = 0;
    virtual std::string getStyle() const = 0;
};

// Windows Products
class WindowsButton : public Button {
public:
    void paint() override {
        std::cout << "Rendering Windows-style button" << std::endl;
    }

    std::string getStyle() const override {
        return "Windows";
    }
};

class WindowsCheckbox : public Checkbox {
public:
    void paint() override {
        std::cout << "Rendering Windows-style checkbox" << std::endl;
    }

    std::string getStyle() const override {
        return "Windows";
    }
};

class WindowsTextField : public TextField {
public:
    void paint() override {
        std::cout << "Rendering Windows-style text field" << std::endl;
    }

    std::string getStyle() const override {
        return "Windows";
    }
};

// Mac Products
class MacButton : public Button {
public:
    void paint() override {
        std::cout << "Rendering Mac-style button" << std::endl;
    }

    std::string getStyle() const override {
        return "Mac";
    }
};

class MacCheckbox : public Checkbox {
public:
    void paint() override {
        std::cout << "Rendering Mac-style checkbox" << std::endl;
    }

    std::string getStyle() const override {
        return "Mac";
    }
};

class MacTextField : public TextField {
public:
    void paint() override {
        std::cout << "Rendering Mac-style text field" << std::endl;
    }

    std::string getStyle() const override {
        return "Mac";
    }
};

// Abstract Factory
class GUIFactory {
public:
    virtual ~GUIFactory() = default;
    virtual std::unique_ptr<Button> createButton() = 0;
    virtual std::unique_ptr<Checkbox> createCheckbox() = 0;
    virtual std::unique_ptr<TextField> createTextField() = 0;
};

// Concrete Factories
class WindowsFactory : public GUIFactory {
public:
    std::unique_ptr<Button> createButton() override {
        return std::make_unique<WindowsButton>();
    }

    std::unique_ptr<Checkbox> createCheckbox() override {
        return std::make_unique<WindowsCheckbox>();
    }

    std::unique_ptr<TextField> createTextField() override {
        return std::make_unique<WindowsTextField>();
    }
};

class MacFactory : public GUIFactory {
public:
    std::unique_ptr<Button> createButton() override {
        return std::make_unique<MacButton>();
    }

    std::unique_ptr<Checkbox> createCheckbox() override {
        return std::make_unique<MacCheckbox>();
    }

    std::unique_ptr<TextField> createTextField() override {
        return std::make_unique<MacTextField>();
    }
};

// Client code
class Application {
public:
    Application(std::unique_ptr<GUIFactory> factory)
        : factory_(std::move(factory)) {}

    void createUI() {
        button_ = factory_->createButton();
        checkbox_ = factory_->createCheckbox();
        textField_ = factory_->createTextField();
    }

    void paint() {
        button_->paint();
        checkbox_->paint();
        textField_->paint();
    }

private:
    std::unique_ptr<GUIFactory> factory_;
    std::unique_ptr<Button> button_;
    std::unique_ptr<Checkbox> checkbox_;
    std::unique_ptr<TextField> textField_;
};

// Usage
int main() {
    std::string osType = "Windows"; // Could be detected at runtime

    std::unique_ptr<GUIFactory> factory;

    if (osType == "Windows") {
        factory = std::make_unique<WindowsFactory>();
    } else {
        factory = std::make_unique<MacFactory>();
    }

    Application app(std::move(factory));
    app.createUI();
    app.paint();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod

# Abstract Products
class Button(ABC):
    @abstractmethod
    def paint(self) -> None:
        pass

    @abstractmethod
    def get_style(self) -> str:
        pass

class Checkbox(ABC):
    @abstractmethod
    def paint(self) -> None:
        pass

    @abstractmethod
    def get_style(self) -> str:
        pass

class TextField(ABC):
    @abstractmethod
    def paint(self) -> None:
        pass

    @abstractmethod
    def get_style(self) -> str:
        pass

# Windows Products
class WindowsButton(Button):
    def paint(self) -> None:
        print("Rendering Windows-style button")

    def get_style(self) -> str:
        return "Windows"

class WindowsCheckbox(Checkbox):
    def paint(self) -> None:
        print("Rendering Windows-style checkbox")

    def get_style(self) -> str:
        return "Windows"

class WindowsTextField(TextField):
    def paint(self) -> None:
        print("Rendering Windows-style text field")

    def get_style(self) -> str:
        return "Windows"

# Mac Products
class MacButton(Button):
    def paint(self) -> None:
        print("Rendering Mac-style button")

    def get_style(self) -> str:
        return "Mac"

class MacCheckbox(Checkbox):
    def paint(self) -> None:
        print("Rendering Mac-style checkbox")

    def get_style(self) -> str:
        return "Mac"

class MacTextField(TextField):
    def paint(self) -> None:
        print("Rendering Mac-style text field")

    def get_style(self) -> str:
        return "Mac"

# Abstract Factory
class GUIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button:
        pass

    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass

    @abstractmethod
    def create_text_field(self) -> TextField:
        pass

# Concrete Factories
class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()

    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()

    def create_text_field(self) -> TextField:
        return WindowsTextField()

class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()

    def create_checkbox(self) -> Checkbox:
        return MacCheckbox()

    def create_text_field(self) -> TextField:
        return MacTextField()

# Client code
class Application:
    def __init__(self, factory: GUIFactory):
        self.factory = factory
        self.button = None
        self.checkbox = None
        self.text_field = None

    def create_ui(self) -> None:
        self.button = self.factory.create_button()
        self.checkbox = self.factory.create_checkbox()
        self.text_field = self.factory.create_text_field()

    def paint(self) -> None:
        self.button.paint()
        self.checkbox.paint()
        self.text_field.paint()

# Usage
if __name__ == "__main__":
    import platform

    os_type = platform.system()

    if os_type == "Windows":
        factory = WindowsFactory()
    else:
        factory = MacFactory()

    app = Application(factory)
    app.create_ui()
    app.paint()
```

**Advantages**:
- Ensures compatibility between products from the same family
- Avoids tight coupling between concrete products and client code
- Single Responsibility Principle: product creation in one place
- Open/Closed Principle: introduce new variants without breaking existing code

**Disadvantages**:
- Code becomes more complicated due to many new interfaces and classes
- Adding new product types requires extending all factories

---

#### Builder Pattern

**Intent**: Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

**Problem**: Creating complex objects with many optional components or configuration options leads to constructor pollution (too many constructor parameters) or many constructors.

**Solution**: Extract object construction code out of its own class and move it to separate objects called builders. The pattern organizes object construction into a set of steps.

**When to Use**:
- Algorithm for creating a complex object should be independent of the parts
- Construction process must allow different representations
- Object has many optional parameters (telescoping constructor problem)

**Real-World Examples**:
- Building complex documents (HTML, PDF)
- Creating database queries
- Building HTTP requests
- Constructing meals at restaurants
- Building cars with various options

**Implementation in C++**:

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Product
class Pizza {
public:
    void setDough(const std::string& dough) { dough_ = dough; }
    void setSauce(const std::string& sauce) { sauce_ = sauce; }
    void setCheese(const std::string& cheese) { cheese_ = cheese; }
    void addTopping(const std::string& topping) { toppings_.push_back(topping); }
    void setSize(const std::string& size) { size_ = size; }
    void setCrust(const std::string& crust) { crust_ = crust; }

    void describe() const {
        std::cout << "Pizza:" << std::endl;
        std::cout << "  Size: " << size_ << std::endl;
        std::cout << "  Dough: " << dough_ << std::endl;
        std::cout << "  Crust: " << crust_ << std::endl;
        std::cout << "  Sauce: " << sauce_ << std::endl;
        std::cout << "  Cheese: " << cheese_ << std::endl;
        std::cout << "  Toppings: ";
        for (const auto& topping : toppings_) {
            std::cout << topping << " ";
        }
        std::cout << std::endl;
    }

private:
    std::string dough_;
    std::string sauce_;
    std::string cheese_;
    std::vector<std::string> toppings_;
    std::string size_;
    std::string crust_;
};

// Abstract Builder
class PizzaBuilder {
public:
    virtual ~PizzaBuilder() = default;

    virtual void buildDough() = 0;
    virtual void buildSauce() = 0;
    virtual void buildCheese() = 0;
    virtual void buildToppings() = 0;
    virtual void buildSize() = 0;
    virtual void buildCrust() = 0;

    std::unique_ptr<Pizza> getPizza() { return std::move(pizza_); }

    void reset() { pizza_ = std::make_unique<Pizza>(); }

protected:
    std::unique_ptr<Pizza> pizza_;
};

// Concrete Builder 1
class MargheritaPizzaBuilder : public PizzaBuilder {
public:
    MargheritaPizzaBuilder() { reset(); }

    void buildDough() override {
        pizza_->setDough("Thin crust dough");
    }

    void buildSauce() override {
        pizza_->setSauce("Tomato sauce");
    }

    void buildCheese() override {
        pizza_->setCheese("Mozzarella");
    }

    void buildToppings() override {
        pizza_->addTopping("Fresh basil");
        pizza_->addTopping("Tomato slices");
    }

    void buildSize() override {
        pizza_->setSize("Medium");
    }

    void buildCrust() override {
        pizza_->setCrust("Regular");
    }
};

// Concrete Builder 2
class PepperoniPizzaBuilder : public PizzaBuilder {
public:
    PepperoniPizzaBuilder() { reset(); }

    void buildDough() override {
        pizza_->setDough("Thick crust dough");
    }

    void buildSauce() override {
        pizza_->setSauce("Spicy tomato sauce");
    }

    void buildCheese() override {
        pizza_->setCheese("Extra mozzarella");
    }

    void buildToppings() override {
        pizza_->addTopping("Pepperoni");
        pizza_->addTopping("Mushrooms");
        pizza_->addTopping("Olives");
    }

    void buildSize() override {
        pizza_->setSize("Large");
    }

    void buildCrust() override {
        pizza_->setCrust("Stuffed");
    }
};

// Director (optional but useful for complex builds)
class PizzaDirector {
public:
    void setBuilder(PizzaBuilder* builder) {
        builder_ = builder;
    }

    void makePizza() {
        builder_->buildSize();
        builder_->buildDough();
        builder_->buildCrust();
        builder_->buildSauce();
        builder_->buildCheese();
        builder_->buildToppings();
    }

private:
    PizzaBuilder* builder_;
};

// Fluent Builder Interface (Modern C++ approach)
class FluentPizzaBuilder {
public:
    FluentPizzaBuilder() : pizza_(std::make_unique<Pizza>()) {}

    FluentPizzaBuilder& setSize(const std::string& size) {
        pizza_->setSize(size);
        return *this;
    }

    FluentPizzaBuilder& setDough(const std::string& dough) {
        pizza_->setDough(dough);
        return *this;
    }

    FluentPizzaBuilder& setCrust(const std::string& crust) {
        pizza_->setCrust(crust);
        return *this;
    }

    FluentPizzaBuilder& setSauce(const std::string& sauce) {
        pizza_->setSauce(sauce);
        return *this;
    }

    FluentPizzaBuilder& setCheese(const std::string& cheese) {
        pizza_->setCheese(cheese);
        return *this;
    }

    FluentPizzaBuilder& addTopping(const std::string& topping) {
        pizza_->addTopping(topping);
        return *this;
    }

    std::unique_ptr<Pizza> build() {
        return std::move(pizza_);
    }

private:
    std::unique_ptr<Pizza> pizza_;
};

// Usage
int main() {
    // Traditional approach with director
    PizzaDirector director;

    MargheritaPizzaBuilder margheritaBuilder;
    director.setBuilder(&margheritaBuilder);
    director.makePizza();
    auto margherita = margheritaBuilder.getPizza();
    margherita->describe();

    std::cout << "\n---\n\n";

    PepperoniPizzaBuilder pepperoniBuilder;
    director.setBuilder(&pepperoniBuilder);
    director.makePizza();
    auto pepperoni = pepperoniBuilder.getPizza();
    pepperoni->describe();

    std::cout << "\n---\n\n";

    // Fluent interface approach
    auto customPizza = FluentPizzaBuilder()
        .setSize("Extra Large")
        .setDough("Whole wheat")
        .setCrust("Thin")
        .setSauce("BBQ sauce")
        .setCheese("Cheddar")
        .addTopping("Chicken")
        .addTopping("Onions")
        .addTopping("Peppers")
        .build();

    customPizza->describe();

    return 0;
}
```

**Implementation in Python**:

```python
from typing import List
from abc import ABC, abstractmethod

# Product
class Pizza:
    def __init__(self):
        self.dough = ""
        self.sauce = ""
        self.cheese = ""
        self.toppings: List[str] = []
        self.size = ""
        self.crust = ""

    def describe(self) -> None:
        print("Pizza:")
        print(f"  Size: {self.size}")
        print(f"  Dough: {self.dough}")
        print(f"  Crust: {self.crust}")
        print(f"  Sauce: {self.sauce}")
        print(f"  Cheese: {self.cheese}")
        print(f"  Toppings: {', '.join(self.toppings)}")

# Abstract Builder
class PizzaBuilder(ABC):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._pizza = Pizza()

    @abstractmethod
    def build_dough(self) -> None:
        pass

    @abstractmethod
    def build_sauce(self) -> None:
        pass

    @abstractmethod
    def build_cheese(self) -> None:
        pass

    @abstractmethod
    def build_toppings(self) -> None:
        pass

    @abstractmethod
    def build_size(self) -> None:
        pass

    @abstractmethod
    def build_crust(self) -> None:
        pass

    def get_pizza(self) -> Pizza:
        pizza = self._pizza
        self.reset()
        return pizza

# Concrete Builders
class MargheritaPizzaBuilder(PizzaBuilder):
    def build_dough(self) -> None:
        self._pizza.dough = "Thin crust dough"

    def build_sauce(self) -> None:
        self._pizza.sauce = "Tomato sauce"

    def build_cheese(self) -> None:
        self._pizza.cheese = "Mozzarella"

    def build_toppings(self) -> None:
        self._pizza.toppings = ["Fresh basil", "Tomato slices"]

    def build_size(self) -> None:
        self._pizza.size = "Medium"

    def build_crust(self) -> None:
        self._pizza.crust = "Regular"

class PepperoniPizzaBuilder(PizzaBuilder):
    def build_dough(self) -> None:
        self._pizza.dough = "Thick crust dough"

    def build_sauce(self) -> None:
        self._pizza.sauce = "Spicy tomato sauce"

    def build_cheese(self) -> None:
        self._pizza.cheese = "Extra mozzarella"

    def build_toppings(self) -> None:
        self._pizza.toppings = ["Pepperoni", "Mushrooms", "Olives"]

    def build_size(self) -> None:
        self._pizza.size = "Large"

    def build_crust(self) -> None:
        self._pizza.crust = "Stuffed"

# Director
class PizzaDirector:
    def __init__(self, builder: PizzaBuilder = None):
        self._builder = builder

    def set_builder(self, builder: PizzaBuilder) -> None:
        self._builder = builder

    def make_pizza(self) -> None:
        self._builder.build_size()
        self._builder.build_dough()
        self._builder.build_crust()
        self._builder.build_sauce()
        self._builder.build_cheese()
        self._builder.build_toppings()

# Fluent Builder (Pythonic approach)
class FluentPizzaBuilder:
    def __init__(self):
        self._pizza = Pizza()

    def set_size(self, size: str):
        self._pizza.size = size
        return self

    def set_dough(self, dough: str):
        self._pizza.dough = dough
        return self

    def set_crust(self, crust: str):
        self._pizza.crust = crust
        return self

    def set_sauce(self, sauce: str):
        self._pizza.sauce = sauce
        return self

    def set_cheese(self, cheese: str):
        self._pizza.cheese = cheese
        return self

    def add_topping(self, topping: str):
        self._pizza.toppings.append(topping)
        return self

    def build(self) -> Pizza:
        return self._pizza

# Usage
if __name__ == "__main__":
    # Traditional approach with director
    director = PizzaDirector()

    margherita_builder = MargheritaPizzaBuilder()
    director.set_builder(margherita_builder)
    director.make_pizza()
    margherita = margherita_builder.get_pizza()
    margherita.describe()

    print("\n---\n")

    pepperoni_builder = PepperoniPizzaBuilder()
    director.set_builder(pepperoni_builder)
    director.make_pizza()
    pepperoni = pepperoni_builder.get_pizza()
    pepperoni.describe()

    print("\n---\n")

    # Fluent interface approach
    custom_pizza = (FluentPizzaBuilder()
        .set_size("Extra Large")
        .set_dough("Whole wheat")
        .set_crust("Thin")
        .set_sauce("BBQ sauce")
        .set_cheese("Cheddar")
        .add_topping("Chicken")
        .add_topping("Onions")
        .add_topping("Peppers")
        .build())

    custom_pizza.describe()
```

**Advantages**:
- Allows construction of complex objects step by step
- Can reuse same construction code for different representations
- Single Responsibility Principle: isolates complex construction code
- Telescoping constructor problem solved

**Disadvantages**:
- Overall complexity increases (many new classes)
- Clients are tied to concrete builder classes

---

#### Prototype Pattern

**Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

**Problem**: Creating objects is expensive (database queries, network calls, complex initialization), and you need many similar objects.

**Solution**: Delegate the cloning process to the actual objects being cloned. Declare a common interface for all objects that support cloning.

**When to Use**:
- Object creation is expensive
- Avoid subclasses of object creator (Factory Method alternative)
- Number of possible object states is limited
- Classes to instantiate are specified at runtime

**Real-World Examples**:
- Cell mitosis in biology
- Copying documents/files
- Cloning game objects with different skins
- Creating test data
- Undo/redo operations

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

// Prototype interface
class Shape {
public:
    virtual ~Shape() = default;
    virtual std::unique_ptr<Shape> clone() const = 0;
    virtual void draw() const = 0;
    virtual std::string getType() const = 0;

    // Common properties
    int x_, y_;
    std::string color_;

protected:
    Shape() : x_(0), y_(0), color_("black") {}
    Shape(int x, int y, const std::string& color)
        : x_(x), y_(y), color_(color) {}
};

// Concrete Prototype 1
class Circle : public Shape {
public:
    int radius_;

    Circle() : Shape(), radius_(10) {}

    Circle(int x, int y, const std::string& color, int radius)
        : Shape(x, y, color), radius_(radius) {}

    // Copy constructor
    Circle(const Circle& other)
        : Shape(other.x_, other.y_, other.color_), radius_(other.radius_) {
        std::cout << "Circle copied" << std::endl;
    }

    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(*this);
    }

    void draw() const override {
        std::cout << "Circle at (" << x_ << "," << y_ << ") "
                  << "with radius " << radius_
                  << " and color " << color_ << std::endl;
    }

    std::string getType() const override {
        return "Circle";
    }
};

// Concrete Prototype 2
class Rectangle : public Shape {
public:
    int width_, height_;

    Rectangle() : Shape(), width_(20), height_(10) {}

    Rectangle(int x, int y, const std::string& color, int width, int height)
        : Shape(x, y, color), width_(width), height_(height) {}

    // Copy constructor
    Rectangle(const Rectangle& other)
        : Shape(other.x_, other.y_, other.color_),
          width_(other.width_), height_(other.height_) {
        std::cout << "Rectangle copied" << std::endl;
    }

    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Rectangle>(*this);
    }

    void draw() const override {
        std::cout << "Rectangle at (" << x_ << "," << y_ << ") "
                  << "with size " << width_ << "x" << height_
                  << " and color " << color_ << std::endl;
    }

    std::string getType() const override {
        return "Rectangle";
    }
};

// Prototype Registry (Prototype Manager)
class ShapeCache {
public:
    static ShapeCache& getInstance() {
        static ShapeCache instance;
        return instance;
    }

    void loadCache() {
        auto circle = std::make_unique<Circle>(0, 0, "red", 15);
        prototypes_["red_circle"] = std::move(circle);

        auto rectangle = std::make_unique<Rectangle>(0, 0, "blue", 30, 20);
        prototypes_["blue_rectangle"] = std::move(rectangle);

        auto smallCircle = std::make_unique<Circle>(0, 0, "green", 5);
        prototypes_["small_green_circle"] = std::move(smallCircle);
    }

    std::unique_ptr<Shape> getShape(const std::string& type) {
        auto it = prototypes_.find(type);
        if (it != prototypes_.end()) {
            return it->second->clone();
        }
        return nullptr;
    }

    void addShape(const std::string& key, std::unique_ptr<Shape> shape) {
        prototypes_[key] = std::move(shape);
    }

private:
    ShapeCache() = default;
    std::unordered_map<std::string, std::unique_ptr<Shape>> prototypes_;
};

// Usage
int main() {
    // Load predefined prototypes
    ShapeCache::getInstance().loadCache();

    // Clone shapes from cache
    auto shape1 = ShapeCache::getInstance().getShape("red_circle");
    shape1->x_ = 10;
    shape1->y_ = 20;
    shape1->draw();

    auto shape2 = ShapeCache::getInstance().getShape("red_circle");
    shape2->x_ = 50;
    shape2->y_ = 60;
    shape2->draw();

    auto shape3 = ShapeCache::getInstance().getShape("blue_rectangle");
    shape3->x_ = 100;
    shape3->y_ = 100;
    shape3->draw();

    // Add custom prototype
    auto customCircle = std::make_unique<Circle>(0, 0, "yellow", 25);
    ShapeCache::getInstance().addShape("custom_yellow", std::move(customCircle));

    auto shape4 = ShapeCache::getInstance().getShape("custom_yellow");
    shape4->draw();

    return 0;
}
```

**Implementation in Python**:

```python
import copy
from abc import ABC, abstractmethod
from typing import Dict

# Prototype interface
class Shape(ABC):
    def __init__(self, x: int = 0, y: int = 0, color: str = "black"):
        self.x = x
        self.y = y
        self.color = color

    @abstractmethod
    def clone(self):
        """Return a deep copy of the object"""
        pass

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass

# Concrete Prototype 1
class Circle(Shape):
    def __init__(self, x: int = 0, y: int = 0, color: str = "black", radius: int = 10):
        super().__init__(x, y, color)
        self.radius = radius

    def clone(self):
        """Deep copy using copy module"""
        print("Circle copied")
        return copy.deepcopy(self)

    def draw(self) -> None:
        print(f"Circle at ({self.x},{self.y}) with radius {self.radius} and color {self.color}")

    def get_type(self) -> str:
        return "Circle"

# Concrete Prototype 2
class Rectangle(Shape):
    def __init__(self, x: int = 0, y: int = 0, color: str = "black",
                 width: int = 20, height: int = 10):
        super().__init__(x, y, color)
        self.width = width
        self.height = height

    def clone(self):
        """Deep copy using copy module"""
        print("Rectangle copied")
        return copy.deepcopy(self)

    def draw(self) -> None:
        print(f"Rectangle at ({self.x},{self.y}) with size {self.width}x{self.height} and color {self.color}")

    def get_type(self) -> str:
        return "Rectangle"

# Prototype Registry
class ShapeCache:
    _instance = None
    _prototypes: Dict[str, Shape] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_cache(self) -> None:
        """Load predefined prototypes"""
        self._prototypes["red_circle"] = Circle(0, 0, "red", 15)
        self._prototypes["blue_rectangle"] = Rectangle(0, 0, "blue", 30, 20)
        self._prototypes["small_green_circle"] = Circle(0, 0, "green", 5)

    def get_shape(self, shape_type: str) -> Shape:
        """Clone a shape from the cache"""
        prototype = self._prototypes.get(shape_type)
        if prototype:
            return prototype.clone()
        raise ValueError(f"Shape type '{shape_type}' not found in cache")

    def add_shape(self, key: str, shape: Shape) -> None:
        """Add a new prototype to the cache"""
        self._prototypes[key] = shape

# Usage
if __name__ == "__main__":
    # Load predefined prototypes
    cache = ShapeCache()
    cache.load_cache()

    # Clone shapes from cache
    shape1 = cache.get_shape("red_circle")
    shape1.x, shape1.y = 10, 20
    shape1.draw()

    shape2 = cache.get_shape("red_circle")
    shape2.x, shape2.y = 50, 60
    shape2.draw()

    shape3 = cache.get_shape("blue_rectangle")
    shape3.x, shape3.y = 100, 100
    shape3.draw()

    # Add custom prototype
    custom_circle = Circle(0, 0, "yellow", 25)
    cache.add_shape("custom_yellow", custom_circle)

    shape4 = cache.get_shape("custom_yellow")
    shape4.draw()
```

**Advantages**:
- Reduces cost of creating complex objects
- Hides complexity of creating new instances
- Allows adding/removing products at runtime
- Configures application with classes dynamically

**Disadvantages**:
- Cloning complex objects with circular references can be tricky
- Deep vs shallow copy considerations

**Related Patterns**:
- Often used with Composite and Decorator patterns
- Designs that use Factory Method can use Prototype instead

### Structural Patterns

#### Adapter Pattern

**Intent**: Convert the interface of a class into another interface that clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.

**Problem**: You want to use an existing class, but its interface doesn't match the one you need. You can't modify the existing class (third-party library, legacy code, or you want to keep it unchanged).

**Solution**: Create an adapter class that wraps the incompatible object and translates calls from the expected interface to the adaptee's interface. There are two main approaches: class adapter (using multiple inheritance) and object adapter (using composition).

**When to Use**:
- You want to use an existing class with an incompatible interface
- You need to create a reusable class that cooperates with unrelated classes
- You need to use several existing subclasses, but it's impractical to adapt their interface by subclassing each one (use object adapter)
- Integrating legacy code with new systems
- Working with third-party libraries

**Real-World Examples**:
- Power adapters (110V to 220V conversion)
- Card readers for different memory card formats
- Media player supporting multiple audio formats
- Database drivers adapting different database APIs
- XML to JSON converters
- Legacy system integration

**Implementation in C++ (Object Adapter)**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

// Target interface - What the client expects
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& audioType, const std::string& fileName) = 0;
};

// Adaptee 1 - Advanced MP4 player with incompatible interface
class AdvancedMP4Player {
public:
    void playMP4(const std::string& fileName) {
        std::cout << "Playing MP4 file: " << fileName << std::endl;
    }
};

// Adaptee 2 - VLC player with incompatible interface
class VLCPlayer {
public:
    void playVLC(const std::string& fileName) {
        std::cout << "Playing VLC file: " << fileName << std::endl;
    }
};

// Adapter - Adapts AdvancedMP4Player and VLCPlayer to MediaPlayer interface
class MediaAdapter : public MediaPlayer {
public:
    MediaAdapter(const std::string& audioType) {
        if (audioType == "mp4") {
            mp4Player_ = std::make_unique<AdvancedMP4Player>();
        } else if (audioType == "vlc") {
            vlcPlayer_ = std::make_unique<VLCPlayer>();
        }
    }

    void play(const std::string& audioType, const std::string& fileName) override {
        if (audioType == "mp4") {
            mp4Player_->playMP4(fileName);
        } else if (audioType == "vlc") {
            vlcPlayer_->playVLC(fileName);
        }
    }

private:
    std::unique_ptr<AdvancedMP4Player> mp4Player_;
    std::unique_ptr<VLCPlayer> vlcPlayer_;
};

// Concrete implementation of target interface
class AudioPlayer : public MediaPlayer {
public:
    void play(const std::string& audioType, const std::string& fileName) override {
        // Built-in support for mp3
        if (audioType == "mp3") {
            std::cout << "Playing MP3 file: " << fileName << std::endl;
        }
        // Use adapter for other formats
        else if (audioType == "mp4" || audioType == "vlc") {
            auto adapter = std::make_unique<MediaAdapter>(audioType);
            adapter->play(audioType, fileName);
        } else {
            std::cout << "Invalid media type: " << audioType << std::endl;
        }
    }
};

// Real-world example: Shape compatibility (legacy square to new rectangle interface)
class LegacyRectangle {
public:
    void draw(int x1, int y1, int x2, int y2) {
        std::cout << "Legacy Rectangle from (" << x1 << "," << y1
                  << ") to (" << x2 << "," << y2 << ")" << std::endl;
    }
};

// New shape interface
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() = 0;
    virtual void resize(int percentage) = 0;
};

// Adapter for legacy rectangle
class RectangleAdapter : public Shape {
public:
    RectangleAdapter(int x, int y, int width, int height)
        : x_(x), y_(y), width_(width), height_(height) {
        legacyRect_ = std::make_unique<LegacyRectangle>();
    }

    void draw() override {
        legacyRect_->draw(x_, y_, x_ + width_, y_ + height_);
    }

    void resize(int percentage) override {
        width_ = static_cast<int>(width_ * percentage / 100.0);
        height_ = static_cast<int>(height_ * percentage / 100.0);
        std::cout << "Resized to " << width_ << "x" << height_ << std::endl;
    }

private:
    std::unique_ptr<LegacyRectangle> legacyRect_;
    int x_, y_, width_, height_;
};
```

**Class Adapter Example (Using Multiple Inheritance)**:

```cpp
// Class adapter - inherits from both target and adaptee
class ClassMediaAdapter : public MediaPlayer, private AdvancedMP4Player {
public:
    void play(const std::string& audioType, const std::string& fileName) override {
        if (audioType == "mp4") {
            playMP4(fileName);  // Direct call to inherited method
        }
    }
};

// Usage
int main() {
    // Object adapter example
    AudioPlayer player;
    player.play("mp3", "song.mp3");
    player.play("mp4", "video.mp4");
    player.play("vlc", "movie.vlc");
    player.play("avi", "movie.avi");

    std::cout << "\n---\n\n";

    // Shape adapter example
    RectangleAdapter rect(10, 20, 100, 50);
    rect.draw();
    rect.resize(150);
    rect.draw();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod

# Target interface
class MediaPlayer(ABC):
    @abstractmethod
    def play(self, audio_type: str, file_name: str) -> None:
        pass

# Adaptee 1
class AdvancedMP4Player:
    def play_mp4(self, file_name: str) -> None:
        print(f"Playing MP4 file: {file_name}")

# Adaptee 2
class VLCPlayer:
    def play_vlc(self, file_name: str) -> None:
        print(f"Playing VLC file: {file_name}")

# Adapter
class MediaAdapter(MediaPlayer):
    def __init__(self, audio_type: str):
        self.audio_type = audio_type
        if audio_type == "mp4":
            self.advanced_player = AdvancedMP4Player()
        elif audio_type == "vlc":
            self.advanced_player = VLCPlayer()

    def play(self, audio_type: str, file_name: str) -> None:
        if audio_type == "mp4":
            self.advanced_player.play_mp4(file_name)
        elif audio_type == "vlc":
            self.advanced_player.play_vlc(file_name)

# Concrete target
class AudioPlayer(MediaPlayer):
    def play(self, audio_type: str, file_name: str) -> None:
        # Built-in support for mp3
        if audio_type == "mp3":
            print(f"Playing MP3 file: {file_name}")
        # Use adapter for other formats
        elif audio_type in ["mp4", "vlc"]:
            adapter = MediaAdapter(audio_type)
            adapter.play(audio_type, file_name)
        else:
            print(f"Invalid media type: {audio_type}")

# Legacy system adapter example
class LegacyRectangle:
    def draw(self, x1: int, y1: int, x2: int, y2: int) -> None:
        print(f"Legacy Rectangle from ({x1},{y1}) to ({x2},{y2})")

class Shape(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def resize(self, percentage: int) -> None:
        pass

class RectangleAdapter(Shape):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.legacy_rect = LegacyRectangle()
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self) -> None:
        self.legacy_rect.draw(self.x, self.y, self.x + self.width, self.y + self.height)

    def resize(self, percentage: int) -> None:
        self.width = int(self.width * percentage / 100)
        self.height = int(self.height * percentage / 100)
        print(f"Resized to {self.width}x{self.height}")

# Usage
if __name__ == "__main__":
    player = AudioPlayer()
    player.play("mp3", "song.mp3")
    player.play("mp4", "video.mp4")
    player.play("vlc", "movie.vlc")
    player.play("avi", "movie.avi")

    print("\n---\n")

    rect = RectangleAdapter(10, 20, 100, 50)
    rect.draw()
    rect.resize(150)
    rect.draw()
```

**Advantages**:
- Single Responsibility Principle: separate interface conversion from business logic
- Open/Closed Principle: introduce new adapters without changing existing code
- Flexibility in adapting multiple incompatible interfaces
- Reuses existing functionality without modification

**Disadvantages**:
- Overall complexity increases due to new interfaces and classes
- Sometimes it's simpler to just change the service class to match the rest of your code

**Related Patterns**:
- **Bridge**: Separates interface from implementation (designed upfront), whereas Adapter makes existing classes work together (retrofitted)
- **Decorator**: Enhances object without changing interface; Adapter changes the interface
- **Proxy**: Provides same interface; Adapter provides different interface

---

#### Bridge Pattern

**Intent**: Decouple an abstraction from its implementation so that the two can vary independently.

**Problem**: When an abstraction can have multiple implementations and you want to avoid a permanent binding between them. Without Bridge, you end up with a combinatorial explosion of subclasses (e.g., Shape → CircleShape, SquareShape; Renderer → OpenGLRenderer, DirectXRenderer → OpenGLCircle, DirectXCircle, OpenGLSquare, DirectXSquare).

**Solution**: Separate the abstraction hierarchy from the implementation hierarchy. The abstraction contains a reference to the implementation and delegates the actual work to it.

**When to Use**:
- You want to avoid permanent binding between abstraction and implementation
- Both abstraction and implementation should be extensible by subclassing
- Changes in implementation shouldn't affect client code
- You want to share implementation among multiple objects (copy-on-write)
- You have a proliferation of classes from a coupled interface/implementation

**Real-World Examples**:
- Graphics rendering across different platforms (OpenGL, DirectX, Vulkan)
- Database drivers (abstract DB operations vs specific database implementations)
- GUI frameworks across operating systems
- Remote controls and devices (abstraction: remote, implementation: TV, radio, etc.)
- Payment processing across different payment gateways

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>

// Implementation hierarchy
class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void renderCircle(float radius) = 0;
    virtual void renderSquare(float side) = 0;
    virtual std::string getName() const = 0;
};

class OpenGLRenderer : public Renderer {
public:
    void renderCircle(float radius) override {
        std::cout << "[OpenGL] Drawing circle with radius " << radius << std::endl;
    }

    void renderSquare(float side) override {
        std::cout << "[OpenGL] Drawing square with side " << side << std::endl;
    }

    std::string getName() const override {
        return "OpenGL";
    }
};

class DirectXRenderer : public Renderer {
public:
    void renderCircle(float radius) override {
        std::cout << "[DirectX] Rendering circle with radius " << radius << std::endl;
    }

    void renderSquare(float side) override {
        std::cout << "[DirectX] Rendering square with side " << side << std::endl;
    }

    std::string getName() const override {
        return "DirectX";
    }
};

class VulkanRenderer : public Renderer {
public:
    void renderCircle(float radius) override {
        std::cout << "[Vulkan] Rendering circle with radius " << radius << std::endl;
    }

    void renderSquare(float side) override {
        std::cout << "[Vulkan] Rendering square with side " << side << std::endl;
    }

    std::string getName() const override {
        return "Vulkan";
    }
};

// Abstraction hierarchy
class Shape {
public:
    virtual ~Shape() = default;

    Shape(std::unique_ptr<Renderer> renderer)
        : renderer_(std::move(renderer)) {}

    virtual void draw() = 0;
    virtual void resize(float factor) = 0;

protected:
    std::unique_ptr<Renderer> renderer_;
};

class Circle : public Shape {
public:
    Circle(std::unique_ptr<Renderer> renderer, float radius)
        : Shape(std::move(renderer)), radius_(radius) {}

    void draw() override {
        std::cout << "Circle: ";
        renderer_->renderCircle(radius_);
    }

    void resize(float factor) override {
        radius_ *= factor;
        std::cout << "Circle resized to radius " << radius_ << std::endl;
    }

private:
    float radius_;
};

class Square : public Shape {
public:
    Square(std::unique_ptr<Renderer> renderer, float side)
        : Shape(std::move(renderer)), side_(side) {}

    void draw() override {
        std::cout << "Square: ";
        renderer_->renderSquare(side_);
    }

    void resize(float factor) override {
        side_ *= factor;
        std::cout << "Square resized to side " << side_ << std::endl;
    }

private:
    float side_;
};

// Real-world example: Remote control and devices
class Device {
public:
    virtual ~Device() = default;
    virtual void powerOn() = 0;
    virtual void powerOff() = 0;
    virtual void setVolume(int volume) = 0;
    virtual void setChannel(int channel) = 0;
};

class TV : public Device {
public:
    void powerOn() override {
        std::cout << "TV: Power ON" << std::endl;
    }

    void powerOff() override {
        std::cout << "TV: Power OFF" << std::endl;
    }

    void setVolume(int volume) override {
        std::cout << "TV: Setting volume to " << volume << std::endl;
    }

    void setChannel(int channel) override {
        std::cout << "TV: Switching to channel " << channel << std::endl;
    }
};

class Radio : public Device {
public:
    void powerOn() override {
        std::cout << "Radio: Power ON" << std::endl;
    }

    void powerOff() override {
        std::cout << "Radio: Power OFF" << std::endl;
    }

    void setVolume(int volume) override {
        std::cout << "Radio: Setting volume to " << volume << std::endl;
    }

    void setChannel(int channel) override {
        std::cout << "Radio: Tuning to station " << channel << " MHz" << std::endl;
    }
};

class RemoteControl {
public:
    RemoteControl(std::shared_ptr<Device> device)
        : device_(device) {}

    virtual ~RemoteControl() = default;

    void togglePower() {
        if (isOn_) {
            device_->powerOff();
            isOn_ = false;
        } else {
            device_->powerOn();
            isOn_ = true;
        }
    }

    void volumeUp() {
        volume_ = std::min(volume_ + 10, 100);
        device_->setVolume(volume_);
    }

    void volumeDown() {
        volume_ = std::max(volume_ - 10, 0);
        device_->setVolume(volume_);
    }

    void channelUp() {
        channel_++;
        device_->setChannel(channel_);
    }

    void channelDown() {
        channel_--;
        device_->setChannel(channel_);
    }

protected:
    std::shared_ptr<Device> device_;
    bool isOn_ = false;
    int volume_ = 50;
    int channel_ = 1;
};

class AdvancedRemoteControl : public RemoteControl {
public:
    using RemoteControl::RemoteControl;

    void mute() {
        device_->setVolume(0);
        std::cout << "Device muted" << std::endl;
    }
};

// Usage
int main() {
    // Bridge pattern with shapes and renderers
    auto circle1 = std::make_unique<Circle>(std::make_unique<OpenGLRenderer>(), 5.0f);
    circle1->draw();
    circle1->resize(1.5f);
    circle1->draw();

    std::cout << "\n";

    auto square1 = std::make_unique<Square>(std::make_unique<DirectXRenderer>(), 10.0f);
    square1->draw();

    std::cout << "\n";

    auto circle2 = std::make_unique<Circle>(std::make_unique<VulkanRenderer>(), 7.0f);
    circle2->draw();

    std::cout << "\n---\n\n";

    // Remote control example
    auto tv = std::make_shared<TV>();
    RemoteControl tvRemote(tv);
    tvRemote.togglePower();
    tvRemote.volumeUp();
    tvRemote.channelUp();

    std::cout << "\n";

    auto radio = std::make_shared<Radio>();
    AdvancedRemoteControl radioRemote(radio);
    radioRemote.togglePower();
    radioRemote.volumeUp();
    radioRemote.mute();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import Protocol

# Implementation hierarchy
class Renderer(ABC):
    @abstractmethod
    def render_circle(self, radius: float) -> None:
        pass

    @abstractmethod
    def render_square(self, side: float) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class OpenGLRenderer(Renderer):
    def render_circle(self, radius: float) -> None:
        print(f"[OpenGL] Drawing circle with radius {radius}")

    def render_square(self, side: float) -> None:
        print(f"[OpenGL] Drawing square with side {side}")

    def get_name(self) -> str:
        return "OpenGL"

class DirectXRenderer(Renderer):
    def render_circle(self, radius: float) -> None:
        print(f"[DirectX] Rendering circle with radius {radius}")

    def render_square(self, side: float) -> None:
        print(f"[DirectX] Rendering square with side {side}")

    def get_name(self) -> str:
        return "DirectX"

# Abstraction hierarchy
class Shape(ABC):
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def resize(self, factor: float) -> None:
        pass

class Circle(Shape):
    def __init__(self, renderer: Renderer, radius: float):
        super().__init__(renderer)
        self.radius = radius

    def draw(self) -> None:
        print("Circle: ", end="")
        self.renderer.render_circle(self.radius)

    def resize(self, factor: float) -> None:
        self.radius *= factor
        print(f"Circle resized to radius {self.radius}")

class Square(Shape):
    def __init__(self, renderer: Renderer, side: float):
        super().__init__(renderer)
        self.side = side

    def draw(self) -> None:
        print("Square: ", end="")
        self.renderer.render_square(self.side)

    def resize(self, factor: float) -> None:
        self.side *= factor
        print(f"Square resized to side {self.side}")

# Device example
class Device(ABC):
    @abstractmethod
    def power_on(self) -> None:
        pass

    @abstractmethod
    def power_off(self) -> None:
        pass

    @abstractmethod
    def set_volume(self, volume: int) -> None:
        pass

    @abstractmethod
    def set_channel(self, channel: int) -> None:
        pass

class TV(Device):
    def power_on(self) -> None:
        print("TV: Power ON")

    def power_off(self) -> None:
        print("TV: Power OFF")

    def set_volume(self, volume: int) -> None:
        print(f"TV: Setting volume to {volume}")

    def set_channel(self, channel: int) -> None:
        print(f"TV: Switching to channel {channel}")

class RemoteControl:
    def __init__(self, device: Device):
        self.device = device
        self.is_on = False
        self.volume = 50
        self.channel = 1

    def toggle_power(self) -> None:
        if self.is_on:
            self.device.power_off()
            self.is_on = False
        else:
            self.device.power_on()
            self.is_on = True

    def volume_up(self) -> None:
        self.volume = min(self.volume + 10, 100)
        self.device.set_volume(self.volume)

    def channel_up(self) -> None:
        self.channel += 1
        self.device.set_channel(self.channel)

# Usage
if __name__ == "__main__":
    circle1 = Circle(OpenGLRenderer(), 5.0)
    circle1.draw()
    circle1.resize(1.5)
    circle1.draw()

    print()

    square1 = Square(DirectXRenderer(), 10.0)
    square1.draw()

    print("\n---\n")

    tv = TV()
    remote = RemoteControl(tv)
    remote.toggle_power()
    remote.volume_up()
    remote.channel_up()
```

**Advantages**:
- Decouples interface from implementation
- Improves extensibility (extend abstraction and implementation independently)
- Hides implementation details from client
- Allows switching implementations at runtime
- Reduces number of subclasses in hierarchies

**Disadvantages**:
- Increases complexity with additional layers of indirection
- Can be harder to understand initially

**Related Patterns**:
- **Abstract Factory**: Can create and configure a particular Bridge
- **Adapter**: Makes unrelated classes work together (retrofitted); Bridge separates abstraction from implementation (designed upfront)

---

#### Composite Pattern

**Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.

**Problem**: You need to represent a hierarchy of objects where individual objects and groups of objects should be treated uniformly. Without Composite, clients must differentiate between leaf nodes and branches.

**Solution**: Define a common interface for both simple (leaf) and complex (composite) objects. Composite objects delegate operations to their children.

**When to Use**:
- You want to represent part-whole hierarchies
- You want clients to ignore the difference between compositions and individual objects
- You have tree-structured data (file systems, GUI components, organization charts)
- You need recursive composition

**Real-World Examples**:
- File systems (files and directories)
- GUI component hierarchies (panels containing buttons, labels, other panels)
- Organization charts (departments containing employees and sub-departments)
- Graphics scenes (shapes containing other shapes)
- Menu systems (menus containing menu items and submenus)

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// Component - Common interface for leaves and composites
class FileSystemComponent {
public:
    virtual ~FileSystemComponent() = default;

    virtual std::string getName() const = 0;
    virtual int getSize() const = 0;
    virtual void display(int depth = 0) const = 0;

    // Composite methods (default implementations)
    virtual void add(std::shared_ptr<FileSystemComponent> component) {
        throw std::runtime_error("Operation not supported");
    }

    virtual void remove(std::shared_ptr<FileSystemComponent> component) {
        throw std::runtime_error("Operation not supported");
    }

    virtual std::shared_ptr<FileSystemComponent> getChild(int index) {
        throw std::runtime_error("Operation not supported");
    }
};

// Leaf - File
class File : public FileSystemComponent {
public:
    File(const std::string& name, int size)
        : name_(name), size_(size) {}

    std::string getName() const override {
        return name_;
    }

    int getSize() const override {
        return size_;
    }

    void display(int depth = 0) const override {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "📄 " << name_ << " (" << size_ << " KB)" << std::endl;
    }

private:
    std::string name_;
    int size_;
};

// Composite - Directory
class Directory : public FileSystemComponent {
public:
    Directory(const std::string& name)
        : name_(name) {}

    std::string getName() const override {
        return name_;
    }

    int getSize() const override {
        int totalSize = 0;
        for (const auto& child : children_) {
            totalSize += child->getSize();
        }
        return totalSize;
    }

    void display(int depth = 0) const override {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "📁 " << name_ << " (" << getSize() << " KB total)" << std::endl;
        for (const auto& child : children_) {
            child->display(depth + 1);
        }
    }

    void add(std::shared_ptr<FileSystemComponent> component) override {
        children_.push_back(component);
    }

    void remove(std::shared_ptr<FileSystemComponent> component) override {
        auto it = std::find(children_.begin(), children_.end(), component);
        if (it != children_.end()) {
            children_.erase(it);
        }
    }

    std::shared_ptr<FileSystemComponent> getChild(int index) override {
        if (index >= 0 && index < children_.size()) {
            return children_[index];
        }
        return nullptr;
    }

private:
    std::string name_;
    std::vector<std::shared_ptr<FileSystemComponent>> children_;
};

// Another example: Graphics
class Graphic {
public:
    virtual ~Graphic() = default;
    virtual void draw() const = 0;
    virtual void move(int x, int y) = 0;
};

class Circle : public Graphic {
public:
    Circle(int x, int y, int radius)
        : x_(x), y_(y), radius_(radius) {}

    void draw() const override {
        std::cout << "Circle at (" << x_ << "," << y_ << ") with radius " << radius_ << std::endl;
    }

    void move(int x, int y) override {
        x_ += x;
        y_ += y;
    }

private:
    int x_, y_, radius_;
};

class Rectangle : public Graphic {
public:
    Rectangle(int x, int y, int width, int height)
        : x_(x), y_(y), width_(width), height_(height) {}

    void draw() const override {
        std::cout << "Rectangle at (" << x_ << "," << y_ << ") "
                  << width_ << "x" << height_ << std::endl;
    }

    void move(int x, int y) override {
        x_ += x;
        y_ += y;
    }

private:
    int x_, y_, width_, height_;
};

class CompositeGraphic : public Graphic {
public:
    void draw() const override {
        std::cout << "Composite graphic containing:" << std::endl;
        for (const auto& graphic : graphics_) {
            graphic->draw();
        }
    }

    void move(int x, int y) override {
        for (auto& graphic : graphics_) {
            graphic->move(x, y);
        }
    }

    void add(std::shared_ptr<Graphic> graphic) {
        graphics_.push_back(graphic);
    }

    void remove(std::shared_ptr<Graphic> graphic) {
        auto it = std::find(graphics_.begin(), graphics_.end(), graphic);
        if (it != graphics_.end()) {
            graphics_.erase(it);
        }
    }

private:
    std::vector<std::shared_ptr<Graphic>> graphics_;
};

// Usage
int main() {
    // File system example
    auto root = std::make_shared<Directory>("root");

    auto home = std::make_shared<Directory>("home");
    auto documents = std::make_shared<Directory>("documents");

    auto file1 = std::make_shared<File>("resume.pdf", 150);
    auto file2 = std::make_shared<File>("photo.jpg", 2500);
    auto file3 = std::make_shared<File>("notes.txt", 45);

    documents->add(file1);
    documents->add(file3);
    home->add(documents);
    home->add(file2);

    auto usr = std::make_shared<Directory>("usr");
    auto bin = std::make_shared<Directory>("bin");
    auto lib = std::make_shared<Directory>("lib");

    auto file4 = std::make_shared<File>("bash", 1200);
    auto file5 = std::make_shared<File>("python", 4500);

    bin->add(file4);
    bin->add(file5);
    usr->add(bin);
    usr->add(lib);

    root->add(home);
    root->add(usr);

    root->display();

    std::cout << "\n---\n\n";

    // Graphics example
    auto allGraphics = std::make_shared<CompositeGraphic>();

    auto circle1 = std::make_shared<Circle>(10, 10, 5);
    auto circle2 = std::make_shared<Circle>(20, 20, 10);
    auto rect1 = std::make_shared<Rectangle>(5, 5, 15, 20);

    auto group1 = std::make_shared<CompositeGraphic>();
    group1->add(circle1);
    group1->add(circle2);

    allGraphics->add(group1);
    allGraphics->add(rect1);

    std::cout << "Drawing all graphics:" << std::endl;
    allGraphics->draw();

    std::cout << "\nMoving all graphics by (5, 5):" << std::endl;
    allGraphics->move(5, 5);
    allGraphics->draw();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import List

# Component
class FileSystemComponent(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def display(self, depth: int = 0) -> None:
        pass

    def add(self, component: 'FileSystemComponent') -> None:
        raise NotImplementedError("Operation not supported")

    def remove(self, component: 'FileSystemComponent') -> None:
        raise NotImplementedError("Operation not supported")

# Leaf
class File(FileSystemComponent):
    def __init__(self, name: str, size: int):
        self._name = name
        self._size = size

    def get_name(self) -> str:
        return self._name

    def get_size(self) -> int:
        return self._size

    def display(self, depth: int = 0) -> None:
        indent = "  " * depth
        print(f"{indent}📄 {self._name} ({self._size} KB)")

# Composite
class Directory(FileSystemComponent):
    def __init__(self, name: str):
        self._name = name
        self._children: List[FileSystemComponent] = []

    def get_name(self) -> str:
        return self._name

    def get_size(self) -> int:
        return sum(child.get_size() for child in self._children)

    def display(self, depth: int = 0) -> None:
        indent = "  " * depth
        print(f"{indent}📁 {self._name} ({self.get_size()} KB total)")
        for child in self._children:
            child.display(depth + 1)

    def add(self, component: FileSystemComponent) -> None:
        self._children.append(component)

    def remove(self, component: FileSystemComponent) -> None:
        self._children.remove(component)

# Graphics example
class Graphic(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def move(self, x: int, y: int) -> None:
        pass

class Circle(Graphic):
    def __init__(self, x: int, y: int, radius: int):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self) -> None:
        print(f"Circle at ({self.x},{self.y}) with radius {self.radius}")

    def move(self, x: int, y: int) -> None:
        self.x += x
        self.y += y

class CompositeGraphic(Graphic):
    def __init__(self):
        self.graphics: List[Graphic] = []

    def draw(self) -> None:
        print("Composite graphic containing:")
        for graphic in self.graphics:
            graphic.draw()

    def move(self, x: int, y: int) -> None:
        for graphic in self.graphics:
            graphic.move(x, y)

    def add(self, graphic: Graphic) -> None:
        self.graphics.append(graphic)

# Usage
if __name__ == "__main__":
    root = Directory("root")
    home = Directory("home")
    documents = Directory("documents")

    file1 = File("resume.pdf", 150)
    file2 = File("photo.jpg", 2500)
    file3 = File("notes.txt", 45)

    documents.add(file1)
    documents.add(file3)
    home.add(documents)
    home.add(file2)

    root.add(home)
    root.display()

    print("\n---\n")

    # Graphics
    all_graphics = CompositeGraphic()
    circle1 = Circle(10, 10, 5)
    circle2 = Circle(20, 20, 10)

    group1 = CompositeGraphic()
    group1.add(circle1)
    group1.add(circle2)

    all_graphics.add(group1)

    print("Drawing all graphics:")
    all_graphics.draw()

    print("\nMoving all by (5, 5):")
    all_graphics.move(5, 5)
    all_graphics.draw()
```

**Advantages**:
- Simplifies client code (treats individual and composite objects uniformly)
- Makes it easier to add new component types
- Supports recursive composition naturally
- Open/Closed Principle: can introduce new elements without breaking existing code

**Disadvantages**:
- Can make design overly general
- Can be difficult to restrict components of a composite
- Type safety: hard to enforce that composite contains only certain types

**Related Patterns**:
- **Iterator**: Often used to traverse composites
- **Visitor**: Can apply operations across composite structure
- **Decorator**: Often used together with Composite

---

#### Decorator Pattern

**Intent**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Problem**: You need to add responsibilities to individual objects without affecting other objects or using subclassing (which is static and affects all instances).

**Solution**: Create decorator classes that wrap the original object. Each decorator implements the same interface as the wrapped object and adds its own behavior before/after delegating to the wrapped object.

**When to Use**:
- You need to add responsibilities to objects dynamically and transparently
- Extension by subclassing is impractical (class is final, or leads to many subclasses)
- Responsibilities can be withdrawn
- You want to add features to objects without changing their interface

**Real-World Examples**:
- Coffee shop beverages with add-ons (milk, sugar, whipped cream)
- Text formatting (bold, italic, underline combinations)
- GUI components with borders, scrollbars
- Stream processing (buffered, compressed, encrypted)
- Middleware in web frameworks

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>

// Component interface
class Coffee {
public:
    virtual ~Coffee() = default;
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

// Concrete Component
class SimpleCoffee : public Coffee {
public:
    std::string getDescription() const override {
        return "Simple Coffee";
    }

    double getCost() const override {
        return 2.0;
    }
};

// Base Decorator
class CoffeeDecorator : public Coffee {
public:
    CoffeeDecorator(std::unique_ptr<Coffee> coffee)
        : coffee_(std::move(coffee)) {}

protected:
    std::unique_ptr<Coffee> coffee_;
};

// Concrete Decorators
class MilkDecorator : public CoffeeDecorator {
public:
    using CoffeeDecorator::CoffeeDecorator;

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Milk";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.5;
    }
};

class SugarDecorator : public CoffeeDecorator {
public:
    using CoffeeDecorator::CoffeeDecorator;

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Sugar";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.2;
    }
};

class WhippedCreamDecorator : public CoffeeDecorator {
public:
    using CoffeeDecorator::CoffeeDecorator;

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Whipped Cream";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.7;
    }
};

class CaramelDecorator : public CoffeeDecorator {
public:
    using CoffeeDecorator::CoffeeDecorator;

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Caramel";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.6;
    }
};

// Another example: Text formatting
class Text {
public:
    virtual ~Text() = default;
    virtual std::string render() const = 0;
};

class PlainText : public Text {
public:
    PlainText(const std::string& content) : content_(content) {}

    std::string render() const override {
        return content_;
    }

private:
    std::string content_;
};

class TextDecorator : public Text {
public:
    TextDecorator(std::unique_ptr<Text> text)
        : text_(std::move(text)) {}

protected:
    std::unique_ptr<Text> text_;
};

class BoldDecorator : public TextDecorator {
public:
    using TextDecorator::TextDecorator;

    std::string render() const override {
        return "<b>" + text_->render() + "</b>";
    }
};

class ItalicDecorator : public TextDecorator {
public:
    using TextDecorator::TextDecorator;

    std::string render() const override {
        return "<i>" + text_->render() + "</i>";
    }
};

class UnderlineDecorator : public TextDecorator {
public:
    using TextDecorator::TextDecorator;

    std::string render() const override {
        return "<u>" + text_->render() + "</u>";
    }
};

// Data stream example
class DataSource {
public:
    virtual ~DataSource() = default;
    virtual void writeData(const std::string& data) = 0;
    virtual std::string readData() = 0;
};

class FileDataSource : public DataSource {
public:
    FileDataSource(const std::string& filename)
        : filename_(filename) {}

    void writeData(const std::string& data) override {
        std::cout << "Writing to file '" << filename_ << "': " << data << std::endl;
        data_ = data;
    }

    std::string readData() override {
        std::cout << "Reading from file '" << filename_ << "'" << std::endl;
        return data_;
    }

private:
    std::string filename_;
    std::string data_;
};

class DataSourceDecorator : public DataSource {
public:
    DataSourceDecorator(std::unique_ptr<DataSource> source)
        : wrappee_(std::move(source)) {}

protected:
    std::unique_ptr<DataSource> wrappee_;
};

class EncryptionDecorator : public DataSourceDecorator {
public:
    using DataSourceDecorator::DataSourceDecorator;

    void writeData(const std::string& data) override {
        std::string encrypted = encrypt(data);
        wrappee_->writeData(encrypted);
    }

    std::string readData() override {
        std::string data = wrappee_->readData();
        return decrypt(data);
    }

private:
    std::string encrypt(const std::string& data) {
        std::cout << "Encrypting data..." << std::endl;
        return "[ENCRYPTED]" + data + "[/ENCRYPTED]";
    }

    std::string decrypt(const std::string& data) {
        std::cout << "Decrypting data..." << std::endl;
        // Simple decryption simulation
        if (data.find("[ENCRYPTED]") == 0) {
            return data.substr(11, data.length() - 23);
        }
        return data;
    }
};

class CompressionDecorator : public DataSourceDecorator {
public:
    using DataSourceDecorator::DataSourceDecorator;

    void writeData(const std::string& data) override {
        std::string compressed = compress(data);
        wrappee_->writeData(compressed);
    }

    std::string readData() override {
        std::string data = wrappee_->readData();
        return decompress(data);
    }

private:
    std::string compress(const std::string& data) {
        std::cout << "Compressing data..." << std::endl;
        return "[COMPRESSED]" + data + "[/COMPRESSED]";
    }

    std::string decompress(const std::string& data) {
        std::cout << "Decompressing data..." << std::endl;
        if (data.find("[COMPRESSED]") == 0) {
            return data.substr(12, data.length() - 26);
        }
        return data;
    }
};

// Usage
int main() {
    // Coffee example
    std::unique_ptr<Coffee> myCoffee = std::make_unique<SimpleCoffee>();
    std::cout << myCoffee->getDescription() << " - $" << myCoffee->getCost() << std::endl;

    myCoffee = std::make_unique<MilkDecorator>(std::move(myCoffee));
    std::cout << myCoffee->getDescription() << " - $" << myCoffee->getCost() << std::endl;

    myCoffee = std::make_unique<SugarDecorator>(std::move(myCoffee));
    std::cout << myCoffee->getDescription() << " - $" << myCoffee->getCost() << std::endl;

    myCoffee = std::make_unique<WhippedCreamDecorator>(std::move(myCoffee));
    std::cout << myCoffee->getDescription() << " - $" << myCoffee->getCost() << std::endl;

    std::cout << "\n---\n\n";

    // Text formatting example
    auto text = std::make_unique<PlainText>("Hello World");
    std::cout << text->render() << std::endl;

    text = std::make_unique<BoldDecorator>(std::move(text));
    std::cout << text->render() << std::endl;

    text = std::make_unique<ItalicDecorator>(std::move(text));
    std::cout << text->render() << std::endl;

    text = std::make_unique<UnderlineDecorator>(std::move(text));
    std::cout << text->render() << std::endl;

    std::cout << "\n---\n\n";

    // Data stream example - combining compression and encryption
    auto source = std::make_unique<FileDataSource>("data.txt");
    source = std::make_unique<CompressionDecorator>(std::move(source));
    source = std::make_unique<EncryptionDecorator>(std::move(source));

    source->writeData("Sensitive information");
    std::cout << "\nReading back:" << std::endl;
    std::string data = source->readData();
    std::cout << "Final data: " << data << std::endl;

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod

# Component
class Coffee(ABC):
    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_cost(self) -> float:
        pass

# Concrete Component
class SimpleCoffee(Coffee):
    def get_description(self) -> str:
        return "Simple Coffee"

    def get_cost(self) -> float:
        return 2.0

# Base Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

# Concrete Decorators
class MilkDecorator(CoffeeDecorator):
    def get_description(self) -> str:
        return self._coffee.get_description() + ", Milk"

    def get_cost(self) -> float:
        return self._coffee.get_cost() + 0.5

class SugarDecorator(CoffeeDecorator):
    def get_description(self) -> str:
        return self._coffee.get_description() + ", Sugar"

    def get_cost(self) -> float:
        return self._coffee.get_cost() + 0.2

class WhippedCreamDecorator(CoffeeDecorator):
    def get_description(self) -> str:
        return self._coffee.get_description() + ", Whipped Cream"

    def get_cost(self) -> float:
        return self._coffee.get_cost() + 0.7

# Text formatting
class Text(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class PlainText(Text):
    def __init__(self, content: str):
        self.content = content

    def render(self) -> str:
        return self.content

class TextDecorator(Text):
    def __init__(self, text: Text):
        self._text = text

class BoldDecorator(TextDecorator):
    def render(self) -> str:
        return f"<b>{self._text.render()}</b>"

class ItalicDecorator(TextDecorator):
    def render(self) -> str:
        return f"<i>{self._text.render()}</i>"

# Usage
if __name__ == "__main__":
    coffee = SimpleCoffee()
    print(f"{coffee.get_description()} - ${coffee.get_cost()}")

    coffee = MilkDecorator(coffee)
    print(f"{coffee.get_description()} - ${coffee.get_cost()}")

    coffee = SugarDecorator(coffee)
    print(f"{coffee.get_description()} - ${coffee.get_cost()}")

    coffee = WhippedCreamDecorator(coffee)
    print(f"{coffee.get_description()} - ${coffee.get_cost()}")

    print("\n---\n")

    text = PlainText("Hello World")
    print(text.render())

    text = BoldDecorator(text)
    print(text.render())

    text = ItalicDecorator(text)
    print(text.render())
```

**Advantages**:
- More flexible than static inheritance
- Responsibilities can be added/removed at runtime
- Avoids feature-laden classes high up in the hierarchy
- Single Responsibility Principle: divide functionality into classes
- Open/Closed Principle: extend behavior without modifying existing code

**Disadvantages**:
- Can result in many small objects (complexity)
- Decorators and their component aren't identical
- Hard to remove a specific decorator from the wrapper stack

**Related Patterns**:
- **Adapter**: Changes interface; Decorator enhances responsibilities
- **Composite**: Decorator can be viewed as degenerate composite with only one component
- **Strategy**: Decorator changes object's skin; Strategy changes object's guts

---

#### Facade Pattern

**Intent**: Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.

**Problem**: A complex subsystem with many interdependent classes requires substantial knowledge to use effectively. Clients shouldn't need to know about subsystem implementation details.

**Solution**: Create a facade class that provides simple methods for client interactions with the subsystem, delegating to appropriate subsystem objects.

**When to Use**:
- You want to provide a simple interface to a complex subsystem
- There are many dependencies between clients and implementation classes
- You want to layer your subsystems
- You need to decouple subsystem from clients and other subsystems

**Real-World Examples**:
- Computer startup (facade hides CPU, memory, hard drive interactions)
- Home theater system (one button to start movie: turn on projector, sound system, DVD player, etc.)
- Online shopping checkout (facade over payment, inventory, shipping systems)
- REST API wrapping multiple microservices
- Compiler facade over lexer, parser, code generator

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>

// Subsystem classes - Complex components
class CPU {
public:
    void freeze() {
        std::cout << "CPU: Freezing processor" << std::endl;
    }

    void jump(long address) {
        std::cout << "CPU: Jumping to address " << address << std::endl;
    }

    void execute() {
        std::cout << "CPU: Executing instructions" << std::endl;
    }
};

class Memory {
public:
    void load(long position, const std::string& data) {
        std::cout << "Memory: Loading data '" << data
                  << "' at position " << position << std::endl;
    }
};

class HardDrive {
public:
    std::string read(long lba, int size) {
        std::cout << "HardDrive: Reading " << size
                  << " bytes from sector " << lba << std::endl;
        return "BOOT_DATA";
    }
};

// Facade
class ComputerFacade {
public:
    ComputerFacade()
        : cpu_(std::make_unique<CPU>()),
          memory_(std::make_unique<Memory>()),
          hardDrive_(std::make_unique<HardDrive>()) {}

    void start() {
        std::cout << "Computer starting up..." << std::endl;
        cpu_->freeze();
        memory_->load(0, hardDrive_->read(0, 1024));
        cpu_->jump(0);
        cpu_->execute();
        std::cout << "Computer started successfully!" << std::endl;
    }

private:
    std::unique_ptr<CPU> cpu_;
    std::unique_ptr<Memory> memory_;
    std::unique_ptr<HardDrive> hardDrive_;
};

// Home Theater example
class Amplifier {
public:
    void on() { std::cout << "Amplifier: ON" << std::endl; }
    void setVolume(int level) {
        std::cout << "Amplifier: Setting volume to " << level << std::endl;
    }
    void off() { std::cout << "Amplifier: OFF" << std::endl; }
};

class DvdPlayer {
public:
    void on() { std::cout << "DVD Player: ON" << std::endl; }
    void play(const std::string& movie) {
        std::cout << "DVD Player: Playing '" << movie << "'" << std::endl;
    }
    void stop() { std::cout << "DVD Player: Stopped" << std::endl; }
    void off() { std::cout << "DVD Player: OFF" << std::endl; }
};

class Projector {
public:
    void on() { std::cout << "Projector: ON" << std::endl; }
    void wideScreenMode() { std::cout << "Projector: Widescreen mode" << std::endl; }
    void off() { std::cout << "Projector: OFF" << std::endl; }
};

class TheaterLights {
public:
    void dim(int level) {
        std::cout << "Theater Lights: Dimming to " << level << "%" << std::endl;
    }
    void on() { std::cout << "Theater Lights: ON" << std::endl; }
};

class Screen {
public:
    void down() { std::cout << "Screen: Going down" << std::endl; }
    void up() { std::cout << "Screen: Going up" << std::endl; }
};

// Home Theater Facade
class HomeTheaterFacade {
public:
    HomeTheaterFacade(
        std::shared_ptr<Amplifier> amp,
        std::shared_ptr<DvdPlayer> dvd,
        std::shared_ptr<Projector> projector,
        std::shared_ptr<Screen> screen,
        std::shared_ptr<TheaterLights> lights)
        : amp_(amp), dvd_(dvd), projector_(projector),
          screen_(screen), lights_(lights) {}

    void watchMovie(const std::string& movie) {
        std::cout << "\nGet ready to watch a movie..." << std::endl;
        lights_->dim(10);
        screen_->down();
        projector_->on();
        projector_->wideScreenMode();
        amp_->on();
        amp_->setVolume(5);
        dvd_->on();
        dvd_->play(movie);
    }

    void endMovie() {
        std::cout << "\nShutting down movie theater..." << std::endl;
        dvd_->stop();
        dvd_->off();
        amp_->off();
        projector_->off();
        screen_->up();
        lights_->on();
    }

private:
    std::shared_ptr<Amplifier> amp_;
    std::shared_ptr<DvdPlayer> dvd_;
    std::shared_ptr<Projector> projector_;
    std::shared_ptr<Screen> screen_;
    std::shared_ptr<TheaterLights> lights_;
};

// Usage
int main() {
    // Computer facade example
    ComputerFacade computer;
    computer.start();

    std::cout << "\n---\n";

    // Home theater facade example
    auto amp = std::make_shared<Amplifier>();
    auto dvd = std::make_shared<DvdPlayer>();
    auto projector = std::make_shared<Projector>();
    auto screen = std::make_shared<Screen>();
    auto lights = std::make_shared<TheaterLights>();

    HomeTheaterFacade homeTheater(amp, dvd, projector, screen, lights);
    homeTheater.watchMovie("Inception");
    homeTheater.endMovie();

    return 0;
}
```

**Implementation in Python**:

```python
# Subsystem classes
class CPU:
    def freeze(self) -> None:
        print("CPU: Freezing processor")

    def jump(self, address: int) -> None:
        print(f"CPU: Jumping to address {address}")

    def execute(self) -> None:
        print("CPU: Executing instructions")

class Memory:
    def load(self, position: int, data: str) -> None:
        print(f"Memory: Loading data '{data}' at position {position}")

class HardDrive:
    def read(self, lba: int, size: int) -> str:
        print(f"HardDrive: Reading {size} bytes from sector {lba}")
        return "BOOT_DATA"

# Facade
class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()

    def start(self) -> None:
        print("Computer starting up...")
        self.cpu.freeze()
        self.memory.load(0, self.hard_drive.read(0, 1024))
        self.cpu.jump(0)
        self.cpu.execute()
        print("Computer started successfully!")

# Home Theater classes
class Amplifier:
    def on(self) -> None:
        print("Amplifier: ON")

    def set_volume(self, level: int) -> None:
        print(f"Amplifier: Setting volume to {level}")

    def off(self) -> None:
        print("Amplifier: OFF")

class DvdPlayer:
    def on(self) -> None:
        print("DVD Player: ON")

    def play(self, movie: str) -> None:
        print(f"DVD Player: Playing '{movie}'")

    def stop(self) -> None:
        print("DVD Player: Stopped")

    def off(self) -> None:
        print("DVD Player: OFF")

class Projector:
    def on(self) -> None:
        print("Projector: ON")

    def wide_screen_mode(self) -> None:
        print("Projector: Widescreen mode")

    def off(self) -> None:
        print("Projector: OFF")

class HomeTheaterFacade:
    def __init__(self, amp: Amplifier, dvd: DvdPlayer, projector: Projector):
        self.amp = amp
        self.dvd = dvd
        self.projector = projector

    def watch_movie(self, movie: str) -> None:
        print("\nGet ready to watch a movie...")
        self.projector.on()
        self.projector.wide_screen_mode()
        self.amp.on()
        self.amp.set_volume(5)
        self.dvd.on()
        self.dvd.play(movie)

    def end_movie(self) -> None:
        print("\nShutting down movie theater...")
        self.dvd.stop()
        self.dvd.off()
        self.amp.off()
        self.projector.off()

# Usage
if __name__ == "__main__":
    computer = ComputerFacade()
    computer.start()

    print("\n---\n")

    amp = Amplifier()
    dvd = DvdPlayer()
    projector = Projector()

    home_theater = HomeTheaterFacade(amp, dvd, projector)
    home_theater.watch_movie("Inception")
    home_theater.end_movie()
```

**Advantages**:
- Simplifies complex subsystems for clients
- Reduces coupling between subsystems and clients
- Promotes weak coupling between subsystems
- Provides a simple default view while allowing access to lower-level features
- Makes libraries easier to use and test

**Disadvantages**:
- Facade can become a god object coupled to all classes of an application
- May introduce additional abstraction layer
- Can limit functionality if not designed to expose all subsystem features

**Related Patterns**:
- **Abstract Factory**: Can be used with Facade to hide platform-specific classes
- **Mediator**: Similar to Facade but abstracts communication between subsystem objects (bidirectional vs. unidirectional)
- **Singleton**: Facade objects are often singletons

---

#### Flyweight Pattern

**Intent**: Use sharing to support large numbers of fine-grained objects efficiently by sharing common state.

**Problem**: Creating a large number of similar objects consumes too much memory. Many objects share common data that doesn't need to be duplicated.

**Solution**: Separate intrinsic state (shared) from extrinsic state (unique). Store intrinsic state in flyweight objects that can be shared; pass extrinsic state to flyweight methods as parameters.

**When to Use**:
- Application uses large numbers of objects
- Storage costs are high because of the quantity of objects
- Most object state can be made extrinsic
- Many groups of objects may be replaced by relatively few shared objects
- Application doesn't depend on object identity

**Real-World Examples**:
- Text editors (character objects sharing font data)
- Game development (particles, trees, grass instances)
- String interning in programming languages
- Connection pools
- Thread pools

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Flyweight - Shared character data
class CharacterStyle {
public:
    CharacterStyle(const std::string& font, int size, const std::string& color)
        : font_(font), size_(size), color_(color) {
        std::cout << "Creating CharacterStyle: " << font << " " << size
                  << "pt " << color << std::endl;
    }

    void display(char symbol, int row, int column) const {
        std::cout << "Character '" << symbol << "' at (" << row << "," << column
                  << ") - Font: " << font_ << " " << size_ << "pt " << color_
                  << std::endl;
    }

    std::string getKey() const {
        return font_ + "_" + std::to_string(size_) + "_" + color_;
    }

private:
    std::string font_;  // Intrinsic state (shared)
    int size_;          // Intrinsic state (shared)
    std::string color_; // Intrinsic state (shared)
};

// Flyweight Factory
class CharacterStyleFactory {
public:
    std::shared_ptr<CharacterStyle> getStyle(
        const std::string& font, int size, const std::string& color) {

        std::string key = font + "_" + std::to_string(size) + "_" + color;

        auto it = styles_.find(key);
        if (it != styles_.end()) {
            std::cout << "Reusing existing style: " << key << std::endl;
            return it->second;
        }

        auto newStyle = std::make_shared<CharacterStyle>(font, size, color);
        styles_[key] = newStyle;
        return newStyle;
    }

    size_t getStyleCount() const {
        return styles_.size();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<CharacterStyle>> styles_;
};

// Context - Character with position (extrinsic state)
class Character {
public:
    Character(char symbol, int row, int column,
              std::shared_ptr<CharacterStyle> style)
        : symbol_(symbol), row_(row), column_(column), style_(style) {}

    void display() const {
        style_->display(symbol_, row_, column_);
    }

private:
    char symbol_;                           // Extrinsic state (unique)
    int row_, column_;                      // Extrinsic state (unique)
    std::shared_ptr<CharacterStyle> style_; // Reference to flyweight
};

// Game example - Trees in a forest
class TreeType {
public:
    TreeType(const std::string& name, const std::string& color, const std::string& texture)
        : name_(name), color_(color), texture_(texture) {
        std::cout << "Creating tree type: " << name << std::endl;
    }

    void draw(int x, int y) const {
        std::cout << name_ << " tree (color: " << color_ << ", texture: " << texture_
                  << ") at (" << x << "," << y << ")" << std::endl;
    }

private:
    std::string name_;    // Intrinsic
    std::string color_;   // Intrinsic
    std::string texture_; // Intrinsic
};

class TreeFactory {
public:
    std::shared_ptr<TreeType> getTreeType(
        const std::string& name, const std::string& color, const std::string& texture) {

        std::string key = name + "_" + color + "_" + texture;

        auto it = treeTypes_.find(key);
        if (it != treeTypes_.end()) {
            return it->second;
        }

        auto newType = std::make_shared<TreeType>(name, color, texture);
        treeTypes_[key] = newType;
        return newType;
    }

    size_t getTypeCount() const {
        return treeTypes_.size();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<TreeType>> treeTypes_;
};

class Tree {
public:
    Tree(int x, int y, std::shared_ptr<TreeType> type)
        : x_(x), y_(y), type_(type) {}

    void draw() const {
        type_->draw(x_, y_);
    }

private:
    int x_, y_;                      // Extrinsic state (unique per tree)
    std::shared_ptr<TreeType> type_; // Intrinsic state (shared)
};

class Forest {
public:
    void plantTree(int x, int y, const std::string& name,
                   const std::string& color, const std::string& texture) {
        auto type = treeFactory_.getTreeType(name, color, texture);
        trees_.push_back(std::make_unique<Tree>(x, y, type));
    }

    void draw() const {
        for (const auto& tree : trees_) {
            tree->draw();
        }
        std::cout << "\nTotal trees: " << trees_.size()
                  << ", Unique tree types: " << treeFactory_.getTypeCount()
                  << std::endl;
    }

private:
    TreeFactory treeFactory_;
    std::vector<std::unique_ptr<Tree>> trees_;
};

// Usage
int main() {
    // Text editor example
    CharacterStyleFactory styleFactory;

    auto arial12Black = styleFactory.getStyle("Arial", 12, "Black");
    auto arial12Red = styleFactory.getStyle("Arial", 12, "Red");
    auto arial14Black = styleFactory.getStyle("Arial", 14, "Black");
    auto arial12Black2 = styleFactory.getStyle("Arial", 12, "Black"); // Reuses

    std::vector<Character> document;
    document.emplace_back('H', 0, 0, arial14Black);
    document.emplace_back('e', 0, 1, arial12Black);
    document.emplace_back('l', 0, 2, arial12Black);
    document.emplace_back('l', 0, 3, arial12Red);
    document.emplace_back('o', 0, 4, arial12Black);

    std::cout << "\nDocument with " << document.size() << " characters:" << std::endl;
    for (const auto& ch : document) {
        ch.display();
    }

    std::cout << "\nTotal unique styles created: "
              << styleFactory.getStyleCount() << std::endl;

    std::cout << "\n---\n\n";

    // Forest example
    Forest forest;
    forest.plantTree(10, 20, "Oak", "Green", "OakTexture");
    forest.plantTree(50, 30, "Pine", "DarkGreen", "PineTexture");
    forest.plantTree(80, 40, "Oak", "Green", "OakTexture");  // Reuses Oak type
    forest.plantTree(120, 50, "Pine", "DarkGreen", "PineTexture"); // Reuses Pine
    forest.plantTree(150, 60, "Birch", "White", "BirchTexture");
    forest.plantTree(200, 70, "Oak", "Green", "OakTexture");  // Reuses Oak type

    std::cout << "\nDrawing forest:" << std::endl;
    forest.draw();

    return 0;
}
```

**Implementation in Python**:

```python
from typing import Dict

# Flyweight
class CharacterStyle:
    def __init__(self, font: str, size: int, color: str):
        self.font = font
        self.size = size
        self.color = color
        print(f"Creating CharacterStyle: {font} {size}pt {color}")

    def display(self, symbol: str, row: int, column: int) -> None:
        print(f"Character '{symbol}' at ({row},{column}) - "
              f"Font: {self.font} {self.size}pt {self.color}")

# Flyweight Factory
class CharacterStyleFactory:
    def __init__(self):
        self._styles: Dict[str, CharacterStyle] = {}

    def get_style(self, font: str, size: int, color: str) -> CharacterStyle:
        key = f"{font}_{size}_{color}"

        if key in self._styles:
            print(f"Reusing existing style: {key}")
            return self._styles[key]

        new_style = CharacterStyle(font, size, color)
        self._styles[key] = new_style
        return new_style

    def get_style_count(self) -> int:
        return len(self._styles)

# Context
class Character:
    def __init__(self, symbol: str, row: int, column: int, style: CharacterStyle):
        self.symbol = symbol      # Extrinsic
        self.row = row           # Extrinsic
        self.column = column     # Extrinsic
        self.style = style       # Intrinsic (shared)

    def display(self) -> None:
        self.style.display(self.symbol, self.row, self.column)

# Tree example
class TreeType:
    def __init__(self, name: str, color: str, texture: str):
        self.name = name
        self.color = color
        self.texture = texture
        print(f"Creating tree type: {name}")

    def draw(self, x: int, y: int) -> None:
        print(f"{self.name} tree (color: {self.color}, texture: {self.texture}) "
              f"at ({x},{y})")

class TreeFactory:
    def __init__(self):
        self._tree_types: Dict[str, TreeType] = {}

    def get_tree_type(self, name: str, color: str, texture: str) -> TreeType:
        key = f"{name}_{color}_{texture}"

        if key in self._tree_types:
            return self._tree_types[key]

        new_type = TreeType(name, color, texture)
        self._tree_types[key] = new_type
        return new_type

    def get_type_count(self) -> int:
        return len(self._tree_types)

class Tree:
    def __init__(self, x: int, y: int, tree_type: TreeType):
        self.x = x               # Extrinsic
        self.y = y               # Extrinsic
        self.type = tree_type    # Intrinsic (shared)

    def draw(self) -> None:
        self.type.draw(self.x, self.y)

class Forest:
    def __init__(self):
        self.tree_factory = TreeFactory()
        self.trees = []

    def plant_tree(self, x: int, y: int, name: str, color: str, texture: str) -> None:
        tree_type = self.tree_factory.get_tree_type(name, color, texture)
        self.trees.append(Tree(x, y, tree_type))

    def draw(self) -> None:
        for tree in self.trees:
            tree.draw()
        print(f"\nTotal trees: {len(self.trees)}, "
              f"Unique tree types: {self.tree_factory.get_type_count()}")

# Usage
if __name__ == "__main__":
    # Text editor
    factory = CharacterStyleFactory()

    arial_12_black = factory.get_style("Arial", 12, "Black")
    arial_12_red = factory.get_style("Arial", 12, "Red")
    arial_12_black_2 = factory.get_style("Arial", 12, "Black")  # Reuses

    document = [
        Character('H', 0, 0, arial_12_black),
        Character('e', 0, 1, arial_12_black),
        Character('l', 0, 2, arial_12_red),
        Character('l', 0, 3, arial_12_black),
        Character('o', 0, 4, arial_12_black),
    ]

    print(f"\nDocument with {len(document)} characters:")
    for ch in document:
        ch.display()

    print(f"\nTotal unique styles: {factory.get_style_count()}")

    print("\n---\n")

    # Forest
    forest = Forest()
    forest.plant_tree(10, 20, "Oak", "Green", "OakTexture")
    forest.plant_tree(50, 30, "Pine", "DarkGreen", "PineTexture")
    forest.plant_tree(80, 40, "Oak", "Green", "OakTexture")  # Reuses
    forest.plant_tree(120, 50, "Birch", "White", "BirchTexture")

    print("\nDrawing forest:")
    forest.draw()
```

**Advantages**:
- Reduces memory consumption significantly
- Reduces total number of objects
- Can improve performance by reducing memory allocation overhead
- Centralizes state management for shared data

**Disadvantages**:
- More complex code (intrinsic vs. extrinsic state separation)
- Runtime costs for computing extrinsic state
- Can make design less intuitive

**Related Patterns**:
- **Composite**: Often combined with Flyweight to implement shared leaf nodes
- **State and Strategy**: Can be implemented as flyweights
- **Singleton**: Flyweight factories are often singletons

---

#### Proxy Pattern

**Intent**: Provide a surrogate or placeholder for another object to control access to it.

**Problem**: You need to control access to an object for various reasons: expensive creation, remote access, access control, logging, lazy initialization, etc.

**Solution**: Create a proxy object with the same interface as the real object. The proxy controls access to the real object and can perform additional operations before/after forwarding requests.

**When to Use**:
- **Virtual Proxy**: Lazy initialization of expensive objects
- **Remote Proxy**: Local representative for remote objects
- **Protection Proxy**: Access control based on permissions
- **Smart Reference**: Additional actions when object is accessed (reference counting, locking, lazy loading)
- **Caching Proxy**: Cache results of expensive operations
- **Logging Proxy**: Log requests before forwarding

**Real-World Examples**:
- Image proxies in web browsers (placeholder until loaded)
- Network proxies and VPNs
- Smart pointers in C++
- Lazy-loaded ORM entities
- Access control in security systems

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

// Subject interface
class Image {
public:
    virtual ~Image() = default;
    virtual void display() = 0;
    virtual std::string getName() const = 0;
};

// Real Subject - Expensive object
class RealImage : public Image {
public:
    RealImage(const std::string& filename)
        : filename_(filename) {
        loadFromDisk();
    }

    void display() override {
        std::cout << "Displaying " << filename_ << std::endl;
    }

    std::string getName() const override {
        return filename_;
    }

private:
    void loadFromDisk() {
        std::cout << "Loading " << filename_ << " from disk (expensive operation)..."
                  << std::endl;
    }

    std::string filename_;
};

// Virtual Proxy - Lazy initialization
class ImageProxy : public Image {
public:
    ImageProxy(const std::string& filename)
        : filename_(filename), realImage_(nullptr) {}

    void display() override {
        if (!realImage_) {
            std::cout << "Proxy: First access, loading real image..." << std::endl;
            realImage_ = std::make_unique<RealImage>(filename_);
        }
        realImage_->display();
    }

    std::string getName() const override {
        return filename_;
    }

private:
    std::string filename_;
    std::unique_ptr<RealImage> realImage_;
};

// Protection Proxy example
class Document {
public:
    virtual ~Document() = default;
    virtual void displayContent() = 0;
    virtual void editContent(const std::string& newContent) = 0;
};

class RealDocument : public Document {
public:
    RealDocument(const std::string& content)
        : content_(content) {}

    void displayContent() override {
        std::cout << "Document content: " << content_ << std::endl;
    }

    void editContent(const std::string& newContent) override {
        content_ = newContent;
        std::cout << "Document updated to: " << content_ << std::endl;
    }

private:
    std::string content_;
};

class DocumentProxy : public Document {
public:
    DocumentProxy(std::unique_ptr<RealDocument> doc, const std::string& userRole)
        : document_(std::move(doc)), userRole_(userRole) {}

    void displayContent() override {
        std::cout << "Proxy: Checking read permissions for " << userRole_ << "..." << std::endl;
        document_->displayContent();
    }

    void editContent(const std::string& newContent) override {
        if (userRole_ == "admin" || userRole_ == "editor") {
            std::cout << "Proxy: " << userRole_ << " has write permission" << std::endl;
            document_->editContent(newContent);
        } else {
            std::cout << "Proxy: Access denied! " << userRole_
                      << " doesn't have write permission" << std::endl;
        }
    }

private:
    std::unique_ptr<RealDocument> document_;
    std::string userRole_;
};

// Caching Proxy example
class DataService {
public:
    virtual ~DataService() = default;
    virtual std::string getData(const std::string& key) = 0;
};

class RealDataService : public DataService {
public:
    std::string getData(const std::string& key) override {
        std::cout << "RealDataService: Fetching '" << key
                  << "' from database (expensive)..." << std::endl;
        return "Data for " + key;
    }
};

class CachingDataServiceProxy : public DataService {
public:
    CachingDataServiceProxy(std::unique_ptr<RealDataService> service)
        : service_(std::move(service)) {}

    std::string getData(const std::string& key) override {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            std::cout << "CachingProxy: Returning cached data for '" << key << "'"
                      << std::endl;
            return it->second;
        }

        std::cout << "CachingProxy: Cache miss, fetching from real service..."
                  << std::endl;
        std::string data = service_->getData(key);
        cache_[key] = data;
        return data;
    }

private:
    std::unique_ptr<RealDataService> service_;
    std::unordered_map<std::string, std::string> cache_;
};

// Usage
int main() {
    // Virtual Proxy - Lazy loading
    std::cout << "=== Virtual Proxy Example ===" << std::endl;
    auto image1 = std::make_unique<ImageProxy>("photo1.jpg");
    auto image2 = std::make_unique<ImageProxy>("photo2.jpg");

    std::cout << "\nImages created (not loaded yet)\n" << std::endl;

    image1->display();  // Loads and displays
    image1->display();  // Just displays (already loaded)

    std::cout << "\n---\n\n";

    // Protection Proxy
    std::cout << "=== Protection Proxy Example ===" << std::endl;
    auto adminDoc = std::make_unique<DocumentProxy>(
        std::make_unique<RealDocument>("Secret Document"),
        "admin"
    );

    auto viewerDoc = std::make_unique<DocumentProxy>(
        std::make_unique<RealDocument>("Public Document"),
        "viewer"
    );

    adminDoc->displayContent();
    adminDoc->editContent("Updated Secret Document");

    std::cout << std::endl;

    viewerDoc->displayContent();
    viewerDoc->editContent("Trying to update");  // Denied

    std::cout << "\n---\n\n";

    // Caching Proxy
    std::cout << "=== Caching Proxy Example ===" << std::endl;
    auto dataService = std::make_unique<CachingDataServiceProxy>(
        std::make_unique<RealDataService>()
    );

    std::cout << dataService->getData("user:123") << std::endl;
    std::cout << std::endl;
    std::cout << dataService->getData("user:123") << std::endl;  // From cache
    std::cout << std::endl;
    std::cout << dataService->getData("user:456") << std::endl;

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional

# Virtual Proxy
class Image(ABC):
    @abstractmethod
    def display(self) -> None:
        pass

class RealImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        print(f"Loading {self.filename} from disk (expensive operation)...")

    def display(self) -> None:
        print(f"Displaying {self.filename}")

class ImageProxy(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._real_image: Optional[RealImage] = None

    def display(self) -> None:
        if self._real_image is None:
            print("Proxy: First access, loading real image...")
            self._real_image = RealImage(self.filename)
        self._real_image.display()

# Protection Proxy
class Document(ABC):
    @abstractmethod
    def display_content(self) -> None:
        pass

    @abstractmethod
    def edit_content(self, new_content: str) -> None:
        pass

class RealDocument(Document):
    def __init__(self, content: str):
        self.content = content

    def display_content(self) -> None:
        print(f"Document content: {self.content}")

    def edit_content(self, new_content: str) -> None:
        self.content = new_content
        print(f"Document updated to: {self.content}")

class DocumentProxy(Document):
    def __init__(self, document: RealDocument, user_role: str):
        self.document = document
        self.user_role = user_role

    def display_content(self) -> None:
        print(f"Proxy: Checking read permissions for {self.user_role}...")
        self.document.display_content()

    def edit_content(self, new_content: str) -> None:
        if self.user_role in ["admin", "editor"]:
            print(f"Proxy: {self.user_role} has write permission")
            self.document.edit_content(new_content)
        else:
            print(f"Proxy: Access denied! {self.user_role} doesn't have write permission")

# Caching Proxy
class DataService(ABC):
    @abstractmethod
    def get_data(self, key: str) -> str:
        pass

class RealDataService(DataService):
    def get_data(self, key: str) -> str:
        print(f"RealDataService: Fetching '{key}' from database (expensive)...")
        return f"Data for {key}"

class CachingDataServiceProxy(DataService):
    def __init__(self, service: RealDataService):
        self.service = service
        self.cache: Dict[str, str] = {}

    def get_data(self, key: str) -> str:
        if key in self.cache:
            print(f"CachingProxy: Returning cached data for '{key}'")
            return self.cache[key]

        print("CachingProxy: Cache miss, fetching from real service...")
        data = self.service.get_data(key)
        self.cache[key] = data
        return data

# Usage
if __name__ == "__main__":
    # Virtual Proxy
    print("=== Virtual Proxy Example ===")
    image1 = ImageProxy("photo1.jpg")
    image2 = ImageProxy("photo2.jpg")

    print("\nImages created (not loaded yet)\n")

    image1.display()  # Loads and displays
    image1.display()  # Just displays

    print("\n---\n")

    # Protection Proxy
    print("=== Protection Proxy Example ===")
    admin_doc = DocumentProxy(RealDocument("Secret Document"), "admin")
    viewer_doc = DocumentProxy(RealDocument("Public Document"), "viewer")

    admin_doc.display_content()
    admin_doc.edit_content("Updated Secret Document")

    print()

    viewer_doc.display_content()
    viewer_doc.edit_content("Trying to update")  # Denied

    print("\n---\n")

    # Caching Proxy
    print("=== Caching Proxy Example ===")
    data_service = CachingDataServiceProxy(RealDataService())

    print(data_service.get_data("user:123"))
    print()
    print(data_service.get_data("user:123"))  # From cache
    print()
    print(data_service.get_data("user:456"))
```

**Advantages**:
- Controls access to the real object
- Can add functionality transparently (logging, caching, lazy loading)
- Open/Closed Principle: introduce new proxies without changing the service
- Can manage lifecycle of service object

**Disadvantages**:
- Code may become more complicated (many new classes)
- Response from service might be delayed
- Additional layer of indirection

**Related Patterns**:
- **Adapter**: Provides different interface; Proxy provides same interface
- **Decorator**: Similar structure but different intent (enhancement vs. access control)
- **Facade**: Provides simplified interface; Proxy provides same interface

---

### Behavioral Patterns

#### Observer Pattern

**Intent**: Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**Problem**: You need to maintain consistency between related objects without making them tightly coupled. When one object changes, an unknown number of other objects need to be updated.

**Solution**: Define Subject (publisher) and Observer (subscriber) interfaces. Subjects maintain a list of observers and notify them automatically of state changes. Observers register/unregister themselves with subjects.

**When to Use**:
- Change to one object requires changing others, and you don't know how many
- Object should notify other objects without knowing who they are
- Need loosely coupled event handling system
- Implementing distributed event handling (MVC, pub-sub systems)

**Real-World Examples**:
- Event listeners in GUI frameworks
- Model-View-Controller (MVC) architecture
- Social media notifications (followers notified of new posts)
- Stock market tickers
- RSS feeds
- Reactive programming (RxJS, ReactiveX)

**Implementation in C++**:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Observer interface
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
    virtual std::string getName() const = 0;
};

// Subject (Observable) interface
class Subject {
public:
    virtual ~Subject() = default;
    virtual void attach(std::shared_ptr<Observer> observer) = 0;
    virtual void detach(std::shared_ptr<Observer> observer) = 0;
    virtual void notify(const std::string& message) = 0;
};

// Concrete Subject - News Agency
class NewsAgency : public Subject {
public:
    void attach(std::shared_ptr<Observer> observer) override {
        observers_.push_back(observer);
        std::cout << "NewsAgency: Attached observer " << observer->getName() << std::endl;
    }

    void detach(std::shared_ptr<Observer> observer) override {
        auto it = std::find(observers_.begin(), observers_.end(), observer);
        if (it != observers_.end()) {
            std::cout << "NewsAgency: Detached observer " << observer->getName() << std::endl;
            observers_.erase(it);
        }
    }

    void notify(const std::string& message) override {
        std::cout << "\nNewsAgency: Broadcasting news..." << std::endl;
        for (auto& observer : observers_) {
            if (auto obs = observer.lock()) {
                obs->update(message);
            }
        }
    }

    void publishNews(const std::string& news) {
        news_ = news;
        notify(news_);
    }

private:
    std::string news_;
    std::vector<std::weak_ptr<Observer>> observers_;
};

// Concrete Observers
class NewsChannel : public Observer {
public:
    NewsChannel(const std::string& name) : name_(name) {}

    void update(const std::string& message) override {
        std::cout << "NewsChannel [" << name_ << "]: Received news - " << message << std::endl;
    }

    std::string getName() const override {
        return name_;
    }

private:
    std::string name_;
};

class Newspaper : public Observer {
public:
    Newspaper(const std::string& name) : name_(name) {}

    void update(const std::string& message) override {
        std::cout << "Newspaper [" << name_ << "]: Printing news - " << message << std::endl;
        headlines_.push_back(message);
    }

    std::string getName() const override {
        return name_;
    }

    void printArchive() const {
        std::cout << "\n" << name_ << " Archive:" << std::endl;
        for (size_t i = 0; i < headlines_.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << headlines_[i] << std::endl;
        }
    }

private:
    std::string name_;
    std::vector<std::string> headlines_;
};

// Weather Station example
class WeatherStation {
public:
    void setMeasurements(float temperature, float humidity, float pressure) {
        temperature_ = temperature;
        humidity_ = humidity;
        pressure_ = pressure;
        measurementsChanged();
    }

    void attach(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);
    }

    void detach(std::shared_ptr<Observer> observer) {
        auto it = std::find(observers_.begin(), observers_.end(), observer);
        if (it != observers_.end()) {
            observers_.erase(it);
        }
    }

private:
    void measurementsChanged() {
        std::string data = "Temp: " + std::to_string(temperature_) + "°C, " +
                          "Humidity: " + std::to_string(humidity_) + "%, " +
                          "Pressure: " + std::to_string(pressure_) + " hPa";

        for (auto& observer : observers_) {
            if (auto obs = observer.lock()) {
                obs->update(data);
            }
        }
    }

    float temperature_ = 0.0f;
    float humidity_ = 0.0f;
    float pressure_ = 0.0f;
    std::vector<std::weak_ptr<Observer>> observers_;
};

class WeatherDisplay : public Observer {
public:
    WeatherDisplay(const std::string& name) : name_(name) {}

    void update(const std::string& message) override {
        std::cout << "Display [" << name_ << "]: " << message << std::endl;
    }

    std::string getName() const override {
        return name_;
    }

private:
    std::string name_;
};

// Usage
int main() {
    // News agency example
    auto newsAgency = std::make_unique<NewsAgency>();

    auto cnn = std::make_shared<NewsChannel>("CNN");
    auto bbc = std::make_shared<NewsChannel>("BBC");
    auto nyt = std::make_shared<Newspaper>("New York Times");

    newsAgency->attach(cnn);
    newsAgency->attach(bbc);
    newsAgency->attach(nyt);

    newsAgency->publishNews("Breaking: Major tech announcement!");

    std::cout << "\nDetaching CNN..." << std::endl;
    newsAgency->detach(cnn);

    newsAgency->publishNews("Update: Market reaches new high");

    nyt->printArchive();

    std::cout << "\n---\n\n";

    // Weather station example
    WeatherStation station;

    auto homeDisplay = std::make_shared<WeatherDisplay>("Home");
    auto officeDisplay = std::make_shared<WeatherDisplay>("Office");
    auto mobileDisplay = std::make_shared<WeatherDisplay>("Mobile");

    station.attach(homeDisplay);
    station.attach(officeDisplay);
    station.attach(mobileDisplay);

    std::cout << "Weather update 1:" << std::endl;
    station.setMeasurements(25.5f, 65.0f, 1013.2f);

    std::cout << "\nWeather update 2:" << std::endl;
    station.setMeasurements(27.0f, 70.0f, 1012.8f);

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import List
from weakref import WeakSet

# Observer interface
class Observer(ABC):
    @abstractmethod
    def update(self, message: str) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

# Subject interface
class Subject(ABC):
    @abstractmethod
    def attach(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def notify(self, message: str) -> None:
        pass

# Concrete Subject
class NewsAgency(Subject):
    def __init__(self):
        self._observers: WeakSet[Observer] = WeakSet()
        self._news: str = ""

    def attach(self, observer: Observer) -> None:
        self._observers.add(observer)
        print(f"NewsAgency: Attached observer {observer.get_name()}")

    def detach(self, observer: Observer) -> None:
        self._observers.discard(observer)
        print(f"NewsAgency: Detached observer {observer.get_name()}")

    def notify(self, message: str) -> None:
        print("\nNewsAgency: Broadcasting news...")
        for observer in self._observers:
            observer.update(message)

    def publish_news(self, news: str) -> None:
        self._news = news
        self.notify(self._news)

# Concrete Observers
class NewsChannel(Observer):
    def __init__(self, name: str):
        self._name = name

    def update(self, message: str) -> None:
        print(f"NewsChannel [{self._name}]: Received news - {message}")

    def get_name(self) -> str:
        return self._name

class Newspaper(Observer):
    def __init__(self, name: str):
        self._name = name
        self._headlines: List[str] = []

    def update(self, message: str) -> None:
        print(f"Newspaper [{self._name}]: Printing news - {message}")
        self._headlines.append(message)

    def get_name(self) -> str:
        return self._name

    def print_archive(self) -> None:
        print(f"\n{self._name} Archive:")
        for i, headline in enumerate(self._headlines, 1):
            print(f"  {i}. {headline}")

# Weather Station example
class WeatherStation:
    def __init__(self):
        self._observers: WeakSet[Observer] = WeakSet()
        self._temperature: float = 0.0
        self._humidity: float = 0.0
        self._pressure: float = 0.0

    def attach(self, observer: Observer) -> None:
        self._observers.add(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.discard(observer)

    def set_measurements(self, temperature: float, humidity: float, pressure: float) -> None:
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self._measurements_changed()

    def _measurements_changed(self) -> None:
        data = f"Temp: {self._temperature}°C, Humidity: {self._humidity}%, Pressure: {self._pressure} hPa"
        for observer in self._observers:
            observer.update(data)

class WeatherDisplay(Observer):
    def __init__(self, name: str):
        self._name = name

    def update(self, message: str) -> None:
        print(f"Display [{self._name}]: {message}")

    def get_name(self) -> str:
        return self._name

# Usage
if __name__ == "__main__":
    # News agency example
    news_agency = NewsAgency()

    cnn = NewsChannel("CNN")
    bbc = NewsChannel("BBC")
    nyt = Newspaper("New York Times")

    news_agency.attach(cnn)
    news_agency.attach(bbc)
    news_agency.attach(nyt)

    news_agency.publish_news("Breaking: Major tech announcement!")

    print("\nDetaching CNN...")
    news_agency.detach(cnn)

    news_agency.publish_news("Update: Market reaches new high")

    nyt.print_archive()

    print("\n---\n")

    # Weather station
    station = WeatherStation()

    home_display = WeatherDisplay("Home")
    office_display = WeatherDisplay("Office")

    station.attach(home_display)
    station.attach(office_display)

    print("Weather update 1:")
    station.set_measurements(25.5, 65.0, 1013.2)

    print("\nWeather update 2:")
    station.set_measurements(27.0, 70.0, 1012.8)
```

**Advantages**:
- Loose coupling between subject and observers
- Open/Closed Principle: add new observers without modifying subject
- Establishes relationships at runtime
- Supports broadcast communication

**Disadvantages**:
- Observers notified in random order
- Memory leaks if observers aren't properly detached
- Can cause unexpected updates if dependencies are complex
- Performance issues with many observers

**Related Patterns**:
- **Mediator**: Both promote loose coupling; Mediator uses centralized communication, Observer uses distributed
- **Singleton**: Subject often implemented as singleton

---

#### Strategy Pattern

**Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Problem**: You have multiple related classes that differ only in their behavior. You need to select an algorithm at runtime, or you have many conditional statements choosing between different variants of the same algorithm.

**Solution**: Extract algorithms into separate classes (strategies) with a common interface. Context class delegates work to a strategy object instead of implementing multiple versions of the algorithm.

**When to Use**:
- You have many related classes differing only in behavior
- You need different variants of an algorithm
- Algorithm uses data clients shouldn't know about
- Class has massive conditional statements selecting different behaviors

**Real-World Examples**:
- Payment processing (credit card, PayPal, cryptocurrency)
- Sorting algorithms (quicksort, mergesort, bubblesort)
- Compression algorithms (ZIP, RAR, TAR)
- Route planning (shortest, fastest, scenic)
- Validation strategies

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// Strategy interface
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
    virtual std::string getName() const = 0;
};

// Concrete Strategies
class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        std::cout << "Sorting using Bubble Sort" << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }

    std::string getName() const override {
        return "Bubble Sort";
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        std::cout << "Sorting using Quick Sort" << std::endl;
        std::sort(data.begin(), data.end());
    }

    std::string getName() const override {
        return "Quick Sort";
    }
};

class MergeSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        std::cout << "Sorting using Merge Sort" << std::endl;
        std::stable_sort(data.begin(), data.end());
    }

    std::string getName() const override {
        return "Merge Sort";
    }
};

// Context
class DataSorter {
public:
    void setStrategy(std::unique_ptr<SortStrategy> strategy) {
        strategy_ = std::move(strategy);
    }

    void sort(std::vector<int>& data) {
        if (strategy_) {
            strategy_->sort(data);
        } else {
            std::cout << "No sorting strategy set!" << std::endl;
        }
    }

private:
    std::unique_ptr<SortStrategy> strategy_;
};

// Payment example
class PaymentStrategy {
public:
    virtual ~PaymentStrategy() = default;
    virtual void pay(double amount) = 0;
};

class CreditCardPayment : public PaymentStrategy {
public:
    CreditCardPayment(const std::string& number, const std::string& cvv)
        : cardNumber_(number), cvv_(cvv) {}

    void pay(double amount) override {
        std::cout << "Paid $" << amount << " using Credit Card ending in "
                  << cardNumber_.substr(cardNumber_.length() - 4) << std::endl;
    }

private:
    std::string cardNumber_;
    std::string cvv_;
};

class PayPalPayment : public PaymentStrategy {
public:
    PayPalPayment(const std::string& email) : email_(email) {}

    void pay(double amount) override {
        std::cout << "Paid $" << amount << " using PayPal account " << email_ << std::endl;
    }

private:
    std::string email_;
};

class ShoppingCart {
public:
    void setPaymentStrategy(std::unique_ptr<PaymentStrategy> strategy) {
        paymentStrategy_ = std::move(strategy);
    }

    void checkout(double amount) {
        if (paymentStrategy_) {
            paymentStrategy_->pay(amount);
        }
    }

private:
    std::unique_ptr<PaymentStrategy> paymentStrategy_;
};

// Usage
int main() {
    // Sorting example
    std::vector<int> data = {64, 34, 25, 12, 22, 11, 90};

    DataSorter sorter;

    sorter.setStrategy(std::make_unique<BubbleSort>());
    auto data1 = data;
    sorter.sort(data1);

    sorter.setStrategy(std::make_unique<QuickSort>());
    auto data2 = data;
    sorter.sort(data2);

    std::cout << "\n---\n\n";

    // Payment example
    ShoppingCart cart;

    cart.setPaymentStrategy(std::make_unique<CreditCardPayment>("1234567890123456", "123"));
    cart.checkout(100.0);

    cart.setPaymentStrategy(std::make_unique<PayPalPayment>("user@example.com"));
    cart.checkout(50.0);

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import List

# Strategy interface
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

# Concrete Strategies
class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> None:
        print("Sorting using Bubble Sort")
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]

    def get_name(self) -> str:
        return "Bubble Sort"

class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> None:
        print("Sorting using Quick Sort")
        data.sort()

    def get_name(self) -> str:
        return "Quick Sort"

# Context
class DataSorter:
    def __init__(self, strategy: SortStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def sort(self, data: List[int]) -> None:
        if self._strategy:
            self._strategy.sort(data)
        else:
            print("No sorting strategy set!")

# Payment example
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> None:
        pass

class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str, cvv: str):
        self.card_number = card_number
        self.cvv = cvv

    def pay(self, amount: float) -> None:
        print(f"Paid ${amount} using Credit Card ending in {self.card_number[-4:]}")

class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email

    def pay(self, amount: float) -> None:
        print(f"Paid ${amount} using PayPal account {self.email}")

class ShoppingCart:
    def __init__(self):
        self._payment_strategy: PaymentStrategy = None

    def set_payment_strategy(self, strategy: PaymentStrategy) -> None:
        self._payment_strategy = strategy

    def checkout(self, amount: float) -> None:
        if self._payment_strategy:
            self._payment_strategy.pay(amount)

# Usage
if __name__ == "__main__":
    # Sorting
    data = [64, 34, 25, 12, 22, 11, 90]

    sorter = DataSorter()

    sorter.set_strategy(BubbleSort())
    data1 = data.copy()
    sorter.sort(data1)

    sorter.set_strategy(QuickSort())
    data2 = data.copy()
    sorter.sort(data2)

    print("\n---\n")

    # Payment
    cart = ShoppingCart()

    cart.set_payment_strategy(CreditCardPayment("1234567890123456", "123"))
    cart.checkout(100.0)

    cart.set_payment_strategy(PayPalPayment("user@example.com"))
    cart.checkout(50.0)
```

**Advantages**:
- Families of related algorithms can be reused
- Open/Closed Principle: introduce new strategies without changing context
- Runtime algorithm switching
- Isolates algorithm implementation from code that uses it
- Eliminates conditional statements

**Disadvantages**:
- Clients must be aware of different strategies
- Increases number of objects
- All strategies must expose same interface (even if some don't use all parameters)

**Related Patterns**:
- **State**: Both encapsulate behavior; Strategy focuses on algorithm, State on object state
- **Template Method**: Uses inheritance; Strategy uses composition
- **Factory Method**: Often used to create appropriate strategy

---

#### Command Pattern

**Intent**: Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

**Problem**: You need to issue requests to objects without knowing anything about the operation being requested or the receiver of the request. You want to support undo/redo, queuing, or logging of operations.

**Solution**: Create command objects that encapsulate all information needed to perform an action or trigger an event. Commands have an execute() method and optionally an undo() method.

**When to Use**:
- Parameterize objects by an action to perform
- Queue operations, schedule their execution, or execute them remotely
- Support undo/redo functionality
- Structure system around high-level operations built on primitive operations
- Support logging changes for crash recovery

**Real-World Examples**:
- GUI buttons and menu items
- Macro recording in applications
- Transaction-based systems
- Task scheduling systems
- Undo/redo in text editors
- Remote control systems

**Implementation in C++**:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stack>

// Receiver
class Light {
public:
    void on() {
        isOn_ = true;
        std::cout << "Light is ON" << std::endl;
    }

    void off() {
        isOn_ = false;
        std::cout << "Light is OFF" << std::endl;
    }

    bool isOn() const { return isOn_; }

private:
    bool isOn_ = false;
};

// Command interface
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

// Concrete Commands
class LightOnCommand : public Command {
public:
    LightOnCommand(std::shared_ptr<Light> light) : light_(light) {}

    void execute() override {
        light_->on();
    }

    void undo() override {
        light_->off();
    }

private:
    std::shared_ptr<Light> light_;
};

class LightOffCommand : public Command {
public:
    LightOffCommand(std::shared_ptr<Light> light) : light_(light) {}

    void execute() override {
        light_->off();
    }

    void undo() override {
        light_->on();
    }

private:
    std::shared_ptr<Light> light_;
};

// Text editor example
class TextEditor {
public:
    void insertText(const std::string& text) {
        content_ += text;
        std::cout << "Inserted: " << text << std::endl;
    }

    void deleteText(size_t length) {
        if (length <= content_.length()) {
            deletedText_ = content_.substr(content_.length() - length);
            content_ = content_.substr(0, content_.length() - length);
            std::cout << "Deleted: " << deletedText_ << std::endl;
        }
    }

    std::string getDeletedText() const { return deletedText_; }
    std::string getContent() const { return content_; }

    void print() const {
        std::cout << "Content: \"" << content_ << "\"" << std::endl;
    }

private:
    std::string content_;
    std::string deletedText_;
};

class InsertCommand : public Command {
public:
    InsertCommand(std::shared_ptr<TextEditor> editor, const std::string& text)
        : editor_(editor), text_(text) {}

    void execute() override {
        editor_->insertText(text_);
    }

    void undo() override {
        editor_->deleteText(text_.length());
    }

private:
    std::shared_ptr<TextEditor> editor_;
    std::string text_;
};

class DeleteCommand : public Command {
public:
    DeleteCommand(std::shared_ptr<TextEditor> editor, size_t length)
        : editor_(editor), length_(length) {}

    void execute() override {
        editor_->deleteText(length_);
        deletedText_ = editor_->getDeletedText();
    }

    void undo() override {
        editor_->insertText(deletedText_);
    }

private:
    std::shared_ptr<TextEditor> editor_;
    size_t length_;
    std::string deletedText_;
};

// Invoker
class RemoteControl {
public:
    void setCommand(std::shared_ptr<Command> command) {
        command_ = command;
    }

    void pressButton() {
        if (command_) {
            command_->execute();
            history_.push(command_);
        }
    }

    void pressUndo() {
        if (!history_.empty()) {
            auto command = history_.top();
            command->undo();
            history_.pop();
        }
    }

private:
    std::shared_ptr<Command> command_;
    std::stack<std::shared_ptr<Command>> history_;
};

// Usage
int main() {
    // Light control example
    auto livingRoomLight = std::make_shared<Light>();

    auto lightOn = std::make_shared<LightOnCommand>(livingRoomLight);
    auto lightOff = std::make_shared<LightOffCommand>(livingRoomLight);

    RemoteControl remote;

    remote.setCommand(lightOn);
    remote.pressButton();

    remote.setCommand(lightOff);
    remote.pressButton();

    remote.pressUndo();  // Undo last command

    std::cout << "\n---\n\n";

    // Text editor example
    auto editor = std::make_shared<TextEditor>();

    std::stack<std::shared_ptr<Command>> commandHistory;

    auto insertHello = std::make_shared<InsertCommand>(editor, "Hello ");
    insertHello->execute();
    commandHistory.push(insertHello);

    auto insertWorld = std::make_shared<InsertCommand>(editor, "World!");
    insertWorld->execute();
    commandHistory.push(insertWorld);

    editor->print();

    // Undo last two commands
    while (!commandHistory.empty()) {
        commandHistory.top()->undo();
        commandHistory.pop();
    }

    editor->print();

    return 0;
}
```

**Implementation in Python**:

```python
from abc import ABC, abstractmethod
from typing import List

# Receiver
class Light:
    def __init__(self):
        self._is_on = False

    def on(self) -> None:
        self._is_on = True
        print("Light is ON")

    def off(self) -> None:
        self._is_on = False
        print("Light is OFF")

    def is_on(self) -> bool:
        return self._is_on

# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def undo(self) -> None:
        pass

# Concrete Commands
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self) -> None:
        self.light.on()

    def undo(self) -> None:
        self.light.off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self) -> None:
        self.light.off()

    def undo(self) -> None:
        self.light.on()

# Text editor
class TextEditor:
    def __init__(self):
        self._content = ""
        self._deleted_text = ""

    def insert_text(self, text: str) -> None:
        self._content += text
        print(f"Inserted: {text}")

    def delete_text(self, length: int) -> None:
        if length <= len(self._content):
            self._deleted_text = self._content[-length:]
            self._content = self._content[:-length]
            print(f"Deleted: {self._deleted_text}")

    def get_deleted_text(self) -> str:
        return self._deleted_text

    def get_content(self) -> str:
        return self._content

    def print_content(self) -> None:
        print(f"Content: \"{self._content}\"")

class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text

    def execute(self) -> None:
        self.editor.insert_text(self.text)

    def undo(self) -> None:
        self.editor.delete_text(len(self.text))

# Invoker
class RemoteControl:
    def __init__(self):
        self._command: Command = None
        self._history: List[Command] = []

    def set_command(self, command: Command) -> None:
        self._command = command

    def press_button(self) -> None:
        if self._command:
            self._command.execute()
            self._history.append(self._command)

    def press_undo(self) -> None:
        if self._history:
            command = self._history.pop()
            command.undo()

# Usage
if __name__ == "__main__":
    # Light control
    living_room_light = Light()

    light_on = LightOnCommand(living_room_light)
    light_off = LightOffCommand(living_room_light)

    remote = RemoteControl()

    remote.set_command(light_on)
    remote.press_button()

    remote.set_command(light_off)
    remote.press_button()

    remote.press_undo()  # Undo

    print("\n---\n")

    # Text editor
    editor = TextEditor()
    command_history = []

    insert_hello = InsertCommand(editor, "Hello ")
    insert_hello.execute()
    command_history.append(insert_hello)

    insert_world = InsertCommand(editor, "World!")
    insert_world.execute()
    command_history.append(insert_world)

    editor.print_content()

    # Undo
    while command_history:
        command_history.pop().undo()

    editor.print_content()
```

**Advantages**:
- Decouples object that invokes operation from one that knows how to perform it
- Commands are first-class objects (can be manipulated and extended)
- Can assemble commands into composite commands (macro commands)
- Easy to add new commands (Open/Closed Principle)
- Supports undo/redo

**Disadvantages**:
- Increases number of classes for each individual command
- Can become complex with many commands

**Related Patterns**:
- **Memento**: Can be used to keep state for undo
- **Composite**: Can be used to implement macro commands
- **Prototype**: Commands that must be copied before being placed on history list

---

## Conclusion

Design patterns are invaluable tools for software developers, providing standardized solutions to recurring design problems. By understanding and applying appropriate design patterns, developers can create more flexible, reusable, and maintainable codebases.

**Key Takeaways**:

1. **Choose the Right Pattern**: Not every problem requires a design pattern. Use patterns when they genuinely simplify your design.

2. **Understand the Trade-offs**: Each pattern has advantages and disadvantages. Consider the complexity vs. flexibility trade-off.

3. **Patterns Work Together**: Many real-world applications combine multiple patterns. For example, MVC uses Observer, Strategy, and Composite patterns.

4. **Start Simple**: Don't over-engineer. Refactor towards patterns when the need becomes clear.

5. **Language Matters**: Some patterns are more natural in certain programming languages. For instance, Strategy pattern is trivial in languages with first-class functions.

**Common Pattern Categories**:

- **Creational** (Singleton, Factory Method, Abstract Factory, Builder, Prototype): Object creation mechanisms
- **Structural** (Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy): Object composition and relationships
- **Behavioral** (Observer, Strategy, Command, and others): Communication between objects

This guide has covered the most fundamental and widely-used design patterns with comprehensive examples in both C++ and Python. Each pattern includes practical implementations, real-world use cases, and guidance on when to apply them. By mastering these patterns, you'll be better equipped to design robust, maintainable, and scalable software systems.

