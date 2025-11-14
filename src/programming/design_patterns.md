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

The Adapter Pattern allows incompatible interfaces to work together. It acts as a bridge between two incompatible interfaces, enabling classes to collaborate without modifying their existing code.

**Example Scenario**: Integrating a new payment system into an existing application that expects a different payment interface.

**Example Implementation in C++:**

```cpp
#include <iostream>

// Existing interface expected by the application
class PaymentProcessor {
public:
    virtual void pay(int amount) = 0;
};

// New payment system with a different interface
class NewPaymentSystem {
public:
    void makePayment(int amount) {
        std::cout << "Processing payment of $" << amount << " through the new system." << std::endl;
    }
};

// Adapter to make NewPaymentSystem compatible with PaymentProcessor
class PaymentAdapter : public PaymentProcessor {
public:
    PaymentAdapter(NewPaymentSystem* newPaymentSystem) : newPaymentSystem_(newPaymentSystem) {}

    void pay(int amount) override {
        newPaymentSystem_->makePayment(amount);
    }

private:
    NewPaymentSystem* newPaymentSystem_;
};

int main() {
    NewPaymentSystem newSystem;
    PaymentAdapter adapter(&newSystem);
    adapter.pay(100); // Output: Processing payment of $100 through the new system.
    return 0;
}
```

**Explanation**:

- `PaymentProcessor` is the existing interface that the application expects.
- `NewPaymentSystem` is the new class with an incompatible interface (`makePayment` method).
- `PaymentAdapter` implements `PaymentProcessor` and internally uses an instance of `NewPaymentSystem` to fulfill the `pay` request.

### Behavioral Patterns

#### Observer Pattern

The Observer Pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. This pattern is commonly used in event handling systems.

**Example Scenario**: Implementing a publish-subscribe mechanism where multiple subscribers listen to updates from a publisher.

**Example Implementation in C++:**

```cpp
#include <iostream>
#include <vector>
#include <string>

// Observer interface
class Observer {
public:
    virtual void update(const std::string& message) = 0;
};

// Subject class
class Subject {
public:
    void addObserver(Observer* observer) {
        observers_.push_back(observer);
    }

    void notify(const std::string& message) {
        for(auto observer : observers_) {
            observer->update(message);
        }
    }

private:
    std::vector<Observer*> observers_;
};

// Concrete Observer
class ConcreteObserver : public Observer {
public:
    ConcreteObserver(const std::string& name) : name_(name) {}

    void update(const std::string& message) override {
        std::cout << name_ << " received message: " << message << std::endl;
    }

private:
    std::string name_;
};

int main() {
    Subject subject;

    ConcreteObserver observer1("Observer 1");
    ConcreteObserver observer2("Observer 2");

    subject.addObserver(&observer1);
    subject.addObserver(&observer2);

    subject.notify("Event occurred!");

    return 0;
}
```

**Explanation**:

- `Observer` is an interface with an `update` method.
- `Subject` maintains a list of observers and notifies them of events.
- `ConcreteObserver` implements the `Observer` interface and reacts to notifications.
- In `main`, observers are added to the subject, and when `notify` is called, all observers receive the message.

## Conclusion

Design patterns are invaluable tools for software developers, providing standardized solutions to recurring design problems. By understanding and applying appropriate design patterns, developers can create more flexible, reusable, and maintainable codebases.

