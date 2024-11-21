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

The Singleton Pattern ensures that a class has only one instance and provides a global point of access to it. This is useful when exactly one object is needed to coordinate actions across the system.

**Example Scenario**: Implementing a logger where a single instance manages all logging operations.

**Example Implementation in C++:**

```cpp
#include <iostream>
#include <mutex>

class Logger {
public:
    // Static method to get the single instance
    static Logger& getInstance() {
        static Logger instance; // Guaranteed to be destroyed and instantiated on first use
        return instance;
    }

    // Delete copy constructor and assignment operator to prevent copies
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Public method to log messages
    void log(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "Log: " << message << std::endl;
    }

private:
    Logger() {} // Private constructor
    std::mutex mutex_;
};

int main() {
    Logger::getInstance().log("Application started.");
    Logger::getInstance().log("An error occurred.");
    return 0;
}
```

**Explanation**:

- The `Logger` class has a private constructor to prevent direct instantiation.
- The static method `getInstance()` returns a reference to the single `Logger` instance.
- Copy constructor and assignment operator are deleted to prevent creating additional instances.
- A mutex is used to ensure thread-safe logging operations.

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

