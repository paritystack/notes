# Flutter

Flutter is Google's UI toolkit for building natively compiled applications for mobile, web, and desktop from a single codebase. It uses the Dart programming language and provides a rich set of pre-designed widgets for creating beautiful, high-performance applications.

## Table of Contents
- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Dart Basics](#dart-basics)
- [Widgets](#widgets)
- [Layouts](#layouts)
- [State Management](#state-management)
- [Navigation and Routing](#navigation-and-routing)
- [Networking](#networking)
- [Local Storage](#local-storage)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Introduction

**Key Features:**
- Single codebase for iOS, Android, web, and desktop
- Fast development with hot reload
- Beautiful, customizable widgets
- Native performance
- Rich animation support
- Strong typing with Dart
- Extensive package ecosystem

**Use Cases:**
- Cross-platform mobile apps
- Material Design and Cupertino (iOS-style) apps
- High-performance UIs
- Apps with complex animations
- MVPs and startups
- Enterprise applications

---

## Setup and Installation

### Install Flutter

**macOS:**
```bash
# Download Flutter SDK
# https://flutter.dev/docs/get-started/install/macos

# Add to PATH
export PATH="$PATH:`pwd`/flutter/bin"

# Run doctor
flutter doctor

# Install Xcode
# Install Android Studio
```

**Windows:**
```bash
# Download Flutter SDK
# https://flutter.dev/docs/get-started/install/windows

# Add to PATH
# Run flutter doctor
```

### Create New Project

```bash
# Create project
flutter create my_app
cd my_app

# Run on iOS
flutter run -d ios

# Run on Android
flutter run -d android

# Run on web
flutter run -d chrome
```

### Project Structure

```
my_app/
├── android/          # Android-specific code
├── ios/              # iOS-specific code
├── lib/              # Dart source code
│   ├── main.dart    # Entry point
│   ├── screens/     # Screen widgets
│   ├── widgets/     # Reusable widgets
│   ├── models/      # Data models
│   ├── services/    # API services
│   └── utils/       # Utilities
├── test/            # Tests
├── pubspec.yaml     # Dependencies
└── README.md
```

---

## Dart Basics

### Variables and Types

```dart
// Variables
var name = 'John';
String city = 'New York';
int age = 30;
double height = 5.9;
bool isActive = true;

// Final and const
final String country = 'USA';        // Runtime constant
const double pi = 3.14159;           // Compile-time constant

// Null safety
String? nullableName;                // Can be null
String nonNullName = 'John';         // Cannot be null

// Late initialization
late String description;
```

### Functions

```dart
// Basic function
String greet(String name) {
  return 'Hello, $name!';
}

// Arrow function
String greet(String name) => 'Hello, $name!';

// Optional parameters
String greet(String name, [String? title]) {
  return title != null ? 'Hello, $title $name!' : 'Hello, $name!';
}

// Named parameters
String greet({required String name, String title = 'Mr.'}) {
  return 'Hello, $title $name!';
}

// Async function
Future<String> fetchData() async {
  await Future.delayed(Duration(seconds: 2));
  return 'Data loaded';
}
```

### Classes

```dart
class User {
  String name;
  int age;

  // Constructor
  User(this.name, this.age);

  // Named constructor
  User.guest() : name = 'Guest', age = 0;

  // Method
  String introduce() {
    return 'I am $name, $age years old';
  }

  // Getter
  bool get isAdult => age >= 18;

  // Setter
  set updateAge(int newAge) {
    if (newAge > 0) age = newAge;
  }
}

// Usage
var user = User('John', 30);
print(user.introduce());
print(user.isAdult);
```

### Lists and Maps

```dart
// Lists
List<String> names = ['John', 'Jane', 'Bob'];
names.add('Alice');
names.remove('Bob');

// Maps
Map<String, int> ages = {
  'John': 30,
  'Jane': 28,
};
ages['Bob'] = 35;

// Iteration
names.forEach((name) => print(name));
ages.forEach((key, value) => print('$key: $value'));
```

---

## Widgets

### Stateless Widget

```dart
import 'package:flutter/material.dart';

class WelcomeScreen extends StatelessWidget {
  final String title;

  const WelcomeScreen({Key? key, required this.title}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Center(
        child: Text(
          'Welcome to Flutter!',
          style: TextStyle(fontSize: 24),
        ),
      ),
    );
  }
}
```

### Stateful Widget

```dart
class CounterScreen extends StatefulWidget {
  @override
  _CounterScreenState createState() => _CounterScreenState();
}

class _CounterScreenState extends State<CounterScreen> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Counter'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Count:'),
            Text(
              '$_counter',
              style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        child: Icon(Icons.add),
      ),
    );
  }
}
```

### Common Widgets

```dart
// Text
Text(
  'Hello Flutter',
  style: TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.bold,
    color: Colors.blue,
  ),
)

// Image
Image.network('https://example.com/image.jpg')
Image.asset('assets/logo.png')

// Button
ElevatedButton(
  onPressed: () {
    print('Button pressed');
  },
  child: Text('Click Me'),
)

// TextField
TextField(
  decoration: InputDecoration(
    labelText: 'Email',
    hintText: 'Enter your email',
    border: OutlineInputBorder(),
  ),
  onChanged: (value) {
    print(value);
  },
)

// Container
Container(
  width: 200,
  height: 100,
  padding: EdgeInsets.all(16),
  margin: EdgeInsets.all(8),
  decoration: BoxDecoration(
    color: Colors.blue,
    borderRadius: BorderRadius.circular(12),
    boxShadow: [
      BoxShadow(
        color: Colors.grey.withOpacity(0.5),
        spreadRadius: 2,
        blurRadius: 5,
        offset: Offset(0, 3),
      ),
    ],
  ),
  child: Text('Styled Container'),
)
```

---

## Layouts

### Column and Row

```dart
Column(
  mainAxisAlignment: MainAxisAlignment.center,
  crossAxisAlignment: CrossAxisAlignment.start,
  children: [
    Text('First'),
    Text('Second'),
    Text('Third'),
  ],
)

Row(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
  children: [
    Icon(Icons.star),
    Icon(Icons.favorite),
    Icon(Icons.thumb_up),
  ],
)
```

### Stack

```dart
Stack(
  children: [
    Container(
      width: 200,
      height: 200,
      color: Colors.blue,
    ),
    Positioned(
      top: 20,
      left: 20,
      child: Text('Overlayed Text'),
    ),
  ],
)
```

### ListView

```dart
// Simple ListView
ListView(
  children: [
    ListTile(
      leading: Icon(Icons.person),
      title: Text('John Doe'),
      subtitle: Text('john@example.com'),
      trailing: Icon(Icons.arrow_forward),
      onTap: () {
        print('Tapped');
      },
    ),
    ListTile(
      leading: Icon(Icons.person),
      title: Text('Jane Smith'),
    ),
  ],
)

// ListView.builder (for large lists)
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) {
    return ListTile(
      title: Text(items[index].name),
    );
  },
)

// ListView.separated
ListView.separated(
  itemCount: items.length,
  itemBuilder: (context, index) => ListTile(
    title: Text(items[index]),
  ),
  separatorBuilder: (context, index) => Divider(),
)
```

### GridView

```dart
GridView.count(
  crossAxisCount: 2,
  children: List.generate(20, (index) {
    return Card(
      child: Center(
        child: Text('Item $index'),
      ),
    );
  }),
)

GridView.builder(
  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
    crossAxisCount: 3,
    crossAxisSpacing: 10,
    mainAxisSpacing: 10,
  ),
  itemCount: items.length,
  itemBuilder: (context, index) {
    return Card(
      child: Image.network(items[index].imageUrl),
    );
  },
)
```

---

## State Management

### Provider

```yaml
# pubspec.yaml
dependencies:
  provider: ^6.0.0
```

```dart
import 'package:provider/provider.dart';

// Model
class Counter with ChangeNotifier {
  int _count = 0;

  int get count => _count;

  void increment() {
    _count++;
    notifyListeners();
  }

  void decrement() {
    _count--;
    notifyListeners();
  }
}

// Main app
void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => Counter(),
      child: MyApp(),
    ),
  );
}

// Consumer widget
class CounterScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Counter')),
      body: Center(
        child: Consumer<Counter>(
          builder: (context, counter, child) {
            return Text(
              '${counter.count}',
              style: TextStyle(fontSize: 48),
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          context.read<Counter>().increment();
        },
        child: Icon(Icons.add),
      ),
    );
  }
}
```

### Riverpod

```yaml
dependencies:
  flutter_riverpod: ^2.0.0
```

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

// Provider
final counterProvider = StateProvider<int>((ref) => 0);

// Main app
void main() {
  runApp(
    ProviderScope(
      child: MyApp(),
    ),
  );
}

// Consumer widget
class CounterScreen extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final counter = ref.watch(counterProvider);

    return Scaffold(
      body: Center(
        child: Text('$counter'),
      ),
      floatingActionButton: FloatingActionButton(
        onPress: () {
          ref.read(counterProvider.notifier).state++;
        },
        child: Icon(Icons.add),
      ),
    );
  }
}
```

---

## Navigation and Routing

### Basic Navigation

```dart
// Navigate to new screen
Navigator.push(
  context,
  MaterialPageRoute(builder: (context) => SecondScreen()),
);

// Navigate back
Navigator.pop(context);

// Navigate with data
Navigator.push(
  context,
  MaterialPageRoute(
    builder: (context) => DetailScreen(id: 123),
  ),
);

// Return data
final result = await Navigator.push(
  context,
  MaterialPageRoute(builder: (context) => SecondScreen()),
);
```

### Named Routes

```dart
// Define routes
MaterialApp(
  initialRoute: '/',
  routes: {
    '/': (context) => HomeScreen(),
    '/details': (context) => DetailsScreen(),
    '/profile': (context) => ProfileScreen(),
  },
)

// Navigate
Navigator.pushNamed(context, '/details');

// With arguments
Navigator.pushNamed(
  context,
  '/details',
  arguments: {'id': 123},
);

// Extract arguments
class DetailsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final args = ModalRoute.of(context)!.settings.arguments as Map;
    final id = args['id'];

    return Scaffold(
      appBar: AppBar(title: Text('Details $id')),
    );
  }
}
```

---

## Networking

### HTTP Package

```yaml
dependencies:
  http: ^0.13.0
```

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

// GET request
Future<List<User>> fetchUsers() async {
  final response = await http.get(
    Uri.parse('https://api.example.com/users'),
  );

  if (response.statusCode == 200) {
    List<dynamic> data = jsonDecode(response.body);
    return data.map((json) => User.fromJson(json)).toList();
  } else {
    throw Exception('Failed to load users');
  }
}

// POST request
Future<User> createUser(String name, String email) async {
  final response = await http.post(
    Uri.parse('https://api.example.com/users'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'name': name,
      'email': email,
    }),
  );

  if (response.statusCode == 201) {
    return User.fromJson(jsonDecode(response.body));
  } else {
    throw Exception('Failed to create user');
  }
}

// FutureBuilder
class UsersList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<User>>(
      future: fetchUsers(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return CircularProgressIndicator();
        } else if (snapshot.hasError) {
          return Text('Error: ${snapshot.error}');
        } else if (snapshot.hasData) {
          return ListView.builder(
            itemCount: snapshot.data!.length,
            itemBuilder: (context, index) {
              return ListTile(
                title: Text(snapshot.data![index].name),
              );
            },
          );
        } else {
          return Text('No data');
        }
      },
    );
  }
}
```

---

## Local Storage

### Shared Preferences

```yaml
dependencies:
  shared_preferences: ^2.0.0
```

```dart
import 'package:shared_preferences/shared_preferences.dart';

// Save data
Future<void> saveData() async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.setString('username', 'John');
  await prefs.setInt('age', 30);
  await prefs.setBool('isLoggedIn', true);
}

// Read data
Future<String?> readData() async {
  final prefs = await SharedPreferences.getInstance();
  return prefs.getString('username');
}

// Remove data
Future<void> removeData() async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.remove('username');
}
```

### SQLite

```yaml
dependencies:
  sqflite: ^2.0.0
  path: ^1.8.0
```

```dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static Database? _database;

  DatabaseHelper._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('users.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    return await openDatabase(
      path,
      version: 1,
      onCreate: _createDB,
    );
  }

  Future _createDB(Database db, int version) async {
    await db.execute('''
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL
      )
    ''');
  }

  Future<int> insert(Map<String, dynamic> row) async {
    final db = await database;
    return await db.insert('users', row);
  }

  Future<List<Map<String, dynamic>>> queryAll() async {
    final db = await database;
    return await db.query('users');
  }

  Future<int> update(Map<String, dynamic> row) async {
    final db = await database;
    int id = row['id'];
    return await db.update('users', row, where: 'id = ?', whereArgs: [id]);
  }

  Future<int> delete(int id) async {
    final db = await database;
    return await db.delete('users', where: 'id = ?', whereArgs: [id]);
  }
}
```

---

## Testing

### Unit Tests

```dart
// test/counter_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:my_app/counter.dart';

void main() {
  test('Counter increments', () {
    final counter = Counter();
    counter.increment();
    expect(counter.count, 1);
  });

  test('Counter decrements', () {
    final counter = Counter();
    counter.decrement();
    expect(counter.count, -1);
  });
}
```

### Widget Tests

```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_app/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build the widget
    await tester.pumpWidget(MyApp());

    // Verify initial state
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the '+' icon and trigger a frame
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verify counter incremented
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
```

---

## Deployment

### Android

```bash
# Build APK
flutter build apk --release

# Build App Bundle (recommended)
flutter build appbundle --release

# Split APKs by ABI
flutter build apk --split-per-abi
```

### iOS

```bash
# Build for iOS
flutter build ios --release

# Or use Xcode
# Open ios/Runner.xcworkspace
# Product > Archive
# Upload to App Store
```

### Configuration

**android/app/build.gradle:**
```gradle
android {
    defaultConfig {
        applicationId "com.example.myapp"
        minSdkVersion 21
        targetSdkVersion 33
        versionCode 1
        versionName "1.0.0"
    }

    signingConfigs {
        release {
            storeFile file("upload-keystore.jks")
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias "upload"
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }
}
```

**ios/Runner/Info.plist:**
```xml
<key>CFBundleDisplayName</key>
<string>My App</string>
<key>CFBundleVersion</key>
<string>1</string>
<key>CFBundleShortVersionString</key>
<string>1.0.0</string>
```

---

## Resources

**Official Documentation:**
- [Flutter Documentation](https://flutter.dev/docs)
- [Dart Documentation](https://dart.dev/guides)
- [Flutter Cookbook](https://flutter.dev/docs/cookbook)

**Packages:**
- [pub.dev](https://pub.dev/) - Official package repository
- [Flutter Awesome](https://flutterawesome.com/) - Curated packages

**Learning:**
- [Flutter Codelabs](https://flutter.dev/docs/codelabs)
- [Flutter Widget of the Week](https://www.youtube.com/playlist?list=PLjxrf2q8roU23XGwz3Km7sQZFTdB996iG)
- [Flutter Community](https://flutter.dev/community)

**Tools:**
- [Flutter DevTools](https://flutter.dev/docs/development/tools/devtools/overview)
- [DartPad](https://dartpad.dev/) - Online IDE
- [FlutterFire](https://firebase.flutter.dev/) - Firebase integration
