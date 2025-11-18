# iOS Development Guide

## Overview

This guide covers the complete iOS development workflow, from setting up Xcode to building, debugging, and deploying applications. It focuses on practical development practices and tools that every iOS developer should know, including both UIKit and SwiftUI approaches.

## Xcode Setup

### Installation

Xcode is Apple's official IDE for iOS, macOS, watchOS, and tvOS development.

```bash
# Install from Mac App Store
# Or download from https://developer.apple.com/xcode/

# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p
# Output: /Applications/Xcode.app/Contents/Developer

# Check Xcode version
xcodebuild -version
```

### Initial Configuration

1. **First Launch**: Accept license agreement
2. **Platforms**: Install iOS SDK and simulators
3. **Preferences**: Configure editor, themes, and behaviors
4. **Accounts**: Add Apple ID for development

### Simulator Management

```bash
# List available simulators
xcrun simctl list devices

# List available device types
xcrun simctl list devicetypes

# Create new simulator
xcrun simctl create "iPhone 15 Pro" "iPhone 15 Pro" "iOS17.0"

# Boot simulator
xcrun simctl boot "iPhone 15 Pro"

# Open Simulator app
open -a Simulator

# Install app on simulator
xcrun simctl install booted YourApp.app

# Uninstall app
xcrun simctl uninstall booted com.example.yourapp
```

### Command Line Tools

```bash
# Build project
xcodebuild -project MyApp.xcodeproj -scheme MyApp -configuration Debug

# Build workspace (for CocoaPods projects)
xcodebuild -workspace MyApp.xcworkspace -scheme MyApp -configuration Release

# Run tests
xcodebuild test -scheme MyApp -destination 'platform=iOS Simulator,name=iPhone 15 Pro'

# Clean build
xcodebuild clean -project MyApp.xcodeproj

# Archive for distribution
xcodebuild archive -scheme MyApp -archivePath build/MyApp.xcarchive
```

## Project Structure

### Standard iOS Project

```
MyApp/
├── MyApp/
│   ├── AppDelegate.swift
│   ├── SceneDelegate.swift
│   ├── ViewControllers/
│   │   ├── MainViewController.swift
│   │   └── DetailViewController.swift
│   ├── Views/
│   │   ├── CustomView.swift
│   │   └── Cells/
│   ├── Models/
│   │   └── User.swift
│   ├── ViewModels/
│   │   └── UserViewModel.swift
│   ├── Services/
│   │   ├── NetworkService.swift
│   │   └── DataStore.swift
│   ├── Resources/
│   │   ├── Assets.xcassets
│   │   └── LaunchScreen.storyboard
│   ├── Storyboards/
│   │   └── Main.storyboard
│   └── Info.plist
├── MyAppTests/
│   └── MyAppTests.swift
├── MyAppUITests/
│   └── MyAppUITests.swift
└── MyApp.xcodeproj
```

### Key Files and Directories

- **AppDelegate.swift**: App lifecycle management
- **SceneDelegate.swift**: Scene lifecycle (iOS 13+)
- **ViewControllers/**: Screen logic and coordination
- **Views/**: Custom UI components
- **Models/**: Data structures
- **Resources/**: Images, colors, launch screens
- **Info.plist**: App configuration and permissions

## Swift Basics

### Variables and Constants

```swift
// Constants (immutable)
let name = "John"
let age: Int = 30
let pi: Double = 3.14159

// Variables (mutable)
var score = 100
var isLoggedIn = false

// Type inference
let city = "San Francisco"  // String inferred

// Optional types
var optionalName: String? = "Jane"
var optionalAge: Int? = nil

// Unwrapping optionals
if let name = optionalName {
    print("Name is \(name)")
} else {
    print("Name is nil")
}

// Guard statement
guard let name = optionalName else {
    print("No name provided")
    return
}

// Nil coalescing
let displayName = optionalName ?? "Anonymous"

// Optional chaining
let uppercasedName = optionalName?.uppercased()
```

### Collections

```swift
// Arrays
var numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.insert(0, at: 0)
numbers.remove(at: 0)

// Dictionaries
var person = ["name": "John", "city": "NYC"]
person["age"] = "30"
person["name"] = nil  // Remove key

// Sets
var uniqueNumbers = Set([1, 2, 3, 2, 1])  // {1, 2, 3}
uniqueNumbers.insert(4)

// Iteration
for number in numbers {
    print(number)
}

for (key, value) in person {
    print("\(key): \(value)")
}

// Map, filter, reduce
let doubled = numbers.map { $0 * 2 }
let evens = numbers.filter { $0 % 2 == 0 }
let sum = numbers.reduce(0, +)
```

### Functions and Closures

```swift
// Basic function
func greet(name: String) -> String {
    return "Hello, \(name)!"
}

// Multiple parameters
func add(a: Int, b: Int) -> Int {
    return a + b
}

// External and internal parameter names
func greet(person name: String, from city: String) {
    print("Hello \(name) from \(city)")
}
greet(person: "John", from: "NYC")

// Default parameters
func greet(name: String = "Guest") {
    print("Hello, \(name)")
}

// Variadic parameters
func sum(_ numbers: Int...) -> Int {
    return numbers.reduce(0, +)
}

// Closures
let multiply = { (a: Int, b: Int) -> Int in
    return a * b
}

// Trailing closure syntax
let sorted = numbers.sorted { $0 > $1 }

// Capturing values
func makeIncrementer(step: Int) -> () -> Int {
    var total = 0
    return {
        total += step
        return total
    }
}
```

### Classes and Structs

```swift
// Struct (value type)
struct Point {
    var x: Double
    var y: Double

    // Computed property
    var magnitude: Double {
        return sqrt(x * x + y * y)
    }

    // Method
    mutating func move(dx: Double, dy: Double) {
        x += dx
        y += dy
    }
}

// Class (reference type)
class Person {
    var name: String
    var age: Int

    // Initializer
    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    // Method
    func greet() {
        print("Hello, I'm \(name)")
    }
}

// Inheritance
class Student: Person {
    var studentId: String

    init(name: String, age: Int, studentId: String) {
        self.studentId = studentId
        super.init(name: name, age: age)
    }

    // Override method
    override func greet() {
        print("Hi, I'm \(name), student \(studentId)")
    }
}

// Protocols
protocol Drawable {
    func draw()
}

class Circle: Drawable {
    func draw() {
        print("Drawing circle")
    }
}

// Extensions
extension String {
    func isPalindrome() -> Bool {
        return self == String(self.reversed())
    }
}
```

## UIKit - View Controllers

### Basic View Controller

```swift
import UIKit

class MainViewController: UIViewController {

    // MARK: - Properties

    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = "Welcome"
        label.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    private let actionButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Tap Me", for: .normal)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // View is about to appear
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // View did appear
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // View is about to disappear
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        // View did disappear
    }

    // MARK: - Setup

    private func setupUI() {
        view.backgroundColor = .systemBackground
        view.addSubview(titleLabel)
        view.addSubview(actionButton)

        actionButton.addTarget(self, action: #selector(handleButtonTap), for: .touchUpInside)
    }

    private func setupConstraints() {
        NSLayoutConstraint.activate([
            titleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 50),

            actionButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            actionButton.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }

    // MARK: - Actions

    @objc private func handleButtonTap() {
        print("Button tapped")
        titleLabel.text = "Button Tapped!"
    }
}
```

### Navigation

```swift
// Push view controller
let detailVC = DetailViewController()
navigationController?.pushViewController(detailVC, animated: true)

// Pop view controller
navigationController?.popViewController(animated: true)

// Pop to root
navigationController?.popToRootViewController(animated: true)

// Present modally
let modalVC = ModalViewController()
modalVC.modalPresentationStyle = .fullScreen
present(modalVC, animated: true, completion: nil)

// Dismiss modal
dismiss(animated: true, completion: nil)

// Pass data between view controllers
let detailVC = DetailViewController()
detailVC.userId = 123
detailVC.userName = "John"
navigationController?.pushViewController(detailVC, animated: true)
```

### Table View

```swift
class UsersViewController: UIViewController {

    private var users: [User] = []

    private lazy var tableView: UITableView = {
        let table = UITableView()
        table.delegate = self
        table.dataSource = self
        table.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        table.translatesAutoresizingMaskIntoConstraints = false
        return table
    }()

    override func viewDidLoad() {
        super.viewDidLoad()

        view.addSubview(tableView)
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])

        loadUsers()
    }

    private func loadUsers() {
        users = [
            User(id: 1, name: "John"),
            User(id: 2, name: "Jane"),
            User(id: 3, name: "Bob")
        ]
        tableView.reloadData()
    }
}

// MARK: - UITableViewDataSource

extension UsersViewController: UITableViewDataSource {

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return users.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        let user = users[indexPath.row]
        cell.textLabel?.text = user.name
        return cell
    }
}

// MARK: - UITableViewDelegate

extension UsersViewController: UITableViewDelegate {

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        let user = users[indexPath.row]
        print("Selected user: \(user.name)")
    }

    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 60
    }
}
```

### Collection View

```swift
class PhotosViewController: UIViewController {

    private var photos: [Photo] = []

    private lazy var collectionView: UICollectionView = {
        let layout = UICollectionViewFlowLayout()
        layout.itemSize = CGSize(width: 100, height: 100)
        layout.minimumInteritemSpacing = 10
        layout.minimumLineSpacing = 10

        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.delegate = self
        cv.dataSource = self
        cv.register(PhotoCell.self, forCellWithReuseIdentifier: "PhotoCell")
        cv.translatesAutoresizingMaskIntoConstraints = false
        cv.backgroundColor = .systemBackground
        return cv
    }()

    override func viewDidLoad() {
        super.viewDidLoad()

        view.addSubview(collectionView)
        NSLayoutConstraint.activate([
            collectionView.topAnchor.constraint(equalTo: view.topAnchor),
            collectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            collectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            collectionView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
}

// MARK: - UICollectionViewDataSource

extension PhotosViewController: UICollectionViewDataSource {

    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return photos.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "PhotoCell", for: indexPath) as! PhotoCell
        cell.configure(with: photos[indexPath.item])
        return cell
    }
}

// MARK: - UICollectionViewDelegate

extension PhotosViewController: UICollectionViewDelegate {

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let photo = photos[indexPath.item]
        print("Selected photo: \(photo.title)")
    }
}

// Custom Cell
class PhotoCell: UICollectionViewCell {

    private let imageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.clipsToBounds = true
        iv.translatesAutoresizingMaskIntoConstraints = false
        return iv
    }()

    override init(frame: CGRect) {
        super.init(frame: frame)
        contentView.addSubview(imageView)
        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: contentView.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor)
        ])
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func configure(with photo: Photo) {
        imageView.image = UIImage(named: photo.imageName)
    }
}
```

## SwiftUI

### Basic Views

```swift
import SwiftUI

// Simple view
struct ContentView: View {
    @State private var name = ""
    @State private var isToggled = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Hello, SwiftUI!")
                .font(.title)
                .foregroundColor(.blue)

            TextField("Enter name", text: $name)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            Text("Hello, \(name)!")

            Toggle("Switch", isOn: $isToggled)
                .padding()

            Button("Tap Me") {
                print("Button tapped")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

// Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
```

### State Management

```swift
// @State - Local state
struct CounterView: View {
    @State private var count = 0

    var body: some View {
        VStack {
            Text("Count: \(count)")
                .font(.largeTitle)

            Button("Increment") {
                count += 1
            }
        }
    }
}

// @Binding - Shared state
struct ParentView: View {
    @State private var text = ""

    var body: some View {
        VStack {
            TextField("Enter text", text: $text)
            ChildView(text: $text)
        }
    }
}

struct ChildView: View {
    @Binding var text: String

    var body: some View {
        Text("You typed: \(text)")
    }
}

// ObservableObject - Complex state
class UserViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false

    func fetchUsers() {
        isLoading = true
        // Fetch users from API
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            self.users = [
                User(id: 1, name: "John"),
                User(id: 2, name: "Jane")
            ]
            self.isLoading = false
        }
    }
}

struct UsersView: View {
    @StateObject private var viewModel = UserViewModel()

    var body: some View {
        List(viewModel.users, id: \.id) { user in
            Text(user.name)
        }
        .onAppear {
            viewModel.fetchUsers()
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView()
            }
        }
    }
}

// @EnvironmentObject - Shared across views
class AppState: ObservableObject {
    @Published var isLoggedIn = false
    @Published var currentUser: User?
}

@main
struct MyApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

struct ProfileView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        if let user = appState.currentUser {
            Text("Welcome, \(user.name)")
        }
    }
}
```

### Lists and Navigation

```swift
struct User: Identifiable {
    let id: Int
    let name: String
    let email: String
}

struct UserListView: View {
    let users = [
        User(id: 1, name: "John Doe", email: "john@example.com"),
        User(id: 2, name: "Jane Smith", email: "jane@example.com"),
        User(id: 3, name: "Bob Johnson", email: "bob@example.com")
    ]

    var body: some View {
        NavigationView {
            List(users) { user in
                NavigationLink(destination: UserDetailView(user: user)) {
                    UserRow(user: user)
                }
            }
            .navigationTitle("Users")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

struct UserRow: View {
    let user: User

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(user.name)
                .font(.headline)
            Text(user.email)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

struct UserDetailView: View {
    let user: User

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.circle.fill")
                .resizable()
                .frame(width: 100, height: 100)
                .foregroundColor(.blue)

            Text(user.name)
                .font(.title)

            Text(user.email)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .navigationTitle("Profile")
        .navigationBarTitleDisplayMode(.inline)
    }
}
```

### Forms and Input

```swift
struct SettingsView: View {
    @State private var username = ""
    @State private var email = ""
    @State private var enableNotifications = true
    @State private var selectedTheme = "Light"
    @State private var fontSize: Double = 14

    let themes = ["Light", "Dark", "Auto"]

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Account")) {
                    TextField("Username", text: $username)
                    TextField("Email", text: $email)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                }

                Section(header: Text("Preferences")) {
                    Toggle("Enable Notifications", isOn: $enableNotifications)

                    Picker("Theme", selection: $selectedTheme) {
                        ForEach(themes, id: \.self) { theme in
                            Text(theme)
                        }
                    }

                    VStack {
                        Text("Font Size: \(Int(fontSize))")
                        Slider(value: $fontSize, in: 10...24, step: 1)
                    }
                }

                Section {
                    Button("Save Changes") {
                        saveSettings()
                    }

                    Button("Reset to Defaults", role: .destructive) {
                        resetSettings()
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func saveSettings() {
        print("Settings saved")
    }

    private func resetSettings() {
        username = ""
        email = ""
        enableNotifications = true
        selectedTheme = "Light"
        fontSize = 14
    }
}
```

## Networking

### URLSession

```swift
import Foundation

// GET Request
func fetchUsers(completion: @escaping (Result<[User], Error>) -> Void) {
    guard let url = URL(string: "https://api.example.com/users") else {
        completion(.failure(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
        return
    }

    URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            completion(.failure(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
            return
        }

        guard let data = data else {
            completion(.failure(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data"])))
            return
        }

        do {
            let users = try JSONDecoder().decode([User].self, from: data)
            completion(.success(users))
        } catch {
            completion(.failure(error))
        }
    }.resume()
}

// POST Request
func createUser(user: User, completion: @escaping (Result<User, Error>) -> Void) {
    guard let url = URL(string: "https://api.example.com/users") else { return }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    do {
        let jsonData = try JSONEncoder().encode(user)
        request.httpBody = jsonData
    } catch {
        completion(.failure(error))
        return
    }

    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }

        guard let data = data else { return }

        do {
            let createdUser = try JSONDecoder().decode(User.self, from: data)
            completion(.success(createdUser))
        } catch {
            completion(.failure(error))
        }
    }.resume()
}

// Async/Await (iOS 15+)
func fetchUsers() async throws -> [User] {
    guard let url = URL(string: "https://api.example.com/users") else {
        throw URLError(.badURL)
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200...299).contains(httpResponse.statusCode) else {
        throw URLError(.badServerResponse)
    }

    let users = try JSONDecoder().decode([User].self, from: data)
    return users
}

// Usage with async/await
Task {
    do {
        let users = try await fetchUsers()
        print("Fetched \(users.count) users")
    } catch {
        print("Error: \(error)")
    }
}
```

### Network Service

```swift
class NetworkService {

    static let shared = NetworkService()

    private init() {}

    func request<T: Decodable>(
        url: URL,
        method: String = "GET",
        body: Data? = nil,
        headers: [String: String]? = nil
    ) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body

        headers?.forEach { key, value in
            request.setValue(value, forHTTPHeaderField: key)
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }

        return try JSONDecoder().decode(T.self, from: data)
    }
}

// Usage
struct APIEndpoints {
    static let baseURL = "https://api.example.com"

    static func users() -> URL {
        URL(string: "\(baseURL)/users")!
    }

    static func user(id: Int) -> URL {
        URL(string: "\(baseURL)/users/\(id)")!
    }
}

Task {
    let users: [User] = try await NetworkService.shared.request(url: APIEndpoints.users())
    print(users)
}
```

## Data Persistence

### UserDefaults

```swift
// Save data
UserDefaults.standard.set("John", forKey: "username")
UserDefaults.standard.set(25, forKey: "age")
UserDefaults.standard.set(true, forKey: "isLoggedIn")

// Retrieve data
let username = UserDefaults.standard.string(forKey: "username")
let age = UserDefaults.standard.integer(forKey: "age")
let isLoggedIn = UserDefaults.standard.bool(forKey: "isLoggedIn")

// Remove data
UserDefaults.standard.removeObject(forKey: "username")

// Save custom objects
struct Settings: Codable {
    var theme: String
    var notifications: Bool
}

let settings = Settings(theme: "dark", notifications: true)
if let encoded = try? JSONEncoder().encode(settings) {
    UserDefaults.standard.set(encoded, forKey: "settings")
}

// Retrieve custom objects
if let data = UserDefaults.standard.data(forKey: "settings"),
   let settings = try? JSONDecoder().decode(Settings.self, from: data) {
    print(settings.theme)
}
```

### FileManager

```swift
// Get documents directory
func getDocumentsDirectory() -> URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
}

// Write to file
func saveToFile(text: String, filename: String) {
    let url = getDocumentsDirectory().appendingPathComponent(filename)
    do {
        try text.write(to: url, atomically: true, encoding: .utf8)
    } catch {
        print("Error writing to file: \(error)")
    }
}

// Read from file
func readFromFile(filename: String) -> String? {
    let url = getDocumentsDirectory().appendingPathComponent(filename)
    do {
        return try String(contentsOf: url, encoding: .utf8)
    } catch {
        print("Error reading file: \(error)")
        return nil
    }
}

// Check if file exists
func fileExists(filename: String) -> Bool {
    let url = getDocumentsDirectory().appendingPathComponent(filename)
    return FileManager.default.fileExists(atPath: url.path)
}

// Delete file
func deleteFile(filename: String) {
    let url = getDocumentsDirectory().appendingPathComponent(filename)
    do {
        try FileManager.default.removeItem(at: url)
    } catch {
        print("Error deleting file: \(error)")
    }
}
```

### Core Data

```swift
import CoreData

// Define Entity (in .xcdatamodeld file)
// Create NSManagedObject subclass

class CoreDataManager {

    static let shared = CoreDataManager()

    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "MyApp")
        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Unable to load persistent stores: \(error)")
            }
        }
        return container
    }()

    var context: NSManagedObjectContext {
        return persistentContainer.viewContext
    }

    func saveContext() {
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                let error = error as NSError
                fatalError("Unresolved error \(error), \(error.userInfo)")
            }
        }
    }

    // Create
    func createUser(name: String, email: String) {
        let user = User(context: context)
        user.name = name
        user.email = email
        user.createdAt = Date()
        saveContext()
    }

    // Read
    func fetchUsers() -> [User] {
        let request: NSFetchRequest<User> = User.fetchRequest()
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching users: \(error)")
            return []
        }
    }

    // Update
    func updateUser(user: User, name: String) {
        user.name = name
        saveContext()
    }

    // Delete
    func deleteUser(user: User) {
        context.delete(user)
        saveContext()
    }
}
```

## Debugging

### Print Debugging

```swift
// Basic print
print("Debug message")

// Print with variables
let name = "John"
let age = 30
print("Name: \(name), Age: \(age)")

// Debug print (only in debug builds)
#if DEBUG
print("Debug build")
#endif

// Custom debug print
func debugLog(_ message: String, file: String = #file, line: Int = #line) {
    #if DEBUG
    print("[\(file):\(line)] \(message)")
    #endif
}
```

### Breakpoints

```
1. Click line number in Xcode to set breakpoint
2. Run app in debug mode (Cmd+R)
3. When breakpoint hits:
   - Step Over: F6
   - Step Into: F7
   - Step Out: F8
   - Continue: Cmd+Ctrl+Y

LLDB Commands:
- po variable         # Print object
- p variable          # Print value
- expr variable = 10  # Change variable
- bt                  # Backtrace
- frame variable      # Show all variables
```

### Instruments

```bash
# Profile app performance
Product > Profile (Cmd+I)

Common Instruments:
- Time Profiler: CPU usage
- Allocations: Memory usage
- Leaks: Memory leaks
- Network: Network activity
- Energy Log: Battery usage
```

### View Debugging

```
Debug > View Debugging > Capture View Hierarchy

- 3D visualization of view hierarchy
- Inspect view properties
- Find overlapping views
- Identify layout issues
```

## Building and Deployment

### Build Configurations

```bash
# Debug build
xcodebuild -scheme MyApp -configuration Debug

# Release build
xcodebuild -scheme MyApp -configuration Release

# Clean build folder
xcodebuild clean -scheme MyApp
```

### Code Signing

```bash
# List available signing identities
security find-identity -v -p codesigning

# Set signing in Xcode
# Project Settings > Signing & Capabilities
# - Select Team
# - Choose signing certificate
# - Enable "Automatically manage signing" (recommended)
```

### Creating Archive

```bash
# Archive from command line
xcodebuild archive \
  -scheme MyApp \
  -configuration Release \
  -archivePath build/MyApp.xcarchive

# Export IPA
xcodebuild -exportArchive \
  -archivePath build/MyApp.xcarchive \
  -exportPath build/MyApp \
  -exportOptionsPlist ExportOptions.plist
```

### App Store Submission

```bash
# Validate app
xcrun altool --validate-app \
  -f MyApp.ipa \
  -t ios \
  -u username@example.com \
  -p app-specific-password

# Upload to App Store Connect
xcrun altool --upload-app \
  -f MyApp.ipa \
  -t ios \
  -u username@example.com \
  -p app-specific-password

# Or use Xcode
# Archive > Distribute App > App Store Connect
```

### TestFlight

```
1. Archive app in Xcode
2. Distribute App > App Store Connect
3. Upload to App Store Connect
4. In App Store Connect:
   - Select app
   - Go to TestFlight tab
   - Add internal/external testers
   - Distribute build
```

## Dependency Management

### Swift Package Manager (SPM)

```swift
// In Xcode:
// File > Add Packages...
// Enter package URL

// Package.swift
// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.6.0"),
    ],
    targets: [
        .target(
            name: "MyApp",
            dependencies: ["Alamofire"]
        ),
    ]
)
```

### CocoaPods

```bash
# Install CocoaPods
sudo gem install cocoapods

# Initialize Podfile
pod init

# Edit Podfile
# platform :ios, '15.0'
# use_frameworks!
#
# target 'MyApp' do
#   pod 'Alamofire', '~> 5.6'
#   pod 'Kingfisher', '~> 7.0'
# end

# Install pods
pod install

# Open workspace (not project)
open MyApp.xcworkspace

# Update pods
pod update
```

## Best Practices

1. **Use SwiftUI** for new projects (iOS 13+)
2. **Follow MVC/MVVM** architecture patterns
3. **Use Auto Layout** or SwiftUI for adaptive layouts
4. **Handle errors gracefully** with proper error messages
5. **Test on real devices** in addition to simulators
6. **Use lazy properties** to defer expensive initialization
7. **Avoid force unwrapping** optionals (use if-let or guard)
8. **Use protocols** for loose coupling
9. **Keep view controllers small** - extract logic to services
10. **Follow Swift API Design Guidelines**

## Related Resources

- [Mobile Development Overview](README.md)
- [Flutter Development](flutter.md)
- [React Native Development](react_native.md)
- [Official Swift Documentation](https://swift.org/documentation/)
- [Apple Developer Documentation](https://developer.apple.com/documentation/)
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
