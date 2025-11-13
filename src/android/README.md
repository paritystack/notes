# Android Development

## Overview

Android is an open-source operating system based on the Linux kernel, designed primarily for mobile devices. It's the world's most popular mobile platform, powering billions of devices worldwide. Android development involves creating applications using Java, Kotlin, or C++ that run on Android devices.

## Key Concepts

### Android Platform Architecture

Android is built on a multi-layered architecture:

1. **Linux Kernel**: Foundation providing core system services
2. **Hardware Abstraction Layer (HAL)**: Interface between hardware and software
3. **Android Runtime (ART)**: Executes app bytecode with optimized performance
4. **Native C/C++ Libraries**: Core system libraries (SQLite, OpenGL, etc.)
5. **Java API Framework**: High-level APIs for app development
6. **System Apps**: Pre-installed applications

### Core Components

Android applications are built using four fundamental components:

1. **Activities**: Single screen with a user interface
2. **Services**: Background operations without UI
3. **Broadcast Receivers**: Respond to system-wide broadcast announcements
4. **Content Providers**: Manage shared app data

## Development Environment

### Prerequisites

- **Java Development Kit (JDK)**: Version 8 or higher
- **Android Studio**: Official IDE for Android development
- **Android SDK**: Software development kit with tools and APIs
- **Gradle**: Build automation system

### Installation

```bash
# Download Android Studio from https://developer.android.com/studio
# Install Android Studio and SDK through the setup wizard

# Verify installation
adb --version
```

## Quick Start

### Creating Your First App

```kotlin
// MainActivity.kt
package com.example.myfirstapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        val textView: TextView = findViewById(R.id.textView)
        textView.text = "Hello, Android!"
    }
}
```

```xml
<!-- res/layout/activity_main.xml -->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center">
    
    <TextView
        android:id="@+id/textView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        android:textSize="24sp" />
        
</LinearLayout>
```

### AndroidManifest.xml

Every Android app must have a manifest file:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.myfirstapp">
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.AppCompat">
        
        <activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
    
    <!-- Permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
</manifest>
```

## Android Application Components

### Activities Lifecycle

```kotlin
class MyActivity : AppCompatActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Initialize activity
        setContentView(R.layout.activity_my)
    }
    
    override fun onStart() {
        super.onStart()
        // Activity is becoming visible
    }
    
    override fun onResume() {
        super.onResume()
        // Activity is interactive
    }
    
    override fun onPause() {
        super.onPause()
        // Activity is losing focus
    }
    
    override fun onStop() {
        super.onStop()
        // Activity is no longer visible
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Activity is being destroyed
    }
}
```

### Intents

Intents are messaging objects used to request actions from other components:

```kotlin
// Explicit Intent - Start specific activity
val intent = Intent(this, SecondActivity::class.java)
intent.putExtra("KEY_NAME", "value")
startActivity(intent)

// Implicit Intent - Let system find appropriate component
val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse("https://www.example.com"))
startActivity(browserIntent)

// Share content
val shareIntent = Intent().apply {
    action = Intent.ACTION_SEND
    putExtra(Intent.EXTRA_TEXT, "Check out this content!")
    type = "text/plain"
}
startActivity(Intent.createChooser(shareIntent, "Share via"))
```

## UI Development

### Views and ViewGroups

```kotlin
// Programmatically create UI
val layout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL
    layoutParams = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.MATCH_PARENT
    )
}

val button = Button(this).apply {
    text = "Click Me"
    setOnClickListener {
        Toast.makeText(context, "Button clicked!", Toast.LENGTH_SHORT).show()
    }
}

layout.addView(button)
setContentView(layout)
```

### RecyclerView Example

```kotlin
// Adapter
class MyAdapter(private val items: List<String>) : 
    RecyclerView.Adapter<MyAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textView: TextView = view.findViewById(R.id.textView)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_layout, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.textView.text = items[position]
    }
    
    override fun getItemCount() = items.size
}

// Usage in Activity
val recyclerView: RecyclerView = findViewById(R.id.recyclerView)
recyclerView.layoutManager = LinearLayoutManager(this)
recyclerView.adapter = MyAdapter(listOf("Item 1", "Item 2", "Item 3"))
```

## Data Storage

### SharedPreferences

```kotlin
// Save data
val sharedPref = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE)
with(sharedPref.edit()) {
    putString("username", "john_doe")
    putInt("score", 100)
    apply()
}

// Read data
val username = sharedPref.getString("username", "default")
val score = sharedPref.getInt("score", 0)
```

### Room Database

```kotlin
// Entity
@Entity(tableName = "users")
data class User(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "name") val name: String,
    @ColumnInfo(name = "email") val email: String
)

// DAO
@Dao
interface UserDao {
    @Query("SELECT * FROM users")
    fun getAllUsers(): List<User>
    
    @Insert
    fun insert(user: User)
    
    @Delete
    fun delete(user: User)
}

// Database
@Database(entities = [User::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
}
```

## Networking

### Retrofit Example

```kotlin
// API Interface
interface ApiService {
    @GET("users/{id}")
    suspend fun getUser(@Path("id") userId: Int): User
    
    @POST("users")
    suspend fun createUser(@Body user: User): User
}

// Implementation
val retrofit = Retrofit.Builder()
    .baseUrl("https://api.example.com/")
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val apiService = retrofit.create(ApiService::class.java)

// Usage with Coroutines
lifecycleScope.launch {
    try {
        val user = apiService.getUser(1)
        // Update UI with user data
    } catch (e: Exception) {
        // Handle error
    }
}
```

## Modern Android Development

### Jetpack Compose

Jetpack Compose is Android's modern toolkit for building native UI:

```kotlin
@Composable
fun Greeting(name: String) {
    Text(
        text = "Hello $name!",
        modifier = Modifier.padding(16.dp),
        style = MaterialTheme.typography.h4
    )
}

@Composable
fun Counter() {
    var count by remember { mutableStateOf(0) }
    
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Count: $count")
        Button(onClick = { count++ }) {
            Text("Increment")
        }
    }
}
```

### ViewModel

```kotlin
class MyViewModel : ViewModel() {
    private val _uiState = MutableLiveData<UiState>()
    val uiState: LiveData<UiState> = _uiState
    
    fun loadData() {
        viewModelScope.launch {
            try {
                val data = repository.getData()
                _uiState.value = UiState.Success(data)
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message)
            }
        }
    }
}

// Usage in Activity
class MainActivity : AppCompatActivity() {
    private val viewModel: MyViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        viewModel.uiState.observe(this) { state ->
            when (state) {
                is UiState.Success -> updateUI(state.data)
                is UiState.Error -> showError(state.message)
            }
        }
    }
}
```

## Testing

### Unit Tests

```kotlin
class CalculatorTest {
    @Test
    fun addition_isCorrect() {
        assertEquals(4, 2 + 2)
    }
    
    @Test
    fun viewModel_loadsData() = runTest {
        val viewModel = MyViewModel(FakeRepository())
        viewModel.loadData()
        
        val state = viewModel.uiState.value
        assertTrue(state is UiState.Success)
    }
}
```

### Instrumented Tests

```kotlin
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)
    
    @Test
    fun testButtonClick() {
        onView(withId(R.id.button))
            .perform(click())
        
        onView(withId(R.id.textView))
            .check(matches(withText("Button clicked!")))
    }
}
```

## Build Configuration

### build.gradle (Module level)

```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.example.myapp'
    compileSdk 34
    
    defaultConfig {
        applicationId "com.example.myapp"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
    }
    
    buildTypes {
        release {
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.11.0'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
}
```

## Best Practices

1. **Follow Material Design Guidelines**: Use Material Components for consistent UI
2. **Handle Configuration Changes**: Save state during rotation
3. **Use Architecture Components**: ViewModel, LiveData, Room
4. **Implement Proper Error Handling**: Never ignore exceptions
5. **Optimize Performance**: Avoid blocking the main thread
6. **Test Your Code**: Write unit and instrumented tests
7. **Follow Android Security Best Practices**: Validate inputs, use encryption
8. **Support Multiple Screen Sizes**: Use responsive layouts
9. **Handle Permissions Properly**: Request permissions at runtime
10. **Keep Libraries Updated**: Use latest stable versions

## Resources

### Documentation

- [Android Developers](https://developer.android.com/)
- [Android API Reference](https://developer.android.com/reference)
- [Kotlin Documentation](https://kotlinlang.org/docs/home.html)
- [Jetpack Compose](https://developer.android.com/jetpack/compose)

### Related Files

- [Android Internals](internals.md) - Understanding Android architecture
- [ADB Commands](adb.md) - Android Debug Bridge reference
- [Development Guide](development.md) - Detailed development workflow
- [Android Binder](binder.md) - Inter-process communication mechanism

## Common Issues

### Gradle Sync Failed
```bash
# Clear Gradle cache
./gradlew clean
# Invalidate caches in Android Studio: File > Invalidate Caches / Restart
```

### App Crashes on Launch
- Check Logcat for stack traces
- Verify all required permissions are declared
- Ensure ProGuard rules are correct for release builds

### Memory Leaks
- Use LeakCanary for detection
- Avoid holding Activity context in long-lived objects
- Unregister listeners and callbacks

## Next Steps

1. Complete the [Android Development Guide](development.md)
2. Learn [ADB commands](adb.md) for debugging
3. Study [Android Internals](internals.md) for deeper understanding
4. Build sample projects to practice
5. Explore Jetpack Compose for modern UI development
