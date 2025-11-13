# Android Development Guide

## Overview

This guide covers the complete Android development workflow, from setting up Android Studio to building, debugging, and deploying applications. It focuses on practical development practices and tools that every Android developer should know.

## Android Studio Setup

### Installation

Android Studio is the official IDE for Android development, built on JetBrains' IntelliJ IDEA.

```bash
# Download from https://developer.android.com/studio

# Linux installation
sudo tar -xzf android-studio-*.tar.gz -C /opt/
cd /opt/android-studio/bin
./studio.sh

# Add to PATH (optional)
echo 'export PATH=$PATH:/opt/android-studio/bin' >> ~/.bashrc
```

### Initial Configuration

1. **Welcome Screen**: Choose "Standard" installation
2. **SDK Components**: Install latest Android SDK and tools
3. **Emulator**: Install Android Emulator and system images
4. **Gradle**: Let Android Studio manage Gradle installation

### SDK Manager

```bash
# Open SDK Manager: Tools > SDK Manager

# Essential SDK Packages:
# - Android SDK Platform (latest)
# - Android SDK Build-Tools
# - Android Emulator
# - Android SDK Platform-Tools
# - Android SDK Tools

# Command-line SDK management
sdkmanager --list
sdkmanager "platform-tools" "platforms;android-34"
sdkmanager --update
```

### AVD Manager

Create virtual devices for testing:

```bash
# Open AVD Manager: Tools > AVD Manager

# Or use command line
avdmanager create avd -n Pixel_7 -k "system-images;android-34;google_apis;x86_64"
avdmanager list avd

# Start emulator from command line
emulator -avd Pixel_7
```

## Project Structure

### Standard Android Project

```
MyApp/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/myapp/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── models/
│   │   │   │   ├── viewmodels/
│   │   │   │   └── repositories/
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   ├── values/
│   │   │   │   ├── drawable/
│   │   │   │   ├── mipmap/
│   │   │   │   └── menu/
│   │   │   └── AndroidManifest.xml
│   │   ├── test/          # Unit tests
│   │   └── androidTest/   # Instrumented tests
│   ├── build.gradle
│   └── proguard-rules.pro
├── gradle/
├── build.gradle
├── settings.gradle
└── gradle.properties
```

### Key Directories

- **java/**: Source code (Kotlin/Java)
- **res/**: Resources (layouts, strings, images)
- **res/layout/**: XML layout files
- **res/values/**: Strings, colors, dimensions, styles
- **res/drawable/**: Images and vector graphics
- **res/mipmap/**: App launcher icons
- **AndroidManifest.xml**: App configuration and permissions

## Activities

### Creating an Activity

Activities represent a single screen in your app.

```kotlin
// MainActivity.kt
package com.example.myapp

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    
    private lateinit var textView: TextView
    private lateinit var button: Button
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize views
        textView = findViewById(R.id.textView)
        button = findViewById(R.id.button)
        
        // Set click listener
        button.setOnClickListener {
            textView.text = "Button clicked!"
        }
    }
}
```

### Activity Lifecycle

```kotlin
class MyActivity : AppCompatActivity() {
    
    private val TAG = "MyActivity"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate called")
        setContentView(R.layout.activity_my)
        
        // Restore saved state
        savedInstanceState?.let {
            val savedText = it.getString("saved_text")
            textView.text = savedText
        }
    }
    
    override fun onStart() {
        super.onStart()
        Log.d(TAG, "onStart called")
        // Activity becoming visible
    }
    
    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume called")
        // Activity in foreground, user can interact
        // Start animations, resume sensors
    }
    
    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause called")
        // Activity losing focus
        // Pause animations, release sensors
    }
    
    override fun onStop() {
        super.onStop()
        Log.d(TAG, "onStop called")
        // Activity no longer visible
        // Release heavy resources
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called")
        // Activity being destroyed
        // Final cleanup
    }
    
    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // Save state before activity is killed
        outState.putString("saved_text", textView.text.toString())
    }
}
```

### Registering Activities

```xml
<!-- AndroidManifest.xml -->
<application>
    <!-- Launcher Activity -->
    <activity
        android:name=".MainActivity"
        android:exported="true">
        <intent-filter>
            <action android:name="android.intent.action.MAIN" />
            <category android:name="android.intent.category.LAUNCHER" />
        </intent-filter>
    </activity>
    
    <!-- Other Activities -->
    <activity
        android:name=".SecondActivity"
        android:label="@string/second_activity_title"
        android:parentActivityName=".MainActivity" />
</application>
```

## Intents

### Explicit Intents

Used to start specific components within your app:

```kotlin
// Start another activity
val intent = Intent(this, SecondActivity::class.java)
startActivity(intent)

// Pass data to activity
val intent = Intent(this, DetailActivity::class.java).apply {
    putExtra("USER_ID", 12345)
    putExtra("USERNAME", "john_doe")
    putExtra("USER_DATA", userData) // Parcelable or Serializable
}
startActivity(intent)

// Receive data in target activity
class DetailActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        val userId = intent.getIntExtra("USER_ID", -1)
        val username = intent.getStringExtra("USERNAME")
        val userData = intent.getParcelableExtra<UserData>("USER_DATA")
    }
}
```

### Activity Results

```kotlin
// Modern approach using Activity Result API
class MainActivity : AppCompatActivity() {
    
    private val getContent = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data?.getStringExtra("RESULT_DATA")
            // Handle result
        }
    }
    
    private fun launchSecondActivity() {
        val intent = Intent(this, SecondActivity::class.java)
        getContent.launch(intent)
    }
}

// Return result from activity
class SecondActivity : AppCompatActivity() {
    
    private fun returnResult() {
        val resultIntent = Intent().apply {
            putExtra("RESULT_DATA", "Some result")
        }
        setResult(Activity.RESULT_OK, resultIntent)
        finish()
    }
}
```

### Implicit Intents

Used to request actions from other apps:

```kotlin
// Open web page
val webpage = Uri.parse("https://www.example.com")
val intent = Intent(Intent.ACTION_VIEW, webpage)
startActivity(intent)

// Make phone call
val phoneNumber = Uri.parse("tel:1234567890")
val intent = Intent(Intent.ACTION_DIAL, phoneNumber)
startActivity(intent)

// Send email
val intent = Intent(Intent.ACTION_SENDTO).apply {
    data = Uri.parse("mailto:")
    putExtra(Intent.EXTRA_EMAIL, arrayOf("recipient@example.com"))
    putExtra(Intent.EXTRA_SUBJECT, "Email subject")
    putExtra(Intent.EXTRA_TEXT, "Email body")
}
startActivity(intent)

// Share content
val shareIntent = Intent().apply {
    action = Intent.ACTION_SEND
    putExtra(Intent.EXTRA_TEXT, "Check this out!")
    type = "text/plain"
}
startActivity(Intent.createChooser(shareIntent, "Share via"))

// Take photo
val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
if (takePictureIntent.resolveActivity(packageManager) != null) {
    startActivity(takePictureIntent)
}

// Pick image from gallery
val pickPhotoIntent = Intent(Intent.ACTION_PICK, 
    MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
startActivity(pickPhotoIntent)
```

### Intent Filters

```xml
<!-- Declare activity can handle specific actions -->
<activity android:name=".ShareActivity">
    <intent-filter>
        <action android:name="android.intent.action.SEND" />
        <category android:name="android.intent.category.DEFAULT" />
        <data android:mimeType="text/plain" />
    </intent-filter>
</activity>

<!-- Handle custom URL scheme -->
<activity android:name=".DeepLinkActivity">
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data
            android:scheme="https"
            android:host="www.example.com"
            android:pathPrefix="/app" />
    </intent-filter>
</activity>
```

## Layouts

### XML Layouts

#### LinearLayout

```xml
<!-- res/layout/activity_main.xml -->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">
    
    <TextView
        android:id="@+id/titleTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="@string/title"
        android:textSize="24sp"
        android:textStyle="bold" />
    
    <EditText
        android:id="@+id/nameEditText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:hint="@string/enter_name"
        android:inputType="textPersonName" />
    
    <Button
        android:id="@+id/submitButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:layout_marginTop="16dp"
        android:text="@string/submit" />
    
</LinearLayout>
```

#### ConstraintLayout

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <TextView
        android:id="@+id/titleTextView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/title"
        android:textSize="24sp"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />
    
    <ImageView
        android:id="@+id/imageView"
        android:layout_width="200dp"
        android:layout_height="200dp"
        android:src="@drawable/placeholder"
        app:layout_constraintTop_toBottomOf="@id/titleTextView"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="24dp" />
    
    <Button
        android:id="@+id/button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/action"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginBottom="32dp" />
    
</androidx.constraintlayout.widget.ConstraintLayout>
```

#### RelativeLayout

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">
    
    <TextView
        android:id="@+id/header"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:text="@string/header"
        android:textSize="20sp" />
    
    <Button
        android:id="@+id/button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_centerInParent="true"
        android:text="@string/click_me" />
    
    <TextView
        android:id="@+id/footer"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:text="@string/footer"
        android:gravity="center" />
    
</RelativeLayout>
```

#### FrameLayout

```xml
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <!-- Background -->
    <ImageView
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:src="@drawable/background"
        android:scaleType="centerCrop" />
    
    <!-- Foreground content -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:orientation="vertical"
        android:padding="32dp">
        
        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="@string/welcome"
            android:textSize="32sp"
            android:textColor="@android:color/white" />
        
    </LinearLayout>
    
</FrameLayout>
```

### View Binding

Safer alternative to findViewById:

```kotlin
// Enable in build.gradle
android {
    buildFeatures {
        viewBinding = true
    }
}

// Usage in Activity
class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Access views directly
        binding.button.setOnClickListener {
            binding.textView.text = "Clicked!"
        }
    }
}

// Usage in Fragment
class MyFragment : Fragment() {
    
    private var _binding: FragmentMyBinding? = null
    private val binding get() = _binding!!
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMyBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
```

### RecyclerView

```kotlin
// Item layout: res/layout/item_user.xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:padding="16dp"
    android:orientation="horizontal">
    
    <TextView
        android:id="@+id/nameTextView"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:textSize="18sp" />
    
</LinearLayout>

// Data class
data class User(val id: Int, val name: String)

// Adapter
class UserAdapter(private val users: List<User>) : 
    RecyclerView.Adapter<UserAdapter.UserViewHolder>() {
    
    class UserViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val nameTextView: TextView = view.findViewById(R.id.nameTextView)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): UserViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_user, parent, false)
        return UserViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: UserViewHolder, position: Int) {
        val user = users[position]
        holder.nameTextView.text = user.name
        holder.itemView.setOnClickListener {
            // Handle click
        }
    }
    
    override fun getItemCount() = users.size
}

// Usage
val recyclerView: RecyclerView = findViewById(R.id.recyclerView)
recyclerView.layoutManager = LinearLayoutManager(this)
recyclerView.adapter = UserAdapter(userList)
```

## Fragments

```kotlin
// Fragment class
class MyFragment : Fragment() {
    
    private var _binding: FragmentMyBinding? = null
    private val binding get() = _binding!!
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMyBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        binding.button.setOnClickListener {
            // Handle click
        }
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

// Add fragment to activity
supportFragmentManager.commit {
    setReorderingAllowed(true)
    add(R.id.fragment_container, MyFragment())
}

// Replace fragment
supportFragmentManager.commit {
    setReorderingAllowed(true)
    replace(R.id.fragment_container, AnotherFragment())
    addToBackStack("transaction_name")
}

// Pass arguments to fragment
val fragment = MyFragment().apply {
    arguments = Bundle().apply {
        putString("ARG_NAME", "value")
        putInt("ARG_ID", 123)
    }
}

// Retrieve arguments in fragment
val name = arguments?.getString("ARG_NAME")
val id = arguments?.getInt("ARG_ID")
```

## Resources

### Strings

```xml
<!-- res/values/strings.xml -->
<resources>
    <string name="app_name">My App</string>
    <string name="welcome_message">Welcome, %1$s!</string>
    <string name="items_count">You have %d items</string>
    
    <plurals name="number_of_items">
        <item quantity="one">%d item</item>
        <item quantity="other">%d items</item>
    </plurals>
</resources>

<!-- Usage in code -->
val welcome = getString(R.string.welcome_message, "John")
val count = getString(R.string.items_count, 5)
val plural = resources.getQuantityString(R.plurals.number_of_items, count, count)
```

### Colors

```xml
<!-- res/values/colors.xml -->
<resources>
    <color name="purple_200">#FFBB86FC</color>
    <color name="purple_500">#FF6200EE</color>
    <color name="purple_700">#FF3700B3</color>
    <color name="teal_200">#FF03DAC5</color>
    <color name="black">#FF000000</color>
    <color name="white">#FFFFFFFF</color>
</resources>
```

### Dimensions

```xml
<!-- res/values/dimens.xml -->
<resources>
    <dimen name="padding_small">8dp</dimen>
    <dimen name="padding_medium">16dp</dimen>
    <dimen name="padding_large">24dp</dimen>
    <dimen name="text_size_small">12sp</dimen>
    <dimen name="text_size_medium">16sp</dimen>
    <dimen name="text_size_large">20sp</dimen>
</resources>
```

### Styles and Themes

```xml
<!-- res/values/styles.xml -->
<resources>
    <!-- Base application theme -->
    <style name="AppTheme" parent="Theme.MaterialComponents.DayNight">
        <item name="colorPrimary">@color/purple_500</item>
        <item name="colorPrimaryVariant">@color/purple_700</item>
        <item name="colorOnPrimary">@color/white</item>
        <item name="colorSecondary">@color/teal_200</item>
    </style>
    
    <!-- Custom style -->
    <style name="CustomButton" parent="Widget.MaterialComponents.Button">
        <item name="android:textColor">@color/white</item>
        <item name="backgroundTint">@color/purple_500</item>
        <item name="cornerRadius">8dp</item>
    </style>
</resources>
```

## Debugging

### Logcat

```kotlin
import android.util.Log

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Different log levels
        Log.v(TAG, "Verbose message")   // Verbose
        Log.d(TAG, "Debug message")     // Debug
        Log.i(TAG, "Info message")      // Info
        Log.w(TAG, "Warning message")   // Warning
        Log.e(TAG, "Error message")     // Error
        
        // Log with exception
        try {
            // Code that might throw
        } catch (e: Exception) {
            Log.e(TAG, "Error occurred", e)
        }
    }
}
```

### Breakpoints

1. Click left margin in code editor to set breakpoint
2. Run app in Debug mode (Shift + F9)
3. Use Debug panel to step through code:
   - Step Over (F8)
   - Step Into (F7)
   - Step Out (Shift + F8)
   - Resume (F9)

### Layout Inspector

```
Tools > Layout Inspector

- View hierarchy in real-time
- Inspect view properties
- Debug rendering issues
```

## Build and Deploy

### Building APK

```bash
# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease

# Install debug APK
./gradlew installDebug

# APK location
# Debug: app/build/outputs/apk/debug/app-debug.apk
# Release: app/build/outputs/apk/release/app-release.apk
```

### Signing Configuration

```gradle
// app/build.gradle
android {
    signingConfigs {
        release {
            storeFile file("release-keystore.jks")
            storePassword "your_store_password"
            keyAlias "your_key_alias"
            keyPassword "your_key_password"
        }
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'),
                         'proguard-rules.pro'
        }
    }
}
```

### ProGuard Rules

```proguard
# app/proguard-rules.pro

# Keep model classes
-keep class com.example.app.models.** { *; }

# Keep Parcelable implementations
-keep class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.** { *; }

# Retrofit
-keepattributes Signature
-keepattributes Exceptions
-keep class retrofit2.** { *; }
```

## Best Practices

1. **Use ConstraintLayout** for complex, flat hierarchies
2. **Implement View Binding** instead of findViewById
3. **Follow Material Design** guidelines
4. **Use string resources** instead of hardcoded strings
5. **Handle configuration changes** properly
6. **Use Fragments** for reusable UI components
7. **Implement proper error handling** and user feedback
8. **Test on multiple devices** and screen sizes
9. **Optimize layouts** to reduce overdraw
10. **Use Android Architecture Components** (ViewModel, LiveData, Room)

## Related Resources

- [Android Overview](README.md)
- [Android Internals](internals.md)
- [ADB Commands](adb.md)
- [Official Android Documentation](https://developer.android.com/)
