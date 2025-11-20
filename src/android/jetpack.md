# Android Jetpack

## Overview

Android Jetpack is a suite of libraries, tools, and architectural guidance designed to help developers build high-quality Android apps more easily. It provides solutions for common development challenges like lifecycle management, background processing, navigation, database management, and UI construction.

Jetpack libraries are unbundled from the Android platform, meaning they can be updated independently of the OS version. They're built on modern design principles like separation of concerns, testability, and reduced boilerplate. All Jetpack components work together seamlessly while remaining individually adoptable.

This guide focuses on a **Compose-first approach**, reflecting modern Android development practices, while also covering integration with traditional View-based systems where relevant.

## Core Architecture Components

### ViewModel

ViewModels store and manage UI-related data in a lifecycle-conscious way, surviving configuration changes like screen rotations.

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
}
```

**Basic ViewModel:**

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class User(val id: String, val name: String, val email: String)

class UserViewModel(
    private val repository: UserRepository
) : ViewModel() {
    // StateFlow for Compose (preferred over LiveData)
    private val _uiState = MutableStateFlow<UiState>(UiState.Loading)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    sealed class UiState {
        object Loading : UiState()
        data class Success(val users: List<User>) : UiState()
        data class Error(val message: String) : UiState()
    }

    init {
        loadUsers()
    }

    fun loadUsers() {
        viewModelScope.launch {
            _uiState.value = UiState.Loading
            try {
                val users = repository.getUsers()
                _uiState.value = UiState.Success(users)
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "Unknown error")
            }
        }
    }

    fun refreshUsers() {
        loadUsers()
    }

    override fun onCleared() {
        super.onCleared()
        // Clean up resources if needed
    }
}
```

**Usage in Compose:**

```kotlin
@Composable
fun UserScreen(
    viewModel: UserViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    when (uiState) {
        is UserViewModel.UiState.Loading -> {
            CircularProgressIndicator()
        }
        is UserViewModel.UiState.Success -> {
            val users = (uiState as UserViewModel.UiState.Success).users
            UserList(users = users, onRefresh = { viewModel.refreshUsers() })
        }
        is UserViewModel.UiState.Error -> {
            val message = (uiState as UserViewModel.UiState.Error).message
            ErrorView(message = message, onRetry = { viewModel.loadUsers() })
        }
    }
}
```

### LiveData & StateFlow

**LiveData** is lifecycle-aware and useful for View-based UIs. **StateFlow** is preferred for Compose as it integrates better with coroutines and Compose's recomposition system.

```kotlin
// LiveData example (traditional Views)
class LegacyViewModel : ViewModel() {
    private val _users = MutableLiveData<List<User>>()
    val users: LiveData<List<User>> = _users

    fun loadUsers() {
        viewModelScope.launch {
            _users.value = repository.getUsers()
        }
    }
}

// StateFlow example (Compose-first)
class ModernViewModel : ViewModel() {
    private val _users = MutableStateFlow<List<User>>(emptyList())
    val users: StateFlow<List<User>> = _users.asStateFlow()

    fun loadUsers() {
        viewModelScope.launch {
            _users.value = repository.getUsers()
        }
    }
}
```

**Converting LiveData to Compose State:**

```kotlin
@Composable
fun ObserveLiveData(viewModel: LegacyViewModel) {
    val users by viewModel.users.observeAsState(initial = emptyList())
    UserList(users = users)
}
```

### Lifecycle

Lifecycle-aware components perform actions in response to lifecycle changes of activities and fragments.

```kotlin
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner

class AppLifecycleObserver : DefaultLifecycleObserver {
    override fun onStart(owner: LifecycleOwner) {
        // App moved to foreground
        Log.d("AppLifecycle", "App in foreground")
    }

    override fun onStop(owner: LifecycleOwner) {
        // App moved to background
        Log.d("AppLifecycle", "App in background")
    }
}

// In Application class
class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        ProcessLifecycleOwner.get().lifecycle.addObserver(AppLifecycleObserver())
    }
}
```

**Lifecycle-aware effects in Compose:**

```kotlin
@Composable
fun LocationScreen() {
    val lifecycleOwner = LocalLifecycleOwner.current

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> {
                    // Start location updates
                }
                Lifecycle.Event.ON_STOP -> {
                    // Stop location updates
                }
                else -> {}
            }
        }

        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
}
```

### SavedStateHandle

Preserves state across process death and recreation.

```kotlin
class SavedStateViewModel(
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {
    // Automatically saved and restored
    var searchQuery: String
        get() = savedStateHandle.get<String>("search_query") ?: ""
        set(value) {
            savedStateHandle["search_query"] = value
        }

    // For StateFlow with SavedStateHandle
    val queryFlow: StateFlow<String> = savedStateHandle.getStateFlow("search_query", "")

    fun updateQuery(query: String) {
        savedStateHandle["search_query"] = query
    }
}
```

## Jetpack Compose

### Basics

Compose is Android's modern declarative UI toolkit that simplifies and accelerates UI development.

```kotlin
// build.gradle.kts
dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.8.2")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
```

**Basic Composables:**

```kotlin
@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello, $name!",
        modifier = modifier.padding(16.dp),
        style = MaterialTheme.typography.headlineMedium
    )
}

@Composable
fun UserCard(user: User, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .clickable { onClick() },
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .background(MaterialTheme.colorScheme.primary, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = user.name.first().toString(),
                    color = MaterialTheme.colorScheme.onPrimary,
                    style = MaterialTheme.typography.titleLarge
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            // User info
            Column {
                Text(
                    text = user.name,
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = user.email,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun UserCardPreview() {
    MaterialTheme {
        UserCard(
            user = User("1", "John Doe", "john@example.com"),
            onClick = {}
        )
    }
}
```

### State Management

Compose recomposes UI when state changes. Understanding state management is crucial.

**remember and rememberSaveable:**

```kotlin
@Composable
fun Counter() {
    // State survives recomposition but NOT configuration changes
    var count by remember { mutableStateOf(0) }

    // State survives both recomposition AND configuration changes
    var savedCount by rememberSaveable { mutableStateOf(0) }

    Column {
        Text("Count: $count")
        Button(onClick = { count++ }) {
            Text("Increment")
        }

        Text("Saved Count: $savedCount")
        Button(onClick = { savedCount++ }) {
            Text("Increment Saved")
        }
    }
}
```

**State Hoisting:**

```kotlin
// Stateless composable - reusable and testable
@Composable
fun SearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    TextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = modifier.fillMaxWidth(),
        placeholder = { Text("Search...") },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) }
    )
}

// Stateful wrapper
@Composable
fun SearchScreen() {
    var searchQuery by rememberSaveable { mutableStateOf("") }
    var searchResults by remember { mutableStateOf<List<User>>(emptyList()) }

    Column {
        SearchBar(
            query = searchQuery,
            onQueryChange = {
                searchQuery = it
                // Perform search
            }
        )

        LazyColumn {
            items(searchResults) { user ->
                UserCard(user = user, onClick = {})
            }
        }
    }
}
```

**Side Effects:**

```kotlin
@Composable
fun AnalyticsScreen(screenName: String) {
    // LaunchedEffect: Run suspend functions tied to composable lifecycle
    LaunchedEffect(screenName) {
        // Called when screenName changes
        analytics.logScreenView(screenName)
    }

    // DisposableEffect: Clean up when key changes or composable leaves composition
    DisposableEffect(Unit) {
        val listener = setupListener()
        onDispose {
            listener.remove()
        }
    }

    // SideEffect: Publish Compose state to non-Compose code
    SideEffect {
        // Called after every successful recomposition
        nonComposeObject.state = currentState
    }
}

@Composable
fun DataLoader(viewModel: DataViewModel) {
    val data by viewModel.data.collectAsStateWithLifecycle()

    // LaunchedEffect with proper cancellation
    LaunchedEffect(Unit) {
        viewModel.loadData()
    }

    DataDisplay(data)
}
```

### Layouts

**Column, Row, Box:**

```kotlin
@Composable
fun LayoutExamples() {
    // Column: Vertical arrangement
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Item 1")
        Text("Item 2")
        Text("Item 3")
    }

    // Row: Horizontal arrangement
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Default.Home, contentDescription = null)
        Text("Home")
        Icon(Icons.Default.KeyboardArrowRight, contentDescription = null)
    }

    // Box: Stack elements
    Box(modifier = Modifier.fillMaxSize()) {
        Image(
            painter = painterResource(R.drawable.background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize()
        )
        Text(
            text = "Overlay Text",
            modifier = Modifier.align(Alignment.Center),
            color = Color.White
        )
    }
}
```

**LazyColumn and LazyRow:**

```kotlin
@Composable
fun UserList(users: List<User>, onUserClick: (User) -> Unit) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(
            items = users,
            key = { it.id } // Important for efficient recomposition
        ) { user ->
            UserCard(user = user, onClick = { onUserClick(user) })
        }
    }
}

@Composable
fun CategoryList(categories: List<Category>) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(horizontal = 16.dp)
    ) {
        items(categories) { category ->
            CategoryChip(category)
        }
    }
}
```

**LazyVerticalGrid:**

```kotlin
@Composable
fun PhotoGrid(photos: List<Photo>) {
    LazyVerticalGrid(
        columns = GridCells.Adaptive(minSize = 128.dp),
        contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(photos, key = { it.id }) { photo ->
            AsyncImage(
                model = photo.url,
                contentDescription = photo.description,
                modifier = Modifier
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Crop
            )
        }
    }
}
```

**Scaffold:**

```kotlin
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(navController: NavController) {
    var selectedItem by remember { mutableStateOf(0) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("My App") },
                actions = {
                    IconButton(onClick = { /* Search */ }) {
                        Icon(Icons.Default.Search, contentDescription = "Search")
                    }
                }
            )
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Home, contentDescription = null) },
                    label = { Text("Home") },
                    selected = selectedItem == 0,
                    onClick = { selectedItem = 0 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Default.Person, contentDescription = null) },
                    label = { Text("Profile") },
                    selected = selectedItem == 1,
                    onClick = { selectedItem = 1 }
                )
            }
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { /* Add item */ }) {
                Icon(Icons.Default.Add, contentDescription = "Add")
            }
        }
    ) { paddingValues ->
        // Content
        Box(modifier = Modifier.padding(paddingValues)) {
            when (selectedItem) {
                0 -> HomeContent()
                1 -> ProfileContent()
            }
        }
    }
}
```

### Navigation

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.navigation:navigation-compose:2.7.7")
}
```

**Basic Navigation:**

```kotlin
sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Details : Screen("details/{userId}") {
        fun createRoute(userId: String) = "details/$userId"
    }
    object Settings : Screen("settings")
}

@Composable
fun AppNavigation() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Screen.Home.route
    ) {
        composable(Screen.Home.route) {
            HomeScreen(
                onUserClick = { userId ->
                    navController.navigate(Screen.Details.createRoute(userId))
                }
            )
        }

        composable(
            route = Screen.Details.route,
            arguments = listOf(
                navArgument("userId") { type = NavType.StringType }
            )
        ) { backStackEntry ->
            val userId = backStackEntry.arguments?.getString("userId")
            DetailsScreen(
                userId = userId,
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(Screen.Settings.route) {
            SettingsScreen()
        }
    }
}
```

**Type-Safe Navigation with Arguments:**

```kotlin
@Composable
fun DetailsScreen(
    userId: String?,
    viewModel: DetailsViewModel = viewModel(
        factory = DetailsViewModel.Factory(userId ?: "")
    )
) {
    val user by viewModel.user.collectAsStateWithLifecycle()

    user?.let {
        UserDetails(user = it)
    }
}
```

**Navigation with Bottom Nav:**

```kotlin
@Composable
fun MainScreen() {
    val navController = rememberNavController()
    var selectedTab by remember { mutableStateOf(0) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = {
                        selectedTab = 0
                        navController.navigate(Screen.Home.route) {
                            popUpTo(navController.graph.startDestinationId)
                            launchSingleTop = true
                        }
                    },
                    icon = { Icon(Icons.Default.Home, contentDescription = null) },
                    label = { Text("Home") }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = {
                        selectedTab = 1
                        navController.navigate(Screen.Settings.route) {
                            popUpTo(navController.graph.startDestinationId)
                            launchSingleTop = true
                        }
                    },
                    icon = { Icon(Icons.Default.Settings, contentDescription = null) },
                    label = { Text("Settings") }
                )
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(paddingValues)
        ) {
            composable(Screen.Home.route) { HomeScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
        }
    }
}
```

### Theming

**Material 3 Theme:**

```kotlin
// ui/theme/Color.kt
val md_theme_light_primary = Color(0xFF6750A4)
val md_theme_light_onPrimary = Color(0xFFFFFFFF)
val md_theme_light_primaryContainer = Color(0xFFEADDFF)
// ... more colors

val md_theme_dark_primary = Color(0xFFD0BCFF)
val md_theme_dark_onPrimary = Color(0xFF381E72)
// ... more colors

// ui/theme/Theme.kt
private val LightColorScheme = lightColorScheme(
    primary = md_theme_light_primary,
    onPrimary = md_theme_light_onPrimary,
    primaryContainer = md_theme_light_primaryContainer,
    // ... more colors
)

private val DarkColorScheme = darkColorScheme(
    primary = md_theme_dark_primary,
    onPrimary = md_theme_dark_onPrimary,
    // ... more colors
)

@Composable
fun MyAppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true, // Android 12+
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context)
            else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
```

**Custom Typography:**

```kotlin
// ui/theme/Type.kt
val Typography = Typography(
    displayLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 57.sp,
        lineHeight = 64.sp,
        letterSpacing = (-0.25).sp
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Bold,
        fontSize = 28.sp,
        lineHeight = 36.sp
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    )
)
```

### Animation

**Simple Animations:**

```kotlin
@Composable
fun AnimatedCounter(count: Int) {
    // Animate size changes
    Text(
        text = count.toString(),
        modifier = Modifier.animateContentSize(),
        style = MaterialTheme.typography.displayLarge
    )
}

@Composable
fun ExpandableCard(expanded: Boolean, onToggle: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize() // Smooth size transition
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.clickable { onToggle() },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Header", modifier = Modifier.weight(1f))
                Icon(
                    imageVector = if (expanded) Icons.Default.KeyboardArrowUp
                                  else Icons.Default.KeyboardArrowDown,
                    contentDescription = null,
                    modifier = Modifier.rotate(
                        animateFloatAsState(if (expanded) 180f else 0f).value
                    )
                )
            }

            AnimatedVisibility(visible = expanded) {
                Text(
                    "Expandable content goes here...",
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        }
    }
}
```

**Transitions:**

```kotlin
@Composable
fun FadeInOutButton(visible: Boolean) {
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(animationSpec = tween(300)) + slideInVertically(),
        exit = fadeOut(animationSpec = tween(300)) + slideOutVertically()
    ) {
        Button(onClick = {}) {
            Text("Click Me")
        }
    }
}
```

### Compose-View Interop

**Using Compose in a Fragment:**

```kotlin
class MyFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return ComposeView(requireContext()).apply {
            setContent {
                MyAppTheme {
                    MyComposableScreen()
                }
            }
        }
    }
}
```

**Using Views in Compose:**

```kotlin
@Composable
fun AndroidViewExample() {
    AndroidView(
        factory = { context ->
            // Create and return the View
            TextView(context).apply {
                text = "This is a traditional View"
                textSize = 18f
            }
        },
        update = { view ->
            // Update the view
            view.text = "Updated text"
        }
    )
}
```

## Persistence

### Room Database

```kotlin
// build.gradle.kts
plugins {
    id("com.google.devtools.ksp") version "1.9.22-1.0.17"
}

dependencies {
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")
}
```

**Entity:**

```kotlin
@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "name") val name: String,
    @ColumnInfo(name = "email") val email: String,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "posts",
    foreignKeys = [
        ForeignKey(
            entity = UserEntity::class,
            parentColumns = ["id"],
            childColumns = ["user_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["user_id"])]
)
data class PostEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "user_id") val userId: String,
    @ColumnInfo(name = "title") val title: String,
    @ColumnInfo(name = "content") val content: String
)
```

**DAO:**

```kotlin
@Dao
interface UserDao {
    @Query("SELECT * FROM users")
    fun getAllUsers(): Flow<List<UserEntity>>

    @Query("SELECT * FROM users WHERE id = :userId")
    suspend fun getUserById(userId: String): UserEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: UserEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUsers(users: List<UserEntity>)

    @Update
    suspend fun updateUser(user: UserEntity)

    @Delete
    suspend fun deleteUser(user: UserEntity)

    @Query("DELETE FROM users WHERE id = :userId")
    suspend fun deleteUserById(userId: String)

    // Relation query
    @Transaction
    @Query("SELECT * FROM users WHERE id = :userId")
    suspend fun getUserWithPosts(userId: String): UserWithPosts?
}

data class UserWithPosts(
    @Embedded val user: UserEntity,
    @Relation(
        parentColumn = "id",
        entityColumn = "user_id"
    )
    val posts: List<PostEntity>
)
```

**Database:**

```kotlin
@Database(
    entities = [UserEntity::class, PostEntity::class],
    version = 2,
    exportSchema = true
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun postDao(): PostDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "app_database"
                )
                    .addMigrations(MIGRATION_1_2)
                    .build()
                INSTANCE = instance
                instance
            }
        }

        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL(
                    "ALTER TABLE users ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0"
                )
            }
        }
    }
}
```

**Repository Pattern:**

```kotlin
class UserRepository(private val userDao: UserDao) {
    val allUsers: Flow<List<User>> = userDao.getAllUsers()
        .map { entities -> entities.map { it.toUser() } }

    suspend fun getUserById(userId: String): User? {
        return userDao.getUserById(userId)?.toUser()
    }

    suspend fun insertUser(user: User) {
        userDao.insertUser(user.toEntity())
    }

    suspend fun deleteUser(userId: String) {
        userDao.deleteUserById(userId)
    }
}

// Extension functions for mapping
fun UserEntity.toUser() = User(id, name, email)
fun User.toEntity() = UserEntity(id, name, email)
```

### DataStore

Modern replacement for SharedPreferences with type safety and coroutines support.

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.datastore:datastore-preferences:1.0.0")
    // For Proto DataStore
    implementation("androidx.datastore:datastore:1.0.0")
}
```

**Preferences DataStore:**

```kotlin
// Create DataStore
val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

// Keys
object PreferencesKeys {
    val THEME_MODE = stringPreferencesKey("theme_mode")
    val NOTIFICATIONS_ENABLED = booleanPreferencesKey("notifications_enabled")
    val USER_NAME = stringPreferencesKey("user_name")
}

// Settings Manager
class SettingsManager(private val dataStore: DataStore<Preferences>) {
    val themeMode: Flow<String> = dataStore.data
        .map { preferences ->
            preferences[PreferencesKeys.THEME_MODE] ?: "system"
        }

    val notificationsEnabled: Flow<Boolean> = dataStore.data
        .map { preferences ->
            preferences[PreferencesKeys.NOTIFICATIONS_ENABLED] ?: true
        }

    suspend fun setThemeMode(mode: String) {
        dataStore.edit { preferences ->
            preferences[PreferencesKeys.THEME_MODE] = mode
        }
    }

    suspend fun setNotificationsEnabled(enabled: Boolean) {
        dataStore.edit { preferences ->
            preferences[PreferencesKeys.NOTIFICATIONS_ENABLED] = enabled
        }
    }
}

// Usage in Compose
@Composable
fun SettingsScreen(settingsManager: SettingsManager) {
    val themeMode by settingsManager.themeMode.collectAsStateWithLifecycle(initialValue = "system")
    val notificationsEnabled by settingsManager.notificationsEnabled
        .collectAsStateWithLifecycle(initialValue = true)

    Column {
        // Theme selector
        DropdownMenu(
            expanded = showThemeMenu,
            onDismissRequest = { showThemeMenu = false }
        ) {
            listOf("light", "dark", "system").forEach { mode ->
                DropdownMenuItem(
                    text = { Text(mode.capitalize()) },
                    onClick = {
                        coroutineScope.launch {
                            settingsManager.setThemeMode(mode)
                        }
                        showThemeMenu = false
                    }
                )
            }
        }

        // Notifications toggle
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("Notifications")
            Switch(
                checked = notificationsEnabled,
                onCheckedChange = { enabled ->
                    coroutineScope.launch {
                        settingsManager.setNotificationsEnabled(enabled)
                    }
                }
            )
        }
    }
}
```

## Background Work

### WorkManager

Handles deferrable, guaranteed background work.

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.work:work-runtime-ktx:2.9.0")
}
```

**Creating a Worker:**

```kotlin
class DataSyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        return try {
            // Get input data
            val userId = inputData.getString("user_id") ?: return Result.failure()

            // Perform work
            val repository = (applicationContext as MyApplication).repository
            repository.syncUserData(userId)

            // Set progress
            setProgress(workDataOf("progress" to 50))

            // Return success with output data
            val outputData = workDataOf("sync_time" to System.currentTimeMillis())
            Result.success(outputData)
        } catch (e: Exception) {
            // Retry if it's a temporary issue
            if (runAttemptCount < 3) {
                Result.retry()
            } else {
                Result.failure()
            }
        }
    }
}
```

**Scheduling Work:**

```kotlin
class WorkScheduler(private val workManager: WorkManager) {

    // One-time work
    fun scheduleDataSync(userId: String) {
        val inputData = workDataOf("user_id" to userId)

        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .setRequiresBatteryNotLow(true)
            .build()

        val syncRequest = OneTimeWorkRequestBuilder<DataSyncWorker>()
            .setInputData(inputData)
            .setConstraints(constraints)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                Duration.ofMinutes(1)
            )
            .build()

        workManager.enqueueUniqueWork(
            "data_sync_$userId",
            ExistingWorkPolicy.REPLACE,
            syncRequest
        )
    }

    // Periodic work
    fun schedulePeriodicSync() {
        val syncRequest = PeriodicWorkRequestBuilder<DataSyncWorker>(
            repeatInterval = 15,
            repeatIntervalTimeUnit = TimeUnit.MINUTES,
            flexTimeInterval = 5,
            flexTimeIntervalUnit = TimeUnit.MINUTES
        )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()
            )
            .build()

        workManager.enqueueUniquePeriodicWork(
            "periodic_sync",
            ExistingPeriodicWorkPolicy.KEEP,
            syncRequest
        )
    }

    // Work chaining
    fun scheduleDataPipeline() {
        val fetchWork = OneTimeWorkRequestBuilder<FetchDataWorker>().build()
        val processWork = OneTimeWorkRequestBuilder<ProcessDataWorker>().build()
        val uploadWork = OneTimeWorkRequestBuilder<UploadDataWorker>().build()

        workManager.beginWith(fetchWork)
            .then(processWork)
            .then(uploadWork)
            .enqueue()
    }
}
```

**Observing Work:**

```kotlin
@Composable
fun WorkProgressScreen(workManager: WorkManager, workId: UUID) {
    val workInfo by workManager.getWorkInfoByIdFlow(workId)
        .collectAsStateWithLifecycle(initialValue = null)

    workInfo?.let { info ->
        when (info.state) {
            WorkInfo.State.ENQUEUED -> Text("Work enqueued")
            WorkInfo.State.RUNNING -> {
                val progress = info.progress.getInt("progress", 0)
                LinearProgressIndicator(progress = progress / 100f)
            }
            WorkInfo.State.SUCCEEDED -> {
                val syncTime = info.outputData.getLong("sync_time", 0)
                Text("Sync completed at ${Date(syncTime)}")
            }
            WorkInfo.State.FAILED -> Text("Work failed")
            WorkInfo.State.CANCELLED -> Text("Work cancelled")
            else -> {}
        }
    }
}
```

### Paging 3

Efficiently load large datasets page by page.

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.paging:paging-runtime:3.2.1")
    implementation("androidx.paging:paging-compose:3.2.1")
}
```

**PagingSource:**

```kotlin
class UserPagingSource(
    private val api: ApiService
) : PagingSource<Int, User>() {

    override suspend fun load(params: LoadParams<Int>): LoadResult<Int, User> {
        return try {
            val page = params.key ?: 1
            val response = api.getUsers(page = page, pageSize = params.loadSize)

            LoadResult.Page(
                data = response.users,
                prevKey = if (page == 1) null else page - 1,
                nextKey = if (response.users.isEmpty()) null else page + 1
            )
        } catch (e: Exception) {
            LoadResult.Error(e)
        }
    }

    override fun getRefreshKey(state: PagingState<Int, User>): Int? {
        return state.anchorPosition?.let { anchorPosition ->
            state.closestPageToPosition(anchorPosition)?.prevKey?.plus(1)
                ?: state.closestPageToPosition(anchorPosition)?.nextKey?.minus(1)
        }
    }
}
```

**Repository with Paging:**

```kotlin
class UserRepository(private val api: ApiService) {
    fun getUsersPager(): Flow<PagingData<User>> {
        return Pager(
            config = PagingConfig(
                pageSize = 20,
                enablePlaceholders = false,
                prefetchDistance = 5
            ),
            pagingSourceFactory = { UserPagingSource(api) }
        ).flow
    }
}
```

**Usage in Compose:**

```kotlin
@Composable
fun UserListScreen(viewModel: UserViewModel = viewModel()) {
    val userPagingItems = viewModel.usersPager.collectAsLazyPagingItems()

    LazyColumn {
        items(
            count = userPagingItems.itemCount,
            key = userPagingItems.itemKey { it.id }
        ) { index ->
            val user = userPagingItems[index]
            user?.let {
                UserCard(user = it, onClick = {})
            }
        }

        // Loading state
        userPagingItems.apply {
            when {
                loadState.refresh is LoadState.Loading -> {
                    item {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator()
                        }
                    }
                }
                loadState.append is LoadState.Loading -> {
                    item {
                        CircularProgressIndicator(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        )
                    }
                }
                loadState.refresh is LoadState.Error -> {
                    val error = (loadState.refresh as LoadState.Error).error
                    item {
                        ErrorView(
                            message = error.message ?: "Unknown error",
                            onRetry = { userPagingItems.retry() }
                        )
                    }
                }
            }
        }
    }
}
```

## Dependency Injection with Hilt

```kotlin
// build.gradle.kts (project)
plugins {
    id("com.google.dagger.hilt.android") version "2.50" apply false
}

// build.gradle.kts (app)
plugins {
    id("com.google.dagger.hilt.android")
    id("com.google.devtools.ksp")
}

dependencies {
    implementation("com.google.dagger:hilt-android:2.50")
    ksp("com.google.dagger:hilt-compiler:2.50")
    implementation("androidx.hilt:hilt-navigation-compose:1.1.0")
}
```

**Setup:**

```kotlin
@HiltAndroidApp
class MyApplication : Application()

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyAppTheme {
                AppNavigation()
            }
        }
    }
}
```

**Modules:**

```kotlin
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return AppDatabase.getInstance(context)
    }

    @Provides
    fun provideUserDao(database: AppDatabase): UserDao {
        return database.userDao()
    }

    @Provides
    @Singleton
    fun provideRetrofit(): Retrofit {
        return Retrofit.Builder()
            .baseUrl("https://api.example.com")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    @Provides
    @Singleton
    fun provideApiService(retrofit: Retrofit): ApiService {
        return retrofit.create(ApiService::class.java)
    }
}

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindUserRepository(
        userRepositoryImpl: UserRepositoryImpl
    ): UserRepository
}
```

**ViewModel with Hilt:**

```kotlin
@HiltViewModel
class UserViewModel @Inject constructor(
    private val repository: UserRepository,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    val users = repository.getAllUsers()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    fun addUser(user: User) {
        viewModelScope.launch {
            repository.insertUser(user)
        }
    }
}

// Usage in Compose
@Composable
fun UserScreen(
    viewModel: UserViewModel = hiltViewModel()
) {
    val users by viewModel.users.collectAsStateWithLifecycle()
    UserList(users = users)
}
```

## Best Practices

### Architecture
- Use **MVVM pattern** with ViewModel + Repository
- Keep business logic out of composables
- Use StateFlow/Flow for reactive data streams
- Implement proper dependency injection (Hilt)
- Separate data models (Entity, Domain, UI)

### State Management
- **Hoist state** to make composables reusable and testable
- Use `rememberSaveable` for state that should survive config changes
- Prefer `StateFlow` over `LiveData` in new code
- Use `collectAsStateWithLifecycle()` to collect flows in Compose
- Avoid holding state in ViewModels that shouldn't survive process death

### Compose Performance
- Use `key` parameter in LazyColumn/LazyRow for stable item identity
- Avoid unnecessary recompositions with `remember` and stable parameters
- Use `derivedStateOf` for computed state
- Prefer immutable data classes for state
- Use `LaunchedEffect` with proper keys to avoid recreation

### Database
- Always use `Flow` for observing database changes
- Implement proper migrations for schema changes
- Use transactions for multi-step operations
- Create indices for frequently queried columns
- Use foreign keys to maintain referential integrity

### Background Work
- Use WorkManager for guaranteed background work
- Use Kotlin coroutines for simple async operations
- Set appropriate constraints (network, battery, etc.)
- Implement proper retry logic with backoff
- Handle work cancellation gracefully

### Error Handling
- Use sealed classes for UI state (Loading, Success, Error)
- Always handle exceptions in coroutines
- Provide user-friendly error messages
- Implement retry mechanisms
- Log errors for debugging

### Testing
- Write unit tests for ViewModels and repositories
- Use fake repositories for testing
- Test composables with `ComposeTestRule`
- Mock dependencies with Hilt test modules
- Test navigation flows

## Common Patterns

### MVVM with Compose

```kotlin
// UI State
sealed interface UiState<out T> {
    object Loading : UiState<Nothing>
    data class Success<T>(val data: T) : UiState<T>
    data class Error(val message: String) : UiState<Nothing>
}

// ViewModel
@HiltViewModel
class ProductsViewModel @Inject constructor(
    private val repository: ProductRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow<UiState<List<Product>>>(UiState.Loading)
    val uiState: StateFlow<UiState<List<Product>>> = _uiState.asStateFlow()

    init {
        loadProducts()
    }

    fun loadProducts() {
        viewModelScope.launch {
            _uiState.value = UiState.Loading
            try {
                repository.getProducts().collect { products ->
                    _uiState.value = UiState.Success(products)
                }
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "Unknown error")
            }
        }
    }
}

// Composable
@Composable
fun ProductsScreen(viewModel: ProductsViewModel = hiltViewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    when (uiState) {
        is UiState.Loading -> LoadingView()
        is UiState.Success -> ProductList((uiState as UiState.Success).data)
        is UiState.Error -> ErrorView((uiState as UiState.Error).message)
    }
}
```

### Repository Pattern

```kotlin
interface ProductRepository {
    fun getProducts(): Flow<List<Product>>
    suspend fun getProductById(id: String): Product?
    suspend fun refreshProducts()
}

class ProductRepositoryImpl @Inject constructor(
    private val api: ApiService,
    private val dao: ProductDao,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) : ProductRepository {

    override fun getProducts(): Flow<List<Product>> {
        return dao.getAllProducts()
            .map { entities -> entities.map { it.toDomain() } }
            .flowOn(ioDispatcher)
    }

    override suspend fun getProductById(id: String): Product? {
        return withContext(ioDispatcher) {
            dao.getProductById(id)?.toDomain()
        }
    }

    override suspend fun refreshProducts() {
        withContext(ioDispatcher) {
            try {
                val products = api.getProducts()
                dao.insertAll(products.map { it.toEntity() })
            } catch (e: Exception) {
                // Handle error
                throw e
            }
        }
    }
}
```

### Pull-to-Refresh Pattern

```kotlin
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RefreshableScreen(viewModel: MyViewModel = hiltViewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val isRefreshing = uiState is UiState.Loading
    val pullRefreshState = rememberPullToRefreshState()

    if (pullRefreshState.isRefreshing) {
        LaunchedEffect(Unit) {
            viewModel.refresh()
        }
    }

    LaunchedEffect(isRefreshing) {
        if (!isRefreshing) {
            pullRefreshState.endRefresh()
        }
    }

    Box(modifier = Modifier.nestedScroll(pullRefreshState.nestedScrollConnection)) {
        LazyColumn {
            // Content
        }

        PullToRefreshContainer(
            state = pullRefreshState,
            modifier = Modifier.align(Alignment.TopCenter)
        )
    }
}
```

## Resources

### Official Documentation
- [Android Jetpack Overview](https://developer.android.com/jetpack)
- [Jetpack Compose Documentation](https://developer.android.com/jetpack/compose)
- [Modern Android Development (MAD)](https://developer.android.com/modern-android-development)
- [Architecture Guide](https://developer.android.com/topic/architecture)

### Related Notes
- [Android README](README.md) - Android overview and quick start
- [Android Internals](internals.md) - Deep dive into Android internals
- [ADB Reference](adb.md) - Android Debug Bridge commands

### Sample Projects
- [Now in Android](https://github.com/android/nowinandroid) - Official Google sample
- [Compose Samples](https://github.com/android/compose-samples) - Compose examples
