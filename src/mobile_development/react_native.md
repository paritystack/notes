# React Native

React Native is a popular JavaScript framework for building native mobile applications using React. It allows developers to use React along with native platform capabilities to build iOS and Android apps from a single codebase, with the ability to share code between platforms.

## Table of Contents
- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Core Components](#core-components)
- [Styling](#styling)
- [Navigation](#navigation)
- [State Management](#state-management)
- [API and Data Fetching](#api-and-data-fetching)
- [Native Modules](#native-modules)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Introduction

**Key Features:**
- Cross-platform development (iOS and Android)
- Native performance
- Hot reloading for fast development
- Large ecosystem and community
- Code reusability with web React
- Native API access
- Over-the-air (OTA) updates

**Use Cases:**
- Cross-platform mobile apps
- MVP development
- Apps requiring frequent updates
- Teams with JavaScript/React expertise
- Apps with shared business logic

---

## Setup and Installation

### Prerequisites

```bash
# Install Node.js (14+)
node --version
npm --version

# Install Watchman (macOS)
brew install watchman

# Install Xcode (macOS, for iOS development)
# Install Android Studio (for Android development)
```

### Create New Project

```bash
# Using React Native CLI
npx react-native init MyApp
cd MyApp

# Using Expo (recommended for beginners)
npx create-expo-app MyApp
cd MyApp
npx expo start
```

### Running the App

```bash
# React Native CLI
# iOS
npx react-native run-ios

# Android
npx react-native run-android

# Expo
npx expo start
# Then press 'i' for iOS or 'a' for Android
```

---

## Core Components

### View and Text

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Hello React Native!</Text>
      <Text style={styles.subtitle}>Welcome to mobile development</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
});
```

### Button and TouchableOpacity

```javascript
import { Button, TouchableOpacity, Alert } from 'react-native';

function MyComponent() {
  const handlePress = () => {
    Alert.alert('Button Pressed', 'You clicked the button!');
  };

  return (
    <View>
      {/* Basic Button */}
      <Button title="Click Me" onPress={handlePress} color="#007AFF" />

      {/* Custom Touchable */}
      <TouchableOpacity
        style={styles.customButton}
        onPress={handlePress}
        activeOpacity={0.7}
      >
        <Text style={styles.buttonText}>Custom Button</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  customButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
```

### TextInput

```javascript
import { useState } from 'react';
import { TextInput } from 'react-native';

function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <View style={styles.form}>
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
        autoCorrect={false}
      />

      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoCapitalize="none"
      />

      <Button
        title="Login"
        onPress={() => handleLogin(email, password)}
      />
    </View>
  );
}
```

### ScrollView and FlatList

```javascript
import { ScrollView, FlatList } from 'react-native';

// ScrollView - for small lists
function SimpleList() {
  return (
    <ScrollView>
      {items.map((item) => (
        <View key={item.id} style={styles.item}>
          <Text>{item.name}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

// FlatList - for large lists (better performance)
function OptimizedList() {
  const DATA = [
    { id: '1', title: 'Item 1' },
    { id: '2', title: 'Item 2' },
    { id: '3', title: 'Item 3' },
  ];

  const renderItem = ({ item }) => (
    <View style={styles.item}>
      <Text style={styles.title}>{item.title}</Text>
    </View>
  );

  return (
    <FlatList
      data={DATA}
      renderItem={renderItem}
      keyExtractor={(item) => item.id}
      onRefresh={() => refreshData()}
      refreshing={loading}
    />
  );
}
```

### Image

```javascript
import { Image } from 'react-native';

function ImageExample() {
  return (
    <View>
      {/* Local Image */}
      <Image
        source={require('./assets/logo.png')}
        style={{ width: 100, height: 100 }}
        resizeMode="contain"
      />

      {/* Remote Image */}
      <Image
        source={{ uri: 'https://example.com/image.jpg' }}
        style={{ width: 200, height: 200 }}
        resizeMode="cover"
      />
    </View>
  );
}
```

---

## Styling

### StyleSheet

```javascript
import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3, // Android shadow
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
});
```

### Flexbox Layout

```javascript
// Flex Direction
<View style={{ flex: 1, flexDirection: 'row' }}>
  <View style={{ flex: 1, backgroundColor: 'red' }} />
  <View style={{ flex: 2, backgroundColor: 'blue' }} />
</View>

// Justify Content
<View style={{ flex: 1, justifyContent: 'space-between' }}>
  <Text>Top</Text>
  <Text>Middle</Text>
  <Text>Bottom</Text>
</View>

// Align Items
<View style={{ flex: 1, alignItems: 'center' }}>
  <Text>Centered Horizontally</Text>
</View>
```

### Responsive Design

```javascript
import { Dimensions, Platform } from 'react-native';

const { width, height } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    width: width * 0.9, // 90% of screen width
    padding: width < 350 ? 10 : 20, // Conditional padding
  },
  image: {
    width: width - 40,
    height: (width - 40) * 0.6, // Aspect ratio
  },
  platformSpecific: {
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
      },
      android: {
        elevation: 5,
      },
    }),
  },
});
```

---

## Navigation

### React Navigation

```bash
npm install @react-navigation/native
npm install react-native-screens react-native-safe-area-context
npm install @react-navigation/stack
```

**Stack Navigator:**
```javascript
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

const Stack = createStackNavigator();

function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text>Home Screen</Text>
      <Button
        title="Go to Details"
        onPress={() => navigation.navigate('Details', { itemId: 42 })}
      />
    </View>
  );
}

function DetailsScreen({ route, navigation }) {
  const { itemId } = route.params;

  return (
    <View style={styles.container}>
      <Text>Details Screen</Text>
      <Text>Item ID: {itemId}</Text>
      <Button title="Go Back" onPress={() => navigation.goBack()} />
    </View>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: { backgroundColor: '#007AFF' },
          headerTintColor: '#fff',
          headerTitleStyle: { fontWeight: 'bold' },
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'Welcome' }}
        />
        <Stack.Screen name="Details" component={DetailsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

**Tab Navigator:**
```javascript
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;

            if (route.name === 'Home') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Settings') {
              iconName = focused ? 'settings' : 'settings-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'gray',
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
```

---

## State Management

### Context API

```javascript
import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);

  const login = async (email, password) => {
    // API call
    const response = await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json();
    setUser(data.user);
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

// Usage
function ProfileScreen() {
  const { user, logout } = useAuth();

  return (
    <View>
      <Text>Welcome, {user?.name}</Text>
      <Button title="Logout" onPress={logout} />
    </View>
  );
}
```

### Redux Toolkit

```bash
npm install @reduxjs/toolkit react-redux
```

```javascript
import { createSlice, configureStore } from '@reduxjs/toolkit';
import { Provider, useSelector, useDispatch } from 'react-redux';

// Slice
const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    decrement: (state) => {
      state.value -= 1;
    },
  },
});

export const { increment, decrement } = counterSlice.actions;

// Store
const store = configureStore({
  reducer: {
    counter: counterSlice.reducer,
  },
});

// Component
function Counter() {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    <View>
      <Text>{count}</Text>
      <Button title="+" onPress={() => dispatch(increment())} />
      <Button title="-" onPress={() => dispatch(decrement())} />
    </View>
  );
}

// App
export default function App() {
  return (
    <Provider store={store}>
      <Counter />
    </Provider>
  );
}
```

---

## API and Data Fetching

### Fetch API

```javascript
import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchUser();
  }, [userId]);

  const fetchUser = async () => {
    try {
      setLoading(true);
      const response = await fetch(`https://api.example.com/users/${userId}`);

      if (!response.ok) {
        throw new Error('Failed to fetch user');
      }

      const data = await response.json();
      setUser(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <Text>Loading...</Text>;
  if (error) return <Text>Error: {error}</Text>;

  return (
    <View>
      <Text>{user.name}</Text>
      <Text>{user.email}</Text>
    </View>
  );
}
```

### Axios

```bash
npm install axios
```

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptors
api.interceptors.request.use(
  (config) => {
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      logout();
    }
    return Promise.reject(error);
  }
);

// Usage
const fetchUsers = async () => {
  const response = await api.get('/users');
  return response.data;
};

const createUser = async (userData) => {
  const response = await api.post('/users', userData);
  return response.data;
};
```

---

## Native Modules

### Accessing Device Features

```javascript
// Camera
import { Camera } from 'expo-camera';

function CameraScreen() {
  const [hasPermission, setHasPermission] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }

  return (
    <Camera style={{ flex: 1 }} type={Camera.Constants.Type.back}>
      {/* Camera UI */}
    </Camera>
  );
}

// Location
import * as Location from 'expo-location';

const getLocation = async () => {
  const { status } = await Location.requestForegroundPermissionsAsync();

  if (status !== 'granted') {
    return;
  }

  const location = await Location.getCurrentPositionAsync({});
  console.log(location.coords);
};

// Notifications
import * as Notifications from 'expo-notifications';

const sendNotification = async () => {
  await Notifications.scheduleNotificationAsync({
    content: {
      title: "You've got mail!",
      body: 'Here is the notification body',
    },
    trigger: { seconds: 2 },
  });
};
```

---

## Performance Optimization

### Memoization

```javascript
import React, { useMemo, useCallback } from 'react';

function ExpensiveComponent({ data }) {
  // Memoize expensive calculations
  const processedData = useMemo(() => {
    return data.map((item) => {
      // Expensive operation
      return processItem(item);
    });
  }, [data]);

  // Memoize callbacks
  const handlePress = useCallback(() => {
    console.log('Button pressed');
  }, []);

  return (
    <View>
      {processedData.map((item) => (
        <Item key={item.id} data={item} onPress={handlePress} />
      ))}
    </View>
  );
}

// Memo component
const Item = React.memo(({ data, onPress }) => {
  return (
    <TouchableOpacity onPress={onPress}>
      <Text>{data.name}</Text>
    </TouchableOpacity>
  );
});
```

### FlatList Optimization

```javascript
<FlatList
  data={data}
  renderItem={renderItem}
  keyExtractor={(item) => item.id}
  // Performance optimizations
  removeClippedSubviews={true}
  maxToRenderPerBatch={10}
  updateCellsBatchingPeriod={50}
  initialNumToRender={10}
  windowSize={10}
  getItemLayout={(data, index) => ({
    length: ITEM_HEIGHT,
    offset: ITEM_HEIGHT * index,
    index,
  })}
/>
```

---

## Testing

### Jest and React Native Testing Library

```bash
npm install --save-dev @testing-library/react-native
```

```javascript
import { render, fireEvent } from '@testing-library/react-native';
import Counter from './Counter';

describe('Counter', () => {
  it('renders correctly', () => {
    const { getByText } = render(<Counter />);
    expect(getByText('Count: 0')).toBeTruthy();
  });

  it('increments counter', () => {
    const { getByText, getByTestId } = render(<Counter />);
    const button = getByTestId('increment-button');

    fireEvent.press(button);

    expect(getByText('Count: 1')).toBeTruthy();
  });
});
```

---

## Deployment

### iOS

```bash
# Build for release
npx react-native run-ios --configuration Release

# Or with Xcode
# Open ios/YourApp.xcworkspace
# Select Generic iOS Device
# Product > Archive
# Upload to App Store
```

### Android

```bash
# Generate release APK
cd android
./gradlew assembleRelease

# APK location:
# android/app/build/outputs/apk/release/app-release.apk

# Generate AAB (App Bundle)
./gradlew bundleRelease
```

---

## Resources

**Official Documentation:**
- [React Native Docs](https://reactnative.dev/)
- [Expo Documentation](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)

**Learning:**
- [React Native Express](https://www.reactnative.express/)
- [React Native School](https://www.reactnativeschool.com/)

**Tools:**
- [React Native Debugger](https://github.com/jhen0409/react-native-debugger)
- [Flipper](https://fbflipper.com/)
- [Reactotron](https://github.com/infinitered/reactotron)
