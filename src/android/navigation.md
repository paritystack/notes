# Navigation

## Overview

Navigation is how users move between screens (destinations) in an app, and how the app handles
the **back stack**, arguments, and deep links. The Jetpack **Navigation component** centralizes
this — for both classic Fragment/View apps and **Jetpack Compose**. It gives you a single
source of truth for the navigation graph, type-safe arguments, consistent Up/Back behavior, and
deep link handling.

Pairs with [App Architecture](app_architecture.md) (one-off navigation events from ViewModels)
and [Jetpack](jetpack.md).

## Core Concepts

| Concept | Meaning |
|---------|---------|
| **NavController** | Drives navigation; owns the back stack |
| **NavGraph** | The set of destinations and the routes between them |
| **Destination** | A screen (composable, fragment) or a nested graph |
| **Route / action** | Identifier (Compose: a route string/type; Views: an action id) |
| **NavHost** | The container that swaps destinations in/out |

## Navigation in Compose

### Define the graph

```kotlin
@Composable
fun AppNav(navController: NavHostController = rememberNavController()) {
    NavHost(navController, startDestination = "feed") {
        composable("feed") {
            FeedScreen(onOpenPost = { id -> navController.navigate("post/$id") })
        }
        composable(
            route = "post/{postId}",
            arguments = listOf(navArgument("postId") { type = NavType.IntType }),
        ) { backStackEntry ->
            val postId = backStackEntry.arguments?.getInt("postId") ?: return@composable
            PostScreen(postId)
        }
    }
}
```

### Type-Safe Navigation (Navigation 2.8+)

Newer Navigation supports **type-safe** routes using `@Serializable` objects instead of stringly
routes — no manual string building/parsing:

```kotlin
@Serializable data object Feed
@Serializable data class Post(val postId: Int)

NavHost(navController, startDestination = Feed) {
    composable<Feed> {
        FeedScreen(onOpenPost = { id -> navController.navigate(Post(id)) })
    }
    composable<Post> { entry ->
        val post: Post = entry.toRoute()
        PostScreen(post.postId)
    }
}
```

### Back stack control

```kotlin
// Navigate and clear up to a destination (e.g. after login)
navController.navigate("home") {
    popUpTo("login") { inclusive = true }
    launchSingleTop = true          // avoid duplicate copies on top
}

navController.popBackStack()        // go back
navController.navigateUp()          // Up (respects parent)
```

## Navigation with Fragments (Views)

Define a graph in XML and host it with a `NavHostFragment`:

```xml
<!-- res/navigation/nav_graph.xml -->
<navigation app:startDestination="@id/feedFragment">
    <fragment android:id="@+id/feedFragment" android:name="...FeedFragment">
        <action android:id="@+id/toPost" app:destination="@id/postFragment"/>
    </fragment>
    <fragment android:id="@+id/postFragment" android:name="...PostFragment">
        <argument android:name="postId" app:argType="integer"/>
    </fragment>
</navigation>
```

**Safe Args** Gradle plugin generates type-safe directions/args classes:

```kotlin
val action = FeedFragmentDirections.toPost(postId = 42)
findNavController().navigate(action)
// In PostFragment:
val args: PostFragmentArgs by navArgs()
val id = args.postId
```

## Passing Arguments

- **Small, navigation-relevant data only** (IDs, flags) — pass via route/args.
- **Don't pass large objects/lists** through navigation args; fetch from a repository/ViewModel
  using the passed ID (single source of truth — see [App Architecture](app_architecture.md)).
- Use a **scoped ViewModel** (e.g. graph-scoped) to share state across a flow of screens.

```kotlin
// Compose: ViewModel scoped to a nested nav graph (shared across its screens)
val parentEntry = remember(entry) { navController.getBackStackEntry("checkout_graph") }
val vm: CheckoutViewModel = hiltViewModel(parentEntry)
```

## Nested Graphs & Multi-Module

Group related destinations into **nested graphs** (e.g. an onboarding or checkout flow). In
modularized apps, each feature module can contribute its own navigation graph that the `:app`
module composes — keeping features decoupled (see [App Architecture](app_architecture.md)).

## Deep Links

Map external URLs/notifications to destinations.

### Compose

```kotlin
composable(
    route = "post/{postId}",
    arguments = listOf(navArgument("postId") { type = NavType.IntType }),
    deepLinks = listOf(navDeepLink { uriPattern = "https://example.com/post/{postId}" }),
) { /* ... */ }
```

### Manifest (App Links)

```xml
<activity android:name=".MainActivity">
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.VIEW"/>
        <category android:name="android.intent.category.DEFAULT"/>
        <category android:name="android.intent.category.BROWSABLE"/>
        <data android:scheme="https" android:host="example.com"/>
    </intent-filter>
</activity>
```

`autoVerify="true"` enables **Android App Links** (verified via a
`assetlinks.json` on your domain) so the link opens your app directly without a chooser.

```bash
# Test a deep link
adb shell am start -a android.intent.action.VIEW -d "https://example.com/post/42"
```

## Bottom Nav / Tabs

Multiple back stacks (one per tab) are supported so switching tabs preserves each tab's state.
In Compose, drive selection from the `NavController`'s current destination; in Views, use
`NavigationBarView.setupWithNavController`.

## Best Practices

1. **Single NavHost per navigation container**; let NavController own the back stack.
2. **Prefer type-safe routes** (Compose `@Serializable`, Views Safe Args) over raw strings.
3. **Pass IDs, not objects** — fetch data from the repository at the destination.
4. **Model navigation as one-off events** from the ViewModel (don't navigate inside recomposition).
5. **Use nested graphs** to group flows and to modularize by feature.
6. **Use verified App Links** (`autoVerify`) for HTTPS deep links; test with `adb`.
7. **Handle Up vs Back** correctly (`navigateUp` vs `popBackStack`).

## Resources

- [Navigation — Android Developers](https://developer.android.com/guide/navigation)
- [Navigation with Compose](https://developer.android.com/develop/ui/compose/navigation)
- [Type-safe navigation](https://developer.android.com/guide/navigation/design/type-safety)
- [Deep links & App Links](https://developer.android.com/training/app-links)

### Related Files

- [App Architecture](app_architecture.md) — navigation events, scoped ViewModels
- [Jetpack](jetpack.md) — Compose & lifecycle building blocks
- [App Security](app_security.md) — verifying App Links / handling untrusted deep links
