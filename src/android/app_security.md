# App Security

## Overview

App security is about protecting user data and your app's integrity against a hostile
environment: shared devices, malware, network attackers, and even rooted/tampered devices.
This doc covers the practical app-developer toolkit — the **Android Keystore**, encrypted
storage, **biometric** authentication, **network security config** + TLS pinning, the
permissions/scoped-storage model, and integrity/anti-tamper checks.

This is the *app* perspective; for platform-level protections see
[SELinux on Android](selinux_android.md) and [Verified Boot & OTA](verified_boot_ota.md), and
for app signing see [APK/AAB Packaging & Signing](app_signing.md).

## The App Sandbox

Each app runs as its own **Linux UID** with a private data directory (`/data/data/<pkg>`) that
only that UID can read — enforced by both DAC (file perms) and MAC
([SELinux](selinux_android.md), per-app MCS categories). Don't fight the sandbox: keep secrets
in your private dir, never on shared/world-readable storage.

```bash
adb shell run-as com.example.app ls -l /data/data/com.example.app   # only on debuggable builds
```

## Android Keystore System

The **Keystore** stores cryptographic keys so the **key material never enters app memory** —
crypto operations happen in a secure container, ideally backed by hardware (**TEE** /
**StrongBox** secure element). You can use keys but not extract them.

```kotlin
val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }

val spec = KeyGenParameterSpec.Builder(
    "my_aes_key",
    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
).apply {
    setBlockModes(KeyProperties.BLOCK_MODE_GCM)
    setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
    setUserAuthenticationRequired(true)     // require unlock/biometric to use the key
    setIsStrongBoxBacked(true)              // use secure element if available
}.build()

val kg = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
kg.init(spec); kg.generateKey()
```

Hardware-backed keys can also produce **key attestation** certificates proving the key lives in
secure hardware.

## Encrypted Storage

For data at rest, use the Jetpack Security libraries (which use Keystore keys under the hood):

```kotlin
// Encrypted DataStore / SharedPreferences alternative
val masterKey = MasterKey.Builder(context)
    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build()

val prefs = EncryptedSharedPreferences.create(
    context, "secret_prefs", masterKey,
    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
)
prefs.edit().putString("token", token).apply()
```

> Note: the older `EncryptedSharedPreferences`/`EncryptedFile` APIs are deprecated in newer
> Jetpack Security releases — check current guidance and prefer DataStore with your own
> Keystore-backed encryption where appropriate. **Never hardcode keys/secrets in the APK** —
> they're trivially extracted.

## Biometric Authentication

Use the **AndroidX Biometric** library for fingerprint/face/PIN, optionally gating a Keystore
key (`setUserAuthenticationRequired(true)`) so data is only decryptable after authentication.

```kotlin
val prompt = BiometricPrompt(this, executor,
    object : BiometricPrompt.AuthenticationCallback() {
        override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
            // proceed (optionally use result.cryptoObject for crypto)
        }
    })

val info = BiometricPrompt.PromptInfo.Builder()
    .setTitle("Unlock")
    .setAllowedAuthenticators(BIOMETRIC_STRONG or DEVICE_CREDENTIAL)
    .build()
prompt.authenticate(info)
```

## Network Security

### Use HTTPS everywhere

Cleartext traffic is blocked by default on modern targets. Configure exceptions and pinning via
**Network Security Configuration** (XML), not code:

```xml
<!-- res/xml/network_security_config.xml -->
<network-security-config>
    <base-config cleartextTrafficPermitted="false"/>
    <domain-config>
        <domain includeSubdomains="true">api.example.com</domain>
        <pin-set expiration="2026-12-31">
            <pin digest="SHA-256">base64primaryPin==</pin>
            <pin digest="SHA-256">base64backupPin==</pin>   <!-- always have a backup pin -->
        </pin-set>
    </domain-config>
</network-security-config>
```

```xml
<application android:networkSecurityConfig="@xml/network_security_config" ... />
```

**Certificate pinning** prevents MITM via rogue CAs but is operationally risky — always ship a
**backup pin** and an expiration, or you can brick connectivity on cert rotation. OkHttp's
`CertificatePinner` is an alternative.

## Permissions Model

- **Install-time** permissions (normal): granted automatically.
- **Runtime** (dangerous) permissions: requested at use time, revocable by the user.
- **Special access**: e.g. overlay, all-files access — gated through Settings.

```kotlin
val launcher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
    if (granted) startCamera() else showRationaleOrFallback()
}
// Request only when needed, with rationale
launcher.launch(Manifest.permission.CAMERA)
```

Principles: **request minimally and just-in-time**, explain why, and degrade gracefully when
denied. Newer Android adds granular media permissions and one-time/approximate-location grants.

## Scoped Storage

Since Android 10/11, apps can't freely roam shared storage. Use:

- **App-specific dirs** (`getExternalFilesDir`) — no permission, auto-removed on uninstall.
- **MediaStore** for shared media; **Storage Access Framework** (`ACTION_OPEN_DOCUMENT`) for
  user-picked files. Avoid the broad `MANAGE_EXTERNAL_STORAGE` unless truly required (Play
  restricts it).

## Integrity & Anti-Tamper

- **Play Integrity API** (replaces SafetyNet Attestation): server-verifiable signals about
  device/app/account integrity. Verify the token **server-side** — client checks are bypassable.
- **R8 obfuscation** raises the bar for reverse engineering (not a security boundary by itself).
- Treat rooted/emulator detection as **signal, not guarantee** — a determined attacker controls
  the device.

```kotlin
// Request an integrity verdict; send the token to your backend to validate.
val manager = IntegrityManagerFactory.create(context)
// manager.requestIntegrityToken(...) → token → server verification
```

## Common Pitfalls

- Exporting components unintentionally — set `android:exported="false"` unless they must be
  public; protect with `signature` permissions where appropriate.
- Logging secrets (tokens, PII) to Logcat.
- Using `WebView` with `setJavaScriptEnabled(true)` + `addJavascriptInterface` on untrusted
  content (RCE risk); enable Safe Browsing and restrict loaded URLs.
- Implicit `PendingIntent`s without `FLAG_IMMUTABLE`.

## Best Practices

1. **Keep secrets out of the APK and out of logs**; store sensitive data Keystore-encrypted.
2. **Use the Keystore** (hardware/StrongBox-backed) for keys; never handle raw key material.
3. **HTTPS only**; configure via Network Security Config; pin carefully with backup pins.
4. **Request permissions minimally and just-in-time**; honor scoped storage.
5. **Gate sensitive actions with BiometricPrompt** and user-auth-required keys.
6. **Verify Play Integrity tokens server-side**; don't trust client-only checks.
7. **Mark components non-exported** by default; use `FLAG_IMMUTABLE` PendingIntents.
8. **Keep dependencies patched**; run dependency/vulnerability scanning in CI.

## Resources

- [App security best practices — Android Developers](https://developer.android.com/privacy-and-security/security-tips)
- [Android Keystore system](https://developer.android.com/privacy-and-security/keystore)
- [BiometricPrompt](https://developer.android.com/training/sign-in/biometric-auth)
- [Network security configuration](https://developer.android.com/privacy-and-security/security-config)
- [Play Integrity API](https://developer.android.com/google/play/integrity)

### Related Files

- [APK/AAB Packaging & Signing](app_signing.md) — signing identity & same-key update rule
- [SELinux on Android](selinux_android.md) — platform MAC backing the sandbox
- [Verified Boot & OTA](verified_boot_ota.md) — device-level integrity
- [Android Internals](internals.md) — UID sandbox & permission model
