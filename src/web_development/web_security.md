# Web Security Mechanisms

> **Domain:** Web Development, Frontend, Security
> **Key Concepts:** CORS, CSP, SOP, Cookies, XSS, CSRF

While backend security (OWASP Top 10) is critical, the Browser has its own set of powerful security sandboxes. Understanding these is mandatory for any frontend or full-stack engineer.

---

## 1. SOP (Same-Origin Policy)

This is the cornerstone of web security.

*   **Rule:** A script loaded from `origin A` cannot access resources (DOM, Cookies, AJAX responses) from `origin B`.
*   **Origin Definition:** `Protocol` + `Domain` + `Port`.
    *   `https://google.com` != `http://google.com` (Protocol mismatch).
    *   `https://google.com` != `https://maps.google.com` (Subdomain mismatch).
*   **Why?** Without SOP, if you visited `evil.com`, their JS could make an AJAX call to `gmail.com` (where you are logged in) and read your emails.

---

## 2. CORS (Cross-Origin Resource Sharing)

SOP is too strict for modern apps (React on `localhost:3000` calling API on `localhost:8000`). **CORS** is the mechanism to *relax* SOP safely.

*   **Mechanism:**
    1.  Browser sends a **Preflight Request** (`OPTIONS`) to the server.
    2.  Server responds with headers: `Access-Control-Allow-Origin: https://my-app.com`.
    3.  If the domain matches, the Browser allows the actual request.
*   **Common Error:** "No 'Access-Control-Allow-Origin' header is present."
    *   *Fix:* Configure the *Server* to whitelist the *Client's* domain. You cannot fix this in client-side code.

---

## 3. CSP (Content Security Policy)

CSP is a defense-in-depth layer against **XSS (Cross-Site Scripting)**. It tells the browser exactly which sources are allowed to run code.

*   **Implementation:** An HTTP Header.
*   **Example:**
    ```http
    Content-Security-Policy: default-src 'self'; img-src https://cdn.example.com; script-src 'self' https://trusted-analytics.com
    ```
*   **Effect:** If an attacker injects `<script src="evil.com/hack.js">`, the browser blocks it because `evil.com` is not in the `script-src` whitelist.

---

## 4. Secure Cookies

Cookies are the primary vector for **Session Hijacking** and **CSRF (Cross-Site Request Forgery)**.

### 4.1. Cookie Attributes
*   **HttpOnly:** JS cannot read this cookie (`document.cookie` returns empty). Prevents XSS from stealing session tokens.
*   **Secure:** Cookie is only sent over HTTPS.
*   **SameSite:**
    *   `Strict`: Cookie is never sent on cross-site requests.
    *   `Lax`: Cookie is sent on top-level navigations (clicking a link) but not on background requests (images, iframes). Prevents CSRF.

---

## 5. XSS (Cross-Site Scripting)

*   **Stored XSS:** Malicious script is saved in the DB (e.g., a comment) and served to other users.
*   **Reflected XSS:** Malicious script is in the URL parameters.
*   **Prevention:**
    1.  **Escape Output:** React/Vue/Angular do this automatically. Never use `dangerouslySetInnerHTML` or `v-html` with user input.
    2.  **Use CSP.**
    3.  **Sanitize Input:** Use libraries like DOMPurify on the backend.

## 6. CSRF (Cross-Site Request Forgery)

*   **Attack:** `evil.com` has a hidden form that POSTs to `bank.com/transfer`. If you are logged in to `bank.com`, the browser automatically sends your cookies. Money is gone.
*   **Prevention:**
    1.  **SameSite=Strict/Lax** cookies (Modern default).
    2.  **CSRF Tokens:** A random token in the HTML form that must match the session. `evil.com` cannot read this token (due to SOP), so the request fails.
