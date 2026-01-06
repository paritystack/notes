# OWASP Top 10

> **Domain:** Cybersecurity, Web Development
> **Key Concepts:** Injection, XSS, Broken Access Control, SSRF

The **OWASP Top 10** is a standard awareness document for developers and web application security. It represents a broad consensus about the most critical security risks to web applications.

---

## 1. Broken Access Control (A01:2021)
Moved to #1 in 2021. This occurs when users can act outside of their intended permissions.

*   **The Attack:**
    *   **IDOR (Insecure Direct Object Reference):** A user changes the URL `app.com/invoice?id=100` to `id=101` and sees someone else's invoice.
    *   **Privilege Escalation:** A standard user forces a browse to `/admin/deleteUser`.
*   **Mitigation:**
    *   **Deny by Default:** All controllers should require auth.
    *   **Domain-Driven checks:** `invoice.owner_id == current_user.id`.
    *   Do not rely on hiding buttons in the UI; secure the API endpoint.

---

## 2. Cryptographic Failures (A02:2021)
Previously "Sensitive Data Exposure". Focuses on failures related to cryptography.

*   **The Attack:**
    *   Storing passwords in plain text or Base64.
    *   Using weak hasing algorithms (MD5, SHA1).
    *   Transmitting data over HTTP instead of HTTPS.
*   **Mitigation:**
    *   Use **Argon2id** or **bcrypt** for passwords.
    *   Enforce **TLS 1.2+** everywhere.
    *   Encrypt data at rest (AES-256).

---

## 3. Injection (A03:2021)
Includes SQL Injection (SQLi), Command Injection, and LDAP Injection.

*   **The Attack:**
    *   Input: `user_input = "'; DROP TABLE users; --"`
    *   Query: `SELECT * FROM accounts WHERE name = '' + user_input`
*   **Mitigation:**
    *   **Parameterized Queries (Prepared Statements):** The database treats input as data, not code.
    *   *Correct:* `db.execute("SELECT * FROM users WHERE id = ?", [user_id])`
    *   *Wrong:* `db.execute(f"SELECT * FROM users WHERE id = {user_id}")`

---

## 4. Insecure Design (A04:2021)
A new category focusing on risks related to design flaws. You can't "code" your way out of a bad design.

*   **Example:** A "recover password" flow that reveals whether an email exists in the system (User Enumeration).
*   **Mitigation:** Threat modeling, secure design patterns.

---

## 5. Security Misconfiguration (A05:2021)
*   **The Attack:**
    *   Leaving default accounts (`admin:admin`).
    *   Leaving debug mode enabled in production (Stack traces visible).
    *   Open cloud storage buckets (S3 world-readable).
*   **Mitigation:**
    *   Automated configuration hardening (Ansible/Terraform).
    *   Remove unused features.

---

## 6. Vulnerable and Outdated Components (A06:2021)
*   **The Attack:** Using a library (e.g., Log4j) with a known CVE.
*   **Mitigation:**
    *   Software Composition Analysis (SCA) tools (e.g., Snyk, Dependabot).
    *   Regular dependency updates.

---

## 7. Identification and Authentication Failures (A07:2021)
*   **The Attack:**
    *   Permitting weak passwords ("123456").
    *   No Rate Limiting (Brute Force).
    *   Weak session management (Session ID in URL).
*   **Mitigation:**
    *   MFA (Multi-Factor Authentication).
    *   Password complexity rules (NIST guidelines).

---

## 8. Software and Data Integrity Failures (A08:2021)
*   **The Attack:** Relying on plugins or libraries from untrusted sources without verification.
*   **Mitigation:** Verify digital signatures. Use lockfiles (`package-lock.json`) with integrity hashes.

---

## 9. Security Logging and Monitoring Failures (A09:2021)
*   **The Issue:** Breaches often take 200+ days to detect because logs are missing.
*   **Mitigation:** Log all login failures, access control failures, and server-side input validation failures.

---

## 10. Server-Side Request Forgery (SSRF) (A10:2021)
*   **The Attack:** An attacker forces the server to make a request to an internal resource.
    *   Input: `avatar_url = "http://169.254.169.254/latest/meta-data/"` (AWS Metadata service).
    *   Result: The server fetches its own AWS credentials and returns them to the attacker.
*   **Mitigation:**
    *   Allow-list permitted domains for outgoing requests.
    *   Block internal IP ranges (10.0.0.0/8, 127.0.0.1) in the HTTP client configuration.
