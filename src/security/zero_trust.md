# Zero Trust Architecture (ZTA)

> **Domain:** Cybersecurity, Network Architecture
> **Key Concepts:** "Never Trust, Always Verify", Least Privilege, Micro-segmentation, Identity Awareness

**Zero Trust** is a security model that assumes breach and verifies each request as though it originates from an open network. It eliminates the traditional concept of a "trusted internal network" (Castle-and-Moat).

---

## 1. The Core Principles

NIST SP 800-207 defines the standard.

1.  **Never Trust, Always Verify:** Every access request is fully authenticated, authorized, and encrypted before granting access.
2.  **Least Privilege:** Users/Services get only the permissions they need, for the specific time they need them.
3.  **Assume Breach:** Design the network as if an attacker is already present. Minimizing the "blast radius" is key.

---

## 2. Shift from Perimeter to Identity

*   **Old Model (Perimeter):** "If IP starts with 10.0.0, you are trusted."
    *   *Flaw:* Once an attacker hacks the VPN or a single server, they can move laterally anywhere.
*   **Zero Trust (Identity):** "I don't care if you are on the corporate Wi-Fi. Who are you? Is your device healthy?"
    *   *Mechanism:* Access policies are based on **Identity** (User/Service Account), **Context** (Device health, Time, Location), and **Data Sensitivity**.

---

## 3. Architecture Components

### 3.1. Policy Enforcement Point (PEP)
The "Gatekeeper." It intercepts the request.
*   *Examples:* Nginx Reverse Proxy, AWS API Gateway, VPN Concentrator.

### 3.2. Policy Decision Point (PDP)
The "Brain." It decides Yes/No based on rules.
*   *Process:* PEP asks PDP -> PDP checks Identity Provider (IdP) + Device Managers -> PDP returns "Allow" -> PEP opens connection.

---

## 4. Implementation Technologies

### 4.1. Micro-segmentation
Instead of one flat network, every workload is its own island.
*   **Tool:** Firewall rules between *every* container.
*   **Example:** The "Frontend" container can talk to "Backend" on port 8080, but "Backend" cannot talk to "Frontend".

### 4.2. mTLS (Mutual TLS)
Standard TLS authenticates the Server to the Client. **mTLS** authenticates the Client to the Server too.
*   **Certificate Authority (CA):** Every service has a signed certificate.
*   **Benefit:** Service A literally cannot connect to Service B without a valid cert, even if they are on the same subnet.
*   **Tool:** Service Meshes (Istio, Linkerd) handle this automatically.

### 4.3. Identity Aware Proxy (IAP)
Replaces the VPN.
*   Instead of `vpn.company.com`, you go to `app.company.com`.
*   The Proxy redirects you to Google/Okta login.
*   If valid, the Proxy tunnels you to the internal app. The app itself is never exposed to the public internet.

---

## 5. Challenges

1.  **Legacy Systems:** Mainframes or old protocols (SMB, FTP) often don't support modern identity tokens.
2.  **Latency:** verifying every single request adds overhead.
3.  **Complexity:** Managing thousands of certificates and granular policies requires automation.

---

## 6. The "BeyondCorp" Model
Google's implementation of Zero Trust.
*   **Observation:** Google employees work from coffee shops without VPNs.
*   **Logic:** The device is managed (MDM). The user is authenticated (2FA). The transport is encrypted (TLS). The network location is irrelevant.
