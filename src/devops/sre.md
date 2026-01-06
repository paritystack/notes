# Site Reliability Engineering (SRE)

> **Domain:** DevOps, Operations, Software Engineering
> **Key Concepts:** SLI, SLO, SLA, Error Budget, Toil, Post-Mortems

**Site Reliability Engineering (SRE)** is a discipline that incorporates aspects of software engineering and applies them to infrastructure and operations problems. Coined by Google, the core tenet is: *"SRE is what happens when you ask a software engineer to design an operations team."*

---

## 1. The Core Philosophy

Traditional "Dev vs. Ops" creates conflict: Devs want to ship features (change), Ops want stability (no change). SRE bridges this by agreeing on a quantifiable level of unreliability.

**Key Principle:** 100% reliability is not the goal. It is expensive and slows down innovation. The goal is "reliable enough" for the user.

---

## 2. The Metrics: SLI, SLO, SLA

These acronyms are often confused but distinct.

### 2.1. SLI (Service Level Indicator)
*   **Definition:** A quantitative measure of some aspect of the level of service that is provided.
*   **The "What":** Real numbers from your monitoring system.
*   **Examples:**
    *   *Latency:* 95th percentile response time.
    *   *Availability:* (Successful Requests / Total Requests) * 100.
    *   *Freshness:* Time since last database update.

### 2.2. SLO (Service Level Objective)
*   **Definition:** A target value or range of values for a service level that is measured by an SLI.
*   **The "Goal":** The internal health target.
*   **Example:** "99.9% of requests in the last 30 days will succeed."
*   **Rule:** SLOs are stricter than SLAs. If you breach an SLO, you alert the team.

### 2.3. SLA (Service Level Agreement)
*   **Definition:** A contract with the user (or customer) that specifies consequences if the SLO is not met.
*   **The "Contract":** The external promise.
*   **Example:** "If availability drops below 99.5%, we will refund 10% of your bill."
*   **Rule:** Monitoring systems don't care about SLAs; lawyers do.

---

## 3. Error Budgets

The **Error Budget** is the most transformative concept in SRE. It aligns incentives between Dev and Ops.

*   **Formula:** $1 - SLO = Error Budget$
*   **Example:** If SLO is 99.9% availability, your error budget is 0.1%. In a 30-day month (43,200 minutes), you are *allowed* to be down for 43.2 minutes.

**Usage:**
*   **If Budget > 0:** Devs can push risky features, perform experiments, or do chaos engineering.
*   **If Budget < 0 (exhausted):** **Code Freeze.** All engineering effort shifts from "Features" to "Stability" (fixing bugs, improving tests) until the budget recovers.

---

## 4. Toil Management

**Toil** is defined as work that is:
1.  Manual
2.  Repetitive
3.  Tactical (devoid of long-term value)
4.  Scales linearly with service growth

**The 50% Rule:** SREs should spend max 50% of their time on Ops work (tickets, on-call). The other 50% *must* be spent on **Engineering** (automating away the toil). If toil exceeds 50%, the team is overloaded and needs to push back work to Devs.

---

## 5. Incident Management

When things break, SREs follow a structured protocol.

1.  **Triage:** Stop the bleeding. Rollback, drain traffic, or shed load. Do not try to "fix" the root cause during the outage; just restore service.
2.  **Post-Mortem (Root Cause Analysis):**
    *   **Blameless:** Never fire someone for an honest mistake. If a human made a mistake, the *system* allowed them to make it. Fix the system.
    *   **Action Items:** Specific engineering tasks to prevent recurrence (e.g., "Add a canary deployment stage," "Add a rate limit").

---

## 6. SRE vs. DevOps

| Feature | DevOps | SRE |
| :--- | :--- | :--- |
| **Focus** | Cultural movement, breaking silos | Concrete implementation of DevOps |
| **Team** | "You build it, you run it" (General) | Specialized team for critical systems |
| **Approach** | CI/CD, Agile | Error Budgets, Software to solve Ops problems |

## 7. The Four Golden Signals

Google recommends monitoring these four metrics for every user-facing system:
1.  **Latency:** Time to serve a request.
2.  **Traffic:** Demand on the system (req/sec).
3.  **Errors:** Rate of requests that fail (5xx, explicit failures).
4.  **Saturation:** How "full" the system is (CPU load, memory usage).
