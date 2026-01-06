# Chaos Engineering

> **Domain:** DevOps, SRE, Distributed Systems
> **Key Concepts:** Blast Radius, Fault Injection, Game Days, Resilience

**Chaos Engineering** is the discipline of experimenting on a system in order to build confidence in the system's capability to withstand turbulent conditions in production. It is **not** just "breaking things randomly"; it is a scientific method.

---

## 1. The Scientific Method

Chaos Engineering follows a strict loop:

1.  **Define Steady State:** What does "normal" look like? (e.g., Error rate < 1%, Latency < 200ms).
2.  **Hypothesis:** "If we kill the primary database node, the secondary will take over in < 30s, and the user will see no errors."
3.  **Experiment:** Inject the failure (kill the node).
4.  **Verify:** Did the Steady State hold?
5.  **Fix:** If it failed, you found a bug. Fix it.

---

## 2. The "Blast Radius"

You don't start by deleting the production database. You minimize the impact.

1.  **Local/Dev:** Run chaos tests in the staging environment.
2.  **Canary:** Run chaos on 1% of production traffic.
3.  **Zone:** Run chaos on one Availability Zone (AZ).
4.  **Region:** (Advanced) Take down an entire AWS Region.

**Goal:** Maximize learning while minimizing customer pain.

---

## 3. Types of Failures (Fault Injection)

### 3.1. Infrastructure
*   **Kill Process:** Terminate the web server process.
*   **Shutdown Server:** `sudo reboot` or terminate EC2 instance.
*   **CPU/Memory Spike:** Run a "stress" command to hog 100% CPU. Does the autoscaler kick in?

### 3.2. Network
*   **Latency:** Add 500ms delay to all packets. Does the UI handle it gracefully?
*   **Packet Loss:** Drop 5% of packets. Does TCP retry storm kill the network?
*   **Blackhole:** Block all traffic to a dependency (e.g., the Payment Gateway). Does the checkout page hang or show a nice error?

### 3.3. Application
*   **Exception Injection:** Force the code to throw `NullPointerException` at a specific line.
*   **Clock Skew:** Change the system time. (Breaks lots of crypto and distributed consensus).

---

## 4. Tools

1.  **Chaos Monkey (Netflix):** The original. Randomly kills instances.
2.  **Chaos Mesh / LitmusChaos:** Kubernetes-native chaos operators.
3.  **Gremlin:** SaaS platform for safe, managed chaos.
4.  **Toxiproxy:** Network simulator for local testing.

---

## 5. Game Days

A **Game Day** is a scheduled team event.
*   **Roles:**
    *   *Master of Disaster:* Runs the attacks.
    *   *First Responders:* The on-call team (trying to fix it).
    *   *Scribe:* Writes down what happened.
*   **Objective:** Test the *people* and *processes*, not just the tech. Do the alerts fire? Is the runbook accurate?

---

## 6. Myths

*   *"We are too small for Chaos."* -> If you are small, fixing it is cheap. If you wait until you are huge, outages cost millions.
*   *"Chaos Engineering creates outages."* -> Chaos Engineering *reveals* outages that are already waiting to happen, usually at 3 AM.
