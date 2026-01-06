# Serverless Patterns

> **Domain:** Cloud Architecture, AWS Lambda, Event-Driven
> **Key Concepts:** FaaS, Cold Starts, Event Sourcing, Fan-Out, Step Functions

**Serverless** is a misnomer. Servers still exist, but they are abstracted away. You pay for execution time (ms), not for idle servers. It forces a shift from "Monolithic" thinking to "Event-Driven" thinking.

---

## 1. The Challenges

### 1.1. Cold Starts
When a function hasn't run in a while, the provider spins down the container. The next request must wait for a new container to spin up (Download Code -> Start Runtime -> Init Code).
*   *Java/Spring:* 5-10 seconds cold start.
*   *Node/Python:* 200-500ms.
*   *Mitigation:* Provisioned Concurrency (keep N warm), use lighter runtimes (Go/Rust), avoid massive dependencies.

### 1.2. Statelessness
Lambda functions are ephemeral. You cannot save a file to `/tmp` and expect it to be there for the next request. State must be external (S3, DynamoDB, Redis).

---

## 2. Common Patterns

### 2.1. The API Gateway Proxy
The standard REST API replacement.
*   **Flow:** Client -> API Gateway -> Lambda -> DynamoDB.
*   **Pros:** Infinite scaling, pay-per-request.
*   **Cons:** Latency (Cold starts + Gateway overhead). Expensive at high scale compared to Containers.

### 2.2. Fan-Out (Pub/Sub)
Process parallel work asynchronously.
*   **Scenario:** A user uploads a video. You need to:
    1.  Generate thumbnail.
    2.  Transcode to 1080p.
    3.  Transcode to 720p.
    4.  Update DB.
*   **Anti-Pattern:** One function calling the others sequentially.
*   **Pattern:**
    1.  Upload -> S3 Event -> SNS Topic (or EventBridge).
    2.  SNS triggers 4 independent Lambdas in parallel.
*   **Benefit:** Total time = Max(Duration), not Sum(Duration). Decoupled logic.

### 2.3. SQS Buffer (Throttling)
Protecting downstream systems.
*   **Scenario:** 10,000 webhooks hit your API in 1 second.
*   **Problem:** If you trigger 10,000 Lambdas, they will hammer your SQL database, crashing it.
*   **Pattern:** API Gateway -> SQS Queue -> Lambda (with concurrency limit).
*   **Logic:** The Queue absorbs the spike. The Lambda processes messages at a controlled rate (e.g., 50 at a time).

### 2.4. Step Functions (Orchestration)
Managing long-running workflows.
*   **Problem:** Lambda creates a thumbnail, fails. How do you retry? What if the whole workflow takes 20 minutes (Lambda max is 15 min)?
*   **Solution:** State Machines (AWS Step Functions).
*   **Logic:**
    *   State 1: Call `Transcode`. Wait for result.
    *   State 2: If Success -> Call `EmailUser`.
    *   State 3: If Fail -> Wait 10s -> Retry.

---

## 3. The Function Monolith (Anti-Pattern)

Do not deploy your entire Express.js/Flask app in a single Lambda function.
*   **Why?**
    1.  **Cold Starts:** You are loading code for *every* route (Profile, Billing, Search) just to serve the *Login* route.
    2.  **Permissions:** This Lambda needs permission to access *all* DB tables, violating Least Privilege.
*   **Better:** One Lambda per route (or related group of routes).

---

## 4. Event Sourcing / CQRS
Serverless fits naturally with CQRS (Command Query Responsibility Segregation).
*   **Write:** Lambda writes to DynamoDB.
*   **Stream:** DynamoDB Stream triggers a Lambda.
*   **Read:** Lambda updates an Elasticsearch cluster for complex queries.

---

## 5. Security

*   **IAM Roles:** Every function should have its own role. The "Thumbnail Generator" role should only have `s3:GetObject` (Source) and `s3:PutObject` (Destination), nothing else.
*   **VPC:** Running Lambda inside a VPC adds significant cold start time (unless using updated Hyperplane ENIs). Avoid unless you need to access RDS/ElastiCache.
