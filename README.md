# 📚 Comprehensive Technical Knowledge Base

A comprehensive, continuously updated collection of technical tutorials and references covering embedded systems, data structures, algorithms, machine learning, web development, DevOps, system design, and more.

**Live Site**: [notes.paritystack.in](https://notes.paritystack.in)

---

## 📋 What's Inside

This knowledge base contains **75+ topics** across **21 categories** with **25,000+ lines** of tutorial-style content, practical code examples, and best practices.

### Core Computer Science (8 topics)
- **Data Structures**: Arrays, Linked Lists, Stacks, Queues, Hash Tables, Trees, Tries, Heaps
- **Algorithms**: Sorting, Searching, Dynamic Programming, Graph Algorithms, Divide & Conquer, Backtracking
- **Big-O Analysis**: Time/space complexity, optimization strategies

### Embedded Systems (6 topics)
- **Communication Protocols**: SPI, UART, I2C
- **Analog I/O**: ADC (Analog-to-Digital), DAC (Digital-to-Analog)
- **Hardware Control**: Timers, PWM, GPIO, Interrupts, Watchdog Timers
- **Microcontrollers**: Arduino, ESP32, STM32, ARM
- **Platforms**: Arduino, Raspberry Pi, STM32, AVR

### Backend & Infrastructure (12 topics)
- **APIs**: REST API Design, HTTP/HTTPS, Status Codes, Authentication
- **Databases**: SQL (PostgreSQL, SQLite, DuckDB), NoSQL (MongoDB, Redis, Cassandra), Event Streaming (Apache Kafka)
- **DevOps**: Docker, Kubernetes, CI/CD Pipelines, GitHub Actions, GitLab CI
- **Networking**: TCP/IP, UDP, DNS, IP Routing (IPv4, IPv6), Firewalls, Real-time (WebSocket, WebRTC), NAT Traversal (STUN, TURN, ICE, PCP, NAT-PMP)
- **Security**: Encryption (symmetric/asymmetric), SSL/TLS, Certificates, Hashing, HMAC, Digital Signatures, OAuth 2.0, JWT
- **Cloud**: AWS, Google Cloud, Azure
- **Linux**: Kernel, Networking Stack, iptables, Netfilter, Commands

### Frontend & Web Development (14 topics)
- **Frontend Frameworks**: React, Next.js (SSR/SSG), Vue.js, Svelte (components, hooks, state management)
- **Backend Frameworks**: Express.js, NestJS (TypeScript), Django, Flask, FastAPI (Python async)
- **CSS**: Tailwind CSS (utility-first framework)
- **APIs**: REST, GraphQL, gRPC (schema design, queries, mutations)
- **Browser APIs**: Storage (localStorage, IndexedDB), Workers (Web Workers, Service Workers), Notifications, Geolocation, File API, Observers (Intersection, Mutation, Resize)
- **Web Fundamentals**: HTML, CSS, JavaScript, TypeScript
- **Tools & Build**: Webpack, Vite, npm, yarn

### Machine Learning & AI (9 topics)
- **Neural Networks**: Architecture, Activation Functions, Training, Backpropagation
- **Deep Learning**: CNNs, RNNs, LSTMs, Transformers
- **Large Language Models (LLMs)**: GPT, BERT, Fine-tuning, Prompt Engineering
- **Generative AI**: Stable Diffusion, ComfyUI, Image Generation
- **Learning Approaches**: Supervised, Unsupervised, Reinforcement Learning
- **Frameworks**: TensorFlow, PyTorch, Hugging Face

### System Design (4 topics)
- **Scalability**: Vertical/Horizontal Scaling, Load Balancing, Sharding, Replication
- **Caching**: Cache-Aside, Write-Through, Write-Behind, Invalidation Strategies, Redis
- **Microservices**: Service decomposition, communication patterns, resilience, service discovery
- **Distributed Systems**: CAP Theorem, Consensus, Message Queues, RPC

### Mobile Development (2 topics)
- **React Native**: Cross-platform mobile development with React, navigation, native modules
- **Flutter**: Google's UI toolkit with Dart, widgets, state management, platform channels

### Testing & Quality (3 topics)
- **Unit Testing**: pytest, Jest, JUnit with practical examples
- **Test-Driven Development**: Red-Green-Refactor cycle, benefits, patterns
- **Integration & E2E Testing**: Best practices, test pyramids

### Finance (8 topics)
- **Stock Market**: Trading fundamentals, analysis techniques
- **Technical Analysis**: Charts, indicators, patterns
- **Options & Futures**: Derivatives, risk management
- **Crypto**: Blockchain basics, trading
- **Fundamental Analysis**: Financial metrics, valuation

### Other Topics (9 topics)
- **Programming Languages**: Python, C, C++, TypeScript, JavaScript, Rust, Java, Go, Lua, SQL, Bash
- **Version Control**: Git commands, workflows, GitHub
- **Tools**: Vim, Tmux, Grep, Find, Docker, Git
- **Android**: Development, ADB, Internals

---

## 🎯 Key Features

✅ **Tutorial-Style Learning**: Clear explanations with theory and practice
✅ **Multi-Language Examples**: Python, TypeScript, JavaScript, C/C++, Java, Rust, Go, Lua, SQL, Bash, Dart
✅ **Real-World Applications**: Practical use cases for every concept
✅ **Best Practices**: Do's and don'ts, common pitfalls
✅ **Visual Diagrams**: Architecture, flowcharts, algorithm visualizations
✅ **Code Examples**: Runnable, production-ready code snippets
✅ **ELI10 Sections**: Simplified explanations for beginners
✅ **Complexity Analysis**: Big-O, time/space analysis
✅ **Further Resources**: Links for deeper learning

---

## 🏗️ Technology Stack

- **Built With**: mdBook (Rust-based static site generator)
- **Format**: Markdown with LaTeX math support
- **Hosting**: GitHub Pages
- **Custom Domain**: notes.paritystack.in
- **CI/CD**: GitHub Actions (automated deployment)

---

## 🚀 Quick Navigation

### For Beginners
Start with these topics to build foundational knowledge:
1. **Data Structures** - Understand how data is organized
2. **Big-O Analysis** - Learn about algorithm efficiency
3. **Sorting & Searching** - Classic algorithms explained
4. **REST APIs** - Understand web communication

### For Intermediate Developers
Deepen your skills:
1. **Dynamic Programming** - Solve complex optimization problems
2. **System Design** - Design scalable systems
3. **Database Design** - SQL vs NoSQL tradeoffs
4. **React** - Modern frontend development

### For Advanced Engineers
Master specialized topics:
1. **Distributed Systems** - CAP theorem, consensus algorithms
2. **Kubernetes** - Container orchestration at scale
3. **Machine Learning** - Neural networks, deep learning
4. **LLMs** - Language models and transformers

---

## 📖 How to Use

### View Online
Visit [notes.paritystack.in](https://notes.paritystack.in) to read all content.

### Build Locally

#### Prerequisites
- Rust (for mdBook)
- Git

#### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/notes.git
cd notes

# Install mdBook
cargo install mdbook

# Build and serve locally
mdbook serve

# Open browser to http://localhost:3000
```

#### Edit Content
```bash
# Edit markdown files in src/
vim src/category/topic.md

# Changes automatically reflect in local server
```

---

## 📚 Category Overview

| Category | Files | Topics | Status |
|----------|-------|--------|--------|
| **Embedded Systems** | 23 | Protocols, hardware, microcontrollers | ✅ Complete |
| **Data Structures** | 6 | Arrays, lists, trees, graphs | ✅ Complete |
| **Algorithms** | 13 | Sorting, searching, DP, graphs | ✅ Complete |
| **Machine Learning** | 13 | Neural nets, deep learning, LLMs | ✅ Complete |
| **Web Development** | 14 | React, Next.js, Vue, Svelte, Express, NestJS, Django, Flask, FastAPI | ✅ Complete |
| **Mobile Development** | 2 | React Native, Flutter | ✅ Complete |
| **DevOps & CI/CD** | 4 | Docker, Kubernetes, pipelines | ✅ Complete |
| **Databases** | 8 | SQL, PostgreSQL, SQLite, DuckDB, NoSQL, MongoDB, Redis, Kafka | ✅ Complete |
| **System Design** | 4 | Scalability, caching, microservices, RPC | ✅ Complete |
| **Testing** | 3 | Unit testing, TDD, best practices | ✅ Complete |
| **Networking** | 20 | TCP/IP, IPv4/IPv6, WebSocket, WebRTC, NAT traversal | ✅ Complete |
| **Security** | 9 | Encryption, SSL/TLS, certificates, OAuth 2.0, JWT | ✅ Complete |
| **Linux** | 10 | Kernel, kernel patterns, commands, networking, iptables, systemd | ✅ Complete |
| **Programming** | 12 | Python, C/C++, TypeScript, JavaScript, Rust, Java, Go, Lua | ✅ Complete |
| **Cloud** | 5 | AWS, GCP, Azure services | ✅ Complete |
| **Finance** | 8 | Markets, analysis, crypto | ✅ Complete |
| **Tools** | 16 | Vim, tmux, grep, git, docker | ✅ Complete |
| **WiFi** | 7 | Standards, security, roaming | ✅ Complete |
| **Android** | 4 | Development, ADB, internals | ✅ Complete |
| **Git** | 4 | Commands, workflows, GitHub | ✅ Complete |
| **AI** | 9 | LLMs, prompt engineering, tools | ✅ Complete |

**Total**: 21 categories, 175+ files, 25,000+ lines of content

---

## 🤝 Contributing

### Report Issues
Found an error or want to improve content? Open an issue on GitHub.

### Submit Updates
```bash
# Create a new branch
git checkout -b improve/topic-name

# Make your changes
# Commit with clear messages
git add .
git commit -m "docs: improve topic-name explanation"

# Push and create pull request
git push -u origin improve/topic-name
```

### Add New Content
Create a new markdown file in the appropriate category:
```bash
# Create new topic
touch src/category/new_topic.md

# Add to src/SUMMARY.md in correct section
```

---

## 📊 Statistics

- **Total Categories**: 21
- **Total Files**: 175+
- **Total Content**: 25,000+ lines
- **Code Examples**: 300+
- **Languages Covered**: 11+
- **Last Updated**: 2024
- **Automated Deployment**: Yes ✅

---

## 🔗 Related Links

- **Live Site**: [notes.paritystack.in](https://notes.paritystack.in)
- **GitHub**: [Repository](https://github.com/yourusername/notes)
- **mdBook Docs**: [https://rust-lang.github.io/mdBook/](https://rust-lang.github.io/mdBook/)

---

## 📝 License

This knowledge base is open source and available for educational use. Feel free to fork, learn, and contribute!

---

## 🎯 Vision

To create a comprehensive, accessible, and continuously updated technical reference that helps developers at all levels master computer science, software engineering, and modern technologies.

**Last Updated**: December 2024
**Maintained By**: Manu

---

## 💡 Tips for Best Learning

1. **Read the Overview First** - Start with README.md in each category
2. **Follow Code Examples** - Try running the examples locally
3. **Review Complexity** - Understand Big-O and trade-offs
4. **Check Resources** - Dive deeper with provided links
5. **Practice Problems** - LeetCode problems for algorithms/data structures

Happy Learning! 🚀
