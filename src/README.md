# Technical Documentation Repository

A comprehensive, well-organized collection of technical documentation, guides, and notes covering software engineering, systems programming, cloud computing, machine learning, and more.

## Overview

This knowledge repository contains in-depth documentation on diverse technical topics, organized into focused domains. Each section includes theoretical foundations, practical examples, best practices, and real-world applications designed for learning, reference, and interview preparation.

## Contents

### Artificial Intelligence & Machine Learning

- **[AI](./ai/)** - Artificial Intelligence, LLMs, prompt engineering, generative AI
  - Large Language Models (GPT, Claude, Llama, DeepSeek R1, Phi)
  - Prompt engineering and software development prompt patterns
  - Transformers architecture, RAG, and vector databases
  - Agent frameworks, tool use, MCP, and agent skills
  - Model serving and inference (vLLM), speech (Whisper)
  - Stable Diffusion, Flux.1, ComfyUI, and image generation
  - Fine-tuning, LoRA, and generative AI applications

- **[Machine Learning](./machine_learning/)** - ML algorithms, frameworks, and deep learning
  - Supervised, unsupervised, and reinforcement learning (incl. deep RL)
  - Neural networks, convolution, and transformer architectures
  - Mixture of Experts (MoE) and deep generative models
  - PyTorch, JAX, NumPy, and Hugging Face
  - Gradient boosting, transfer learning, and evaluation metrics
  - Model quantization (GPTQ, AWQ, INT8/INT4) and CUDA/GPU optimization
  - Notable research papers

### Mathematics

- **[Maths](./maths/)** - Mathematical foundations for CS and ML
  - Linear algebra (vectors, matrices, decompositions)
  - Calculus and optimization
  - Probability and statistics

### Systems Programming & Operating Systems

- **[Linux](./linux/)** - Linux system administration, kernel architecture, and networking
  - Essential commands and shell scripting
  - Kernel development, modules, and driver programming
  - cfg80211 & mac80211 wireless subsystems
  - Networking stack (netfilter, iptables, tc, WireGuard)
  - Device Tree, sysfs, systemd, and eBPF
  - Cross-compilation for embedded systems

- **[Embedded](./embedded/)** - Embedded systems development and microcontrollers
  - Development platforms (Arduino, ESP32, STM32, AVR, Raspberry Pi)
  - Communication protocols (UART, SPI, I2C, CAN, SDIO, Ethernet)
  - Peripherals: GPIO, ADC, DAC, PWM, DMA, timers, RTC, clock systems
  - Wireless: BLE and LoRa
  - Bootloaders, OTA updates, linker scripts, and flash filesystems
  - Security: secure boot, ARM TrustZone-M
  - Debugging: JTAG/SWD, GDB, RTT/semihosting, hardfault analysis
  - Processor design, ISA, CMSIS, and signal integrity

- **[RTOS](./rtos/)** - Real-time operating systems
  - FreeRTOS, ThreadX, Zephyr, and RT-Linux (PREEMPT_RT)
  - Task scheduling and priority management
  - Synchronization primitives
  - Interrupt handling and timing constraints

### Software Development

- **[Programming](./programming/)** - Programming languages and paradigms
  - Python, C, C++, Rust, Go, Zig, Java, Kotlin, Lua
  - JavaScript/TypeScript, Bash, and SQL
  - Language features, idioms, and design patterns
  - Memory management, concurrency, and compilers
  - Interview questions

- **[Algorithms](./algorithms/)** - Algorithm design, analysis, and patterns
  - Sorting, searching, and graph algorithms (flow, SCC, LCA)
  - Dynamic programming, greedy, divide and conquer, backtracking
  - Range query structures (segment/Fenwick trees, sparse table, HLD)
  - Patterns: sliding window, two pointers, monotonic stack/queue, binary search
  - Bit manipulation, string algorithms, hashing, heuristic search
  - Time and space complexity (Big O), interview patterns
  - Raft consensus algorithm

- **[Data Structures](./data_structures/)** - Core data structures and implementations
  - Arrays, linked lists, stacks, queues, circular buffers
  - Trees (BST, AVL, Red-Black, advanced), heaps, tries
  - Hash tables, skip lists, and union-find
  - Graphs, spatial structures, and suffix arrays
  - Probabilistic structures (Bloom filter, MinHash/LSH, HNSW, product quantization)
  - Inverted index, LRU cache, persistent structures, and CRDTs

- **[Web Development](./web_development/)** - Modern web technologies
  - Frontend: React, Next.js, Vue.js, Svelte, SvelteKit
  - Styling: CSS, Tailwind CSS
  - Backend: Express.js, NestJS, Django, Flask, FastAPI
  - APIs: REST, GraphQL, gRPC, and API design
  - WebAssembly for high-performance web applications
  - Browser/Web APIs, accessibility, frontend performance, web security

- **[Mobile Development](./mobile_development/)** - Mobile application development
  - Native iOS development (Swift, SwiftUI, UIKit)
  - Native Android development (Kotlin, Jetpack Compose)
  - Cross-platform: React Native, Flutter
  - Mobile architecture patterns (MVVM, Clean Architecture)

### Cloud & Infrastructure

- **[Cloud](./cloud/)** - Cloud computing platforms and services
  - AWS, Microsoft Azure, Google Cloud Platform
  - Service models (IaaS, PaaS, SaaS, FaaS)
  - Cloud architecture patterns (microservices, serverless, event-driven)
  - Cost optimization and security best practices
  - Multi-cloud and hybrid cloud strategies

- **[DevOps](./devops/)** - DevOps practices, tools, and automation
  - CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI)
  - Containerization (Docker) and orchestration (Kubernetes)
  - Infrastructure as Code (Terraform, CloudFormation)
  - Monitoring, logging, and observability
  - Site Reliability Engineering (SRE) and chaos engineering
  - Cloud deployment strategies

- **[System Design](./system_design/)** - Software architecture and scalability
  - Distributed systems, microservices, and consensus
  - Caching strategies, load balancing, CDNs, consistent hashing
  - Database sharding, replication, and message queues
  - Rate limiting, idempotency, id generation, and RPC
  - High availability, fault tolerance, CAP theorem, and observability
  - Worked design problems (URL shortener, chat, news feed, typeahead,
    ride-sharing, video streaming) and an interview framework

### Data & Databases

- **[Databases](./databases/)** - Database systems and data engineering
  - Relational databases (PostgreSQL, SQLite, DuckDB)
  - Analytical and NoSQL databases (ClickHouse, MongoDB, Redis)
  - Database design, normalization, and indexing
  - SQL query optimization
  - ACID vs BASE and consistency trade-offs
  - Message queues (Apache Kafka), data pipelines, and ETL

### Security

- **[Security](./security/)** - Application security and cryptography
  - Authentication and authorization (OAuth2, JWT)
  - Encryption, hashing, HMAC, and digital signatures
  - Certificates, SSL/TLS, and PKI
  - Zero trust architecture
  - OWASP Top 10 vulnerabilities
  - Security testing and web application security

### Networking

- **[Networking](./networking/)** - Network protocols and architecture
  - OSI and TCP/IP models; IPv4, IPv6, ARP, DHCP, DNS, mDNS
  - Transport: TCP, UDP, QUIC, RTP; MTU/PMTUD
  - Application: HTTP, HTTP/2, WebSocket, gRPC, SSH
  - Routing: BGP/anycast, OSPF/IS-IS
  - Real-time & NAT traversal: WebRTC, STUN, TURN, ICE, NAT-PMP, PCP, UPnP
  - Overlay/container networking, IPsec, WireGuard, firewalls
  - IoT protocols and Ethernet/VLAN

- **[WiFi](./wifi/)** - Wireless networking technologies
  - IEEE 802.11 standards and protocols
  - WiFi configuration and optimization
  - Wireless security (WPA2, WPA3)
  - Troubleshooting wireless networks

### Development Tools & Practices

- **[Tools](./tools/)** - Development tools and utilities
  - Editors and multiplexers (Vim, tmux)
  - Build systems (Make, Ninja, Bazel) and toolchains (GCC, Clang)
  - Code navigation (ctags, cscope) and package management (uv)
  - Text processing (grep, ripgrep, sed, awk, find)
  - Networking & capture (curl, wget, nmap, tcpdump, tshark, Wireshark)
  - Media (ffmpeg), automation (Ansible), Wi-Fi (hostapd, wpa_supplicant), docs (mdBook)

- **[Git](./git/)** - Version control and collaboration
  - Git fundamentals and advanced commands
  - Branching strategies (Git Flow, GitHub Flow)
  - Merge vs rebase workflows
  - Collaboration best practices

- **[Testing](./testing/)** - Testing strategies and frameworks
  - Unit testing, integration testing, E2E testing
  - Test-driven development (TDD)
  - Testing frameworks (pytest, Jest, unittest)
  - Mocking, fixtures, and test coverage
  - Performance and load testing

- **[Debugging](./debugging/)** - Debugging tools and techniques
  - GDB debugger fundamentals
  - Core dump analysis
  - Linux kernel debugging
  - Debugging workflows and best practices

### Platform-Specific

- **[Android](./android/)** - Android development and internals
  - Android application development
  - Android internals and architecture
  - Binder IPC mechanism
  - ADB commands and debugging tools

### Other Topics

- **[Finance](./finance/)** - Markets, trading, and investing
  - Asset classes: stocks, bonds, ETFs, REITs, forex, commodities, crypto
  - Derivatives: options, futures, volatility, and credit markets
  - Strategies: momentum/trend, pairs/mean reversion, event-driven, algorithmic trading
  - Analysis: technical, fundamental, and valuation
  - Risk and portfolio management, interest rates, and market cycles
  - Macroeconomics, private markets, tax strategies, and financial planning

- **[Misc](./misc/)** - Miscellaneous topics and utilities
  - Operating systems and computer graphics
  - Blockchain and Solana
  - Data tooling (pandas, matplotlib)
  - U-Boot, Ubuntu, and BLE reference material

## Documentation Philosophy

This repository emphasizes:

- **Comprehensive Coverage**: From fundamentals to advanced topics
- **Practical Examples**: Real-world code samples and use cases
- **Best Practices**: Industry-standard approaches and patterns
- **Clear Explanations**: Concepts explained with clarity and depth
- **Code Quality**: Well-documented, tested examples
- **Up-to-date**: Regular updates with modern technologies and practices

## Usage

This repository serves multiple purposes:

- **Learning Resource**: Structured guides for learning new technologies
- **Reference Guide**: Quick lookup for syntax, commands, and patterns
- **Interview Preparation**: Core concepts and common interview questions
- **Best Practices**: Proven approaches and design patterns
- **Project Templates**: Boilerplate code and project structures

## Navigation

- Each topic directory contains a comprehensive **README.md** with an overview
- Topics are organized from basics to advanced concepts
- Code examples include explanations and use cases
- Cross-references link related topics across directories

## Contributing

This is a living knowledge base, continuously updated with:

- New technologies and frameworks
- Updated best practices and patterns
- Additional code examples and tutorials
- Refined explanations based on learning
- Community feedback and corrections

---

**Note**: This is an evolving project. Content is regularly updated, reorganized, and expanded to reflect current best practices and emerging technologies.

*Last updated: 2026*
