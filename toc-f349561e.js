// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="index.html">Introduction</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/index.html"><strong aria-hidden="true">1.</strong> Git</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/cheatsheet.html"><strong aria-hidden="true">1.1.</strong> Git Cheatsheet</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/commands.html"><strong aria-hidden="true">1.2.</strong> Git Commands</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/internals.html"><strong aria-hidden="true">1.3.</strong> Git Internals</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/github.html"><strong aria-hidden="true">1.4.</strong> Github</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="git/repo.html"><strong aria-hidden="true">1.5.</strong> Repo</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/index.html"><strong aria-hidden="true">2.</strong> Programming Languages</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/python.html"><strong aria-hidden="true">2.1.</strong> Python</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/c.html"><strong aria-hidden="true">2.2.</strong> C</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/cpp.html"><strong aria-hidden="true">2.3.</strong> C++</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/javascript.html"><strong aria-hidden="true">2.4.</strong> JavaScript</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/typescript.html"><strong aria-hidden="true">2.5.</strong> TypeScript</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/bash.html"><strong aria-hidden="true">2.6.</strong> Bash</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/java.html"><strong aria-hidden="true">2.7.</strong> Java</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/go.html"><strong aria-hidden="true">2.8.</strong> Go</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/lua.html"><strong aria-hidden="true">2.9.</strong> Lua</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/rust.html"><strong aria-hidden="true">2.10.</strong> Rust</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/sql.html"><strong aria-hidden="true">2.11.</strong> SQL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/interview_questions.html"><strong aria-hidden="true">2.12.</strong> Interview Questions</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/design_patterns.html"><strong aria-hidden="true">2.13.</strong> Design Patterns</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/kotlin.html"><strong aria-hidden="true">2.14.</strong> Kotlin</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/concurrency.html"><strong aria-hidden="true">2.15.</strong> Concurrency</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/memory_management.html"><strong aria-hidden="true">2.16.</strong> Memory Management</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="programming/compilers.html"><strong aria-hidden="true">2.17.</strong> Compilers</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/index.html"><strong aria-hidden="true">3.</strong> Linux</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/networking.html"><strong aria-hidden="true">3.1.</strong> Networking</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/kernel.html"><strong aria-hidden="true">3.2.</strong> Kernel</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/kernel_patterns.html"><strong aria-hidden="true">3.3.</strong> Kernel Development Patterns</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/driver_development.html"><strong aria-hidden="true">3.4.</strong> Driver Development</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/device_tree.html"><strong aria-hidden="true">3.5.</strong> Device Tree</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/cross_compilation.html"><strong aria-hidden="true">3.6.</strong> Cross Compilation</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/cfg80211_mac80211.html"><strong aria-hidden="true">3.7.</strong> cfg80211 &amp; mac80211</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/ebpf.html"><strong aria-hidden="true">3.8.</strong> eBPF</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/netlink.html"><strong aria-hidden="true">3.9.</strong> Netlink</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/commands.html"><strong aria-hidden="true">3.10.</strong> Linux Commands</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/netfilter.html"><strong aria-hidden="true">3.11.</strong> Netfilter</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/tc.html"><strong aria-hidden="true">3.12.</strong> TC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/iptables.html"><strong aria-hidden="true">3.13.</strong> IPTables</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/systemd.html"><strong aria-hidden="true">3.14.</strong> systemd</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/sysctl.html"><strong aria-hidden="true">3.15.</strong> sysctl</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/sysfs.html"><strong aria-hidden="true">3.16.</strong> sysfs</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/rtp.html"><strong aria-hidden="true">3.17.</strong> Rtp</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/filesystems.html"><strong aria-hidden="true">3.18.</strong> Filesystems</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/namespace.html"><strong aria-hidden="true">3.19.</strong> Namespace</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/selinux.html"><strong aria-hidden="true">3.20.</strong> Selinux</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/udev.html"><strong aria-hidden="true">3.21.</strong> Udev</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/process.html"><strong aria-hidden="true">3.22.</strong> Process</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="linux/wireguard.html"><strong aria-hidden="true">3.23.</strong> Wireguard</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="android/index.html"><strong aria-hidden="true">4.</strong> Android</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="android/internals.html"><strong aria-hidden="true">4.1.</strong> Android Internals</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="android/binder.html"><strong aria-hidden="true">4.2.</strong> Binder</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="android/adb.html"><strong aria-hidden="true">4.3.</strong> adb</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="android/platform_dev.html"><strong aria-hidden="true">4.4.</strong> Platform Dev</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/index.html"><strong aria-hidden="true">5.</strong> Data Structures</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/arrays.html"><strong aria-hidden="true">5.1.</strong> Arrays</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/linked_lists.html"><strong aria-hidden="true">5.2.</strong> Linked Lists</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/stacks.html"><strong aria-hidden="true">5.3.</strong> Stacks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/queues.html"><strong aria-hidden="true">5.4.</strong> Queues</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/hash_tables.html"><strong aria-hidden="true">5.5.</strong> Hash Tables</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/trees.html"><strong aria-hidden="true">5.6.</strong> Trees</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/graphs.html"><strong aria-hidden="true">5.7.</strong> Graphs</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/heaps.html"><strong aria-hidden="true">5.8.</strong> Heaps</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/tries.html"><strong aria-hidden="true">5.9.</strong> Tries</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="data_structures/bloom_filter.html"><strong aria-hidden="true">5.10.</strong> Bloom Filter</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/index.html"><strong aria-hidden="true">6.</strong> Algorithms</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/big_o.html"><strong aria-hidden="true">6.1.</strong> Big O</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/recursion.html"><strong aria-hidden="true">6.2.</strong> Recursion</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/dynamic_programming.html"><strong aria-hidden="true">6.3.</strong> Dynamic Programming</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/backtracking.html"><strong aria-hidden="true">6.4.</strong> Backtracking</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/divide_and_conquer.html"><strong aria-hidden="true">6.5.</strong> Divide and Conquer</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/greedy_algorithms.html"><strong aria-hidden="true">6.6.</strong> Greedy Algorithms</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/sorting.html"><strong aria-hidden="true">6.7.</strong> Sorting</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/searching.html"><strong aria-hidden="true">6.8.</strong> Searching</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/raft.html"><strong aria-hidden="true">6.9.</strong> Raft Consensus Algorithm</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/graph_algorithms.html"><strong aria-hidden="true">6.10.</strong> Graph Algorithms</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="algorithms/string_algorithms.html"><strong aria-hidden="true">6.11.</strong> String Algorithms</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/index.html"><strong aria-hidden="true">7.</strong> Security</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/hashing.html"><strong aria-hidden="true">7.1.</strong> Hashing</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/encryption.html"><strong aria-hidden="true">7.2.</strong> Encryption</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/digital_signatures.html"><strong aria-hidden="true">7.3.</strong> Digital Signatures</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/certificates.html"><strong aria-hidden="true">7.4.</strong> Certificates</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/ssl_tls.html"><strong aria-hidden="true">7.5.</strong> SSL/TLS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/hmac.html"><strong aria-hidden="true">7.6.</strong> HMAC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/oauth2.html"><strong aria-hidden="true">7.7.</strong> OAuth 2.0</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/jwt.html"><strong aria-hidden="true">7.8.</strong> JWT</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="security/auth.html"><strong aria-hidden="true">7.9.</strong> Auth</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/index.html"><strong aria-hidden="true">8.</strong> Wifi</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/basics.html"><strong aria-hidden="true">8.1.</strong> Wifi Basics</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/standards.html"><strong aria-hidden="true">8.2.</strong> Wifi Standards</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/security.html"><strong aria-hidden="true">8.3.</strong> Wifi Security</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/scanning.html"><strong aria-hidden="true">8.4.</strong> Scanning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/roaming.html"><strong aria-hidden="true">8.5.</strong> Roaming</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/qos_management.html"><strong aria-hidden="true">8.6.</strong> QoS Management</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/eap.html"><strong aria-hidden="true">8.7.</strong> Eap</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="wifi/ofdma.html"><strong aria-hidden="true">8.8.</strong> Ofdma</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/index.html"><strong aria-hidden="true">9.</strong> Machine Learning</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/deep_learning.html"><strong aria-hidden="true">9.1.</strong> Deep Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/neural_networks.html"><strong aria-hidden="true">9.2.</strong> Neural Networks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/supervised_learning.html"><strong aria-hidden="true">9.3.</strong> Supervised Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/unsupervised_learning.html"><strong aria-hidden="true">9.4.</strong> Unsupervised Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/reinforcement_learning.html"><strong aria-hidden="true">9.5.</strong> Reinforcement Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/deep_reinforcement_learning.html"><strong aria-hidden="true">9.6.</strong> Deep Reinforcement Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/generative_models.html"><strong aria-hidden="true">9.7.</strong> Generative Models</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/deep_generative_models.html"><strong aria-hidden="true">9.8.</strong> Deep Generative Models</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/transfer_learning.html"><strong aria-hidden="true">9.9.</strong> Transfer Learning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/transformers.html"><strong aria-hidden="true">9.10.</strong> Transformers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/hugging_face.html"><strong aria-hidden="true">9.11.</strong> Hugging Face</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/pytorch.html"><strong aria-hidden="true">9.12.</strong> PyTorch</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/numpy.html"><strong aria-hidden="true">9.13.</strong> NumPy</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/quantization.html"><strong aria-hidden="true">9.14.</strong> Quantization</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/interesting_papers.html"><strong aria-hidden="true">9.15.</strong> Interesting Papers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/lora.html"><strong aria-hidden="true">9.16.</strong> Lora</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="machine_learning/cuda.html"><strong aria-hidden="true">9.17.</strong> Cuda</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/index.html"><strong aria-hidden="true">10.</strong> AI</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/generative_ai.html"><strong aria-hidden="true">10.1.</strong> Generative AI</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/llms.html"><strong aria-hidden="true">10.2.</strong> LLMs</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/prompt_engineering.html"><strong aria-hidden="true">10.3.</strong> Prompt Engineering</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/llama.html"><strong aria-hidden="true">10.4.</strong> LLAMA</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/stable_diffusion.html"><strong aria-hidden="true">10.5.</strong> Stable Diffusion</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/fluxdev.html"><strong aria-hidden="true">10.6.</strong> Fluxdev</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/comfyui.html"><strong aria-hidden="true">10.7.</strong> ComfyUI</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/fine_tuning.html"><strong aria-hidden="true">10.8.</strong> Fine Tuning</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/deepseek_r1.html"><strong aria-hidden="true">10.9.</strong> Deepseek R1</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/whisper.html"><strong aria-hidden="true">10.10.</strong> Whisper</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/phi.html"><strong aria-hidden="true">10.11.</strong> Phi</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/vllm.html"><strong aria-hidden="true">10.12.</strong> Vllm</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/software_dev_prompts.html"><strong aria-hidden="true">10.13.</strong> Software Dev Prompts</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/agent_frameworks.html"><strong aria-hidden="true">10.14.</strong> Agent Frameworks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="ai/tool_use.html"><strong aria-hidden="true">10.15.</strong> Tool Use</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="cloud/index.html"><strong aria-hidden="true">11.</strong> Cloud</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="cloud/setup.html"><strong aria-hidden="true">11.1.</strong> Setup</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="cloud/google_cloud.html"><strong aria-hidden="true">11.2.</strong> Google Cloud</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="cloud/aws.html"><strong aria-hidden="true">11.3.</strong> AWS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="cloud/azure.html"><strong aria-hidden="true">11.4.</strong> Azure</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/index.html"><strong aria-hidden="true">12.</strong> Tools</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/tmux.html"><strong aria-hidden="true">12.1.</strong> tmux</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/vim.html"><strong aria-hidden="true">12.2.</strong> vim</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/cscope.html"><strong aria-hidden="true">12.3.</strong> cscope</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/ctags.html"><strong aria-hidden="true">12.4.</strong> ctags</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/mdbook.html"><strong aria-hidden="true">12.5.</strong> mdbook</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/sed.html"><strong aria-hidden="true">12.6.</strong> sed</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/awk.html"><strong aria-hidden="true">12.7.</strong> awk</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/curl.html"><strong aria-hidden="true">12.8.</strong> curl</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/wget.html"><strong aria-hidden="true">12.9.</strong> wget</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/grep.html"><strong aria-hidden="true">12.10.</strong> grep</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/find.html"><strong aria-hidden="true">12.11.</strong> find</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/ffmpeg.html"><strong aria-hidden="true">12.12.</strong> ffmpeg</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/make.html"><strong aria-hidden="true">12.13.</strong> make</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/docker.html"><strong aria-hidden="true">12.14.</strong> Docker</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/ansible.html"><strong aria-hidden="true">12.15.</strong> Ansible</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/wpa_supplicant.html"><strong aria-hidden="true">12.16.</strong> wpa_supplicant</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/hostapd.html"><strong aria-hidden="true">12.17.</strong> hostapd</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/nmap.html"><strong aria-hidden="true">12.18.</strong> Nmap</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/tshark.html"><strong aria-hidden="true">12.19.</strong> Tshark</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/wireshark.html"><strong aria-hidden="true">12.20.</strong> Wireshark</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/bazel.html"><strong aria-hidden="true">12.21.</strong> Bazel</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/clang.html"><strong aria-hidden="true">12.22.</strong> Clang</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/gcc.html"><strong aria-hidden="true">12.23.</strong> Gcc</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/ninja.html"><strong aria-hidden="true">12.24.</strong> Ninja</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/ripgrep.html"><strong aria-hidden="true">12.25.</strong> Ripgrep</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="tools/tcpdump.html"><strong aria-hidden="true">12.26.</strong> Tcpdump</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/index.html"><strong aria-hidden="true">13.</strong> Embedded</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/avr.html"><strong aria-hidden="true">13.1.</strong> AVR</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/stm32.html"><strong aria-hidden="true">13.2.</strong> STM32</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/esp32.html"><strong aria-hidden="true">13.3.</strong> ESP32</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/raspberry_pi.html"><strong aria-hidden="true">13.4.</strong> Raspberry Pi</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/arduino.html"><strong aria-hidden="true">13.5.</strong> Arduino</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/spi.html"><strong aria-hidden="true">13.6.</strong> SPI</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/i2c.html"><strong aria-hidden="true">13.7.</strong> I2C</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/uart.html"><strong aria-hidden="true">13.8.</strong> UART</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/usb.html"><strong aria-hidden="true">13.9.</strong> USB</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/can.html"><strong aria-hidden="true">13.10.</strong> CAN</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/sdio.html"><strong aria-hidden="true">13.11.</strong> SDIO</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/ethernet.html"><strong aria-hidden="true">13.12.</strong> Ethernet</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/pwm.html"><strong aria-hidden="true">13.13.</strong> PWM</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/adc.html"><strong aria-hidden="true">13.14.</strong> ADC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/dac.html"><strong aria-hidden="true">13.15.</strong> DAC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/rtc.html"><strong aria-hidden="true">13.16.</strong> RTC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/gpio.html"><strong aria-hidden="true">13.17.</strong> GPIO</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/interrupts.html"><strong aria-hidden="true">13.18.</strong> Interrupts</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/timers.html"><strong aria-hidden="true">13.19.</strong> Timers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/watchdog.html"><strong aria-hidden="true">13.20.</strong> Watchdog</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/power_management.html"><strong aria-hidden="true">13.21.</strong> Power Management</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="embedded/debugging.html"><strong aria-hidden="true">13.22.</strong> Debugging</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/index.html"><strong aria-hidden="true">14.</strong> Networking</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/osi_model.html"><strong aria-hidden="true">14.1.</strong> OSI Model</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/tcp_ip_model.html"><strong aria-hidden="true">14.2.</strong> TCP/IP Model</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/ip.html"><strong aria-hidden="true">14.3.</strong> IP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/ipv4.html"><strong aria-hidden="true">14.4.</strong> IPv4</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/ipv6.html"><strong aria-hidden="true">14.5.</strong> IPv6</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/tcp.html"><strong aria-hidden="true">14.6.</strong> TCP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/udp.html"><strong aria-hidden="true">14.7.</strong> UDP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/http.html"><strong aria-hidden="true">14.8.</strong> HTTP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/dns.html"><strong aria-hidden="true">14.9.</strong> DNS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/mdns.html"><strong aria-hidden="true">14.10.</strong> mDNS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/firewalls.html"><strong aria-hidden="true">14.11.</strong> Firewalls</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/stun.html"><strong aria-hidden="true">14.12.</strong> STUN</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/turn.html"><strong aria-hidden="true">14.13.</strong> TURN</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/ice.html"><strong aria-hidden="true">14.14.</strong> ICE</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/pcp.html"><strong aria-hidden="true">14.15.</strong> PCP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/nat_pmp.html"><strong aria-hidden="true">14.16.</strong> NAT-PMP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/upnp.html"><strong aria-hidden="true">14.17.</strong> UPnP</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/websocket.html"><strong aria-hidden="true">14.18.</strong> WebSocket</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="networking/webrtc.html"><strong aria-hidden="true">14.19.</strong> WebRTC</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/index.html"><strong aria-hidden="true">15.</strong> Finance</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/general.html"><strong aria-hidden="true">15.1.</strong> General</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/technical_analysis.html"><strong aria-hidden="true">15.2.</strong> Technical Analysis</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/fundamental_analysis.html"><strong aria-hidden="true">15.3.</strong> Fundamental Analysis</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/stocks.html"><strong aria-hidden="true">15.4.</strong> Stocks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/options.html"><strong aria-hidden="true">15.5.</strong> Options</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/futures.html"><strong aria-hidden="true">15.6.</strong> Futures</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="finance/crypto.html"><strong aria-hidden="true">15.7.</strong> Crypto</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/index.html"><strong aria-hidden="true">16.</strong> Databases</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/database_design.html"><strong aria-hidden="true">16.1.</strong> Database Design</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/sql.html"><strong aria-hidden="true">16.2.</strong> SQL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/postgres.html"><strong aria-hidden="true">16.3.</strong> PostgreSQL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/sqlite.html"><strong aria-hidden="true">16.4.</strong> SQLite</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/duckdb.html"><strong aria-hidden="true">16.5.</strong> DuckDB</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/nosql.html"><strong aria-hidden="true">16.6.</strong> NoSQL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/mongodb.html"><strong aria-hidden="true">16.7.</strong> MongoDB</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/redis.html"><strong aria-hidden="true">16.8.</strong> Redis</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="databases/kafka.html"><strong aria-hidden="true">16.9.</strong> Apache Kafka</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/index.html"><strong aria-hidden="true">17.</strong> Web Development</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/react.html"><strong aria-hidden="true">17.1.</strong> React</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/nextjs.html"><strong aria-hidden="true">17.2.</strong> Next.js</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/vuejs.html"><strong aria-hidden="true">17.3.</strong> Vue.js</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/svelte.html"><strong aria-hidden="true">17.4.</strong> Svelte</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/sveltekit.html"><strong aria-hidden="true">17.5.</strong> SvelteKit</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/tailwind.html"><strong aria-hidden="true">17.6.</strong> Tailwind CSS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/expressjs.html"><strong aria-hidden="true">17.7.</strong> Express.js</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/nestjs.html"><strong aria-hidden="true">17.8.</strong> NestJS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/django.html"><strong aria-hidden="true">17.9.</strong> Django</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/flask.html"><strong aria-hidden="true">17.10.</strong> Flask</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/fastapi.html"><strong aria-hidden="true">17.11.</strong> FastAPI</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/webassembly.html"><strong aria-hidden="true">17.12.</strong> WebAssembly</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/web_apis.html"><strong aria-hidden="true">17.13.</strong> Web APIs</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/rest_apis.html"><strong aria-hidden="true">17.14.</strong> REST APIs</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/graphql.html"><strong aria-hidden="true">17.15.</strong> GraphQL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/grpc.html"><strong aria-hidden="true">17.16.</strong> gRPC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/css.html"><strong aria-hidden="true">17.17.</strong> Css</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/api_design.html"><strong aria-hidden="true">17.18.</strong> Api Design</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="web_development/frontend_performance.html"><strong aria-hidden="true">17.19.</strong> Frontend Performance</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/index.html"><strong aria-hidden="true">18.</strong> DevOps</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/docker.html"><strong aria-hidden="true">18.1.</strong> Docker</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/kubernetes.html"><strong aria-hidden="true">18.2.</strong> Kubernetes</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/cicd.html"><strong aria-hidden="true">18.3.</strong> CI/CD</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/cloud-deployment.html"><strong aria-hidden="true">18.4.</strong> Cloud-Deployment</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/github-actions.html"><strong aria-hidden="true">18.5.</strong> Github-Actions</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/infrastructure.html"><strong aria-hidden="true">18.6.</strong> Infrastructure</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/monitoring.html"><strong aria-hidden="true">18.7.</strong> Monitoring</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/terraform.html"><strong aria-hidden="true">18.8.</strong> Terraform</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="devops/observability.html"><strong aria-hidden="true">18.9.</strong> Observability</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/index.html"><strong aria-hidden="true">19.</strong> System Design</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/scalability.html"><strong aria-hidden="true">19.1.</strong> Scalability</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/caching.html"><strong aria-hidden="true">19.2.</strong> Caching</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/rpc.html"><strong aria-hidden="true">19.3.</strong> RPC</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/microservices.html"><strong aria-hidden="true">19.4.</strong> Microservices</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/databases.html"><strong aria-hidden="true">19.5.</strong> Databases</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/design_patterns.html"><strong aria-hidden="true">19.6.</strong> Design Patterns</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/distributed_consensus.html"><strong aria-hidden="true">19.7.</strong> Distributed Consensus</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/message_queues.html"><strong aria-hidden="true">19.8.</strong> Message Queues</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/distributed_systems.html"><strong aria-hidden="true">19.9.</strong> Distributed Systems</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="system_design/load_balancing.html"><strong aria-hidden="true">19.10.</strong> Load Balancing</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="mobile_development/index.html"><strong aria-hidden="true">20.</strong> Mobile Development</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="mobile_development/react_native.html"><strong aria-hidden="true">20.1.</strong> React Native</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="mobile_development/flutter.html"><strong aria-hidden="true">20.2.</strong> Flutter</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="mobile_development/android_dev.html"><strong aria-hidden="true">20.3.</strong> Android Dev</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="mobile_development/ios_dev.html"><strong aria-hidden="true">20.4.</strong> Ios Dev</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/index.html"><strong aria-hidden="true">21.</strong> Testing</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/unit_testing.html"><strong aria-hidden="true">21.1.</strong> Unit Testing</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/integration.html"><strong aria-hidden="true">21.2.</strong> Integration Testing</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/tdd.html"><strong aria-hidden="true">21.3.</strong> TDD</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/pytest.html"><strong aria-hidden="true">21.4.</strong> pytest</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/e2e_testing.html"><strong aria-hidden="true">21.5.</strong> E2E Testing</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="debugging/index.html"><strong aria-hidden="true">22.</strong> Debugging</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="debugging/gdb.html"><strong aria-hidden="true">22.1.</strong> GDB</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="debugging/tools.html"><strong aria-hidden="true">22.2.</strong> Binary Analysis Tools</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="debugging/core_dump.html"><strong aria-hidden="true">22.3.</strong> Core Dump Analysis</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="debugging/linux_kernel.html"><strong aria-hidden="true">22.4.</strong> Linux Kernel Debugging</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/index.html"><strong aria-hidden="true">23.</strong> Miscellaneous</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/math.html"><strong aria-hidden="true">23.1.</strong> Mathematics</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/statistics.html"><strong aria-hidden="true">23.2.</strong> Statistics</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/matplotlib.html"><strong aria-hidden="true">23.3.</strong> Matplotlib</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/pandas.html"><strong aria-hidden="true">23.4.</strong> Pandas</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/blockchain.html"><strong aria-hidden="true">23.5.</strong> Blockchain</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/operating_systems.html"><strong aria-hidden="true">23.6.</strong> Operating Systems</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/computer_graphics.html"><strong aria-hidden="true">23.7.</strong> Computer Graphics</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/uboot.html"><strong aria-hidden="true">23.8.</strong> Uboot</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="misc/ubuntu.html"><strong aria-hidden="true">23.9.</strong> Ubuntu</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="rtos/index.html"><strong aria-hidden="true">24.</strong> RTOS</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="rtos/freertos.html"><strong aria-hidden="true">24.1.</strong> FreeRTOS</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="rtos/threadx.html"><strong aria-hidden="true">24.2.</strong> ThreadX</a></span></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        const sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level >= level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

