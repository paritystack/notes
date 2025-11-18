# Blockchain Website Crawling Guide

This guide provides best practices and configuration examples for crawling blockchain-related websites using this crawler library.

## Overview

Blockchain websites (explorers, DeFi platforms, documentation sites) often have unique characteristics that require special consideration when crawling:

- Dynamic content loaded via JavaScript
- Real-time data updates
- Rate limiting and API quotas
- Complex URL structures
- Large datasets
- WebSocket connections for live data
- Anti-bot protections (Cloudflare, etc.)
- High-frequency content updates
- Decentralized architectures

## Table of Contents

1. [Common Blockchain Website Types](#common-blockchain-website-types)
2. [Best Practices](#best-practices)
3. [Advanced Crawling Techniques](#advanced-crawling-techniques)
4. [Platform-Specific Guides](#platform-specific-guides)
5. [Example Configurations](#example-configurations)
6. [Data Extraction Patterns](#data-extraction-patterns)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Security and Privacy](#security-and-privacy)
10. [Ethical Considerations](#ethical-considerations)

## Common Blockchain Website Types

### Block Explorers

Block explorers (Etherscan, Blockchain.com, BscScan, etc.) display blockchain data through web interfaces, providing human-readable access to transaction data, smart contract information, and network statistics.

**Challenges:**
- Heavy JavaScript rendering (React, Vue.js applications)
- Infinite scroll patterns for transaction lists
- Aggressive rate limiting (often 1-5 requests/second)
- Dynamic content updates via WebSockets
- Anti-bot protections (Cloudflare, reCAPTCHA)
- Pagination tokens that expire quickly
- Content loaded asynchronously after initial page load

**Recommended Settings:**
```toml
[fetcher]
mode = "dynamic"  # Most explorers require JavaScript
required_element = "div.transaction-list"  # Validate content loaded
wait_time = 3.0  # Wait for async content to load
user_agent = "Mozilla/5.0 (compatible; ResearchBot/1.0)"

[network]
min_delay = 2.0
max_delay = 5.0  # Respect rate limits
retry_attempts = 3
timeout = 30.0  # Some pages take time to render

[patterns]
# Avoid crawling individual transaction/address pages
exclude = [
    ".*\\/tx\\/0x[a-fA-F0-9]{64}$",
    ".*\\/address\\/0x[a-fA-F0-9]{40}$",
    ".*\\/block\\/\\d+$"
]
```

**Specific Explorer Notes:**

**Etherscan (Ethereum):**
- Heavily JavaScript-dependent
- Requires dynamic mode for most pages
- Consider using their API instead for bulk data
- Free tier: 5 calls/second

**BscScan (Binance Smart Chain):**
- Similar structure to Etherscan (same codebase)
- Apply same patterns and delays

**Blockchain.com:**
- Mixed static/dynamic content
- Use auto mode for flexibility
- Better tolerance for static fetching on older pages

### Blockchain Documentation Sites

Documentation for blockchain protocols, APIs, and development tools. These sites typically use static site generators (Docusaurus, VuePress, GitBook) making them easier to crawl than dynamic explorers.

**Common Platforms:**
- Ethereum.org (Ethereum documentation)
- Solana Docs (Solana development guides)
- Polkadot Wiki (Substrate and Polkadot)
- Cosmos Hub Docs (Cosmos SDK)
- Hyperledger documentation
- Web3.js, Ethers.js, and other library docs

**Challenges:**
- Version-specific documentation paths
- Multiple language variants
- Deep navigation hierarchies
- Search pages with minimal content value
- API reference pages with generated content

**Example Configuration:**
```toml
[crawler]
name = "blockchain_docs"
start_urls = ["https://ethereum.org/en/developers/docs/"]
max_depth = 5
follow_redirects = true

[fetcher]
mode = "static"  # Docs often work with static fetching
user_agent = "DocumentationBot/1.0"

[patterns]
include = ["^https://ethereum\\.org/en/developers/.*"]
exclude = [
    ".*\\#.*",  # Exclude anchor links
    ".*\\/translations/.*",  # Exclude translation files
    ".*\\.pdf$",  # Exclude PDF downloads
    ".*\\/search\\?.*"  # Exclude search result pages
]

[network]
min_delay = 0.5
max_delay = 1.5  # Docs sites typically more tolerant
```

**Best Practices:**
- Check for versioned documentation (v1, v2, latest)
- Look for sidebar navigation to identify main sections
- Exclude community/forum sections if only docs needed
- Save markdown sources if available

### DeFi Platforms

Decentralized finance platforms with protocol information and analytics. These include DEXs (decentralized exchanges), lending protocols, yield aggregators, and DeFi analytics dashboards.

**Examples:**
- Uniswap (DEX)
- Aave (Lending protocol)
- DeFi Llama (Analytics)
- DeFi Pulse (DeFi tracking)
- Yearn Finance (Yield aggregator)

**Considerations:**
- Often heavily JavaScript-dependent (React SPAs)
- May require wallet connection (not crawlable via web scraping)
- Focus on informational pages only (about, docs, blog)
- Real-time price data via WebSocket (can't be scraped effectively)
- Many features are dApp-only (skip interactive features)

**Recommended Approach:**
```toml
[crawler]
name = "defi_info"
start_urls = ["https://docs.uniswap.org/"]
max_depth = 3

[fetcher]
mode = "auto"
required_element = "main"

[patterns]
include = [
    ".*\\/docs/.*",
    ".*\\/blog/.*",
    ".*\\/about.*"
]
exclude = [
    ".*\\/app/.*",  # Exclude dApp pages
    ".*\\/swap.*",  # Exclude interactive features
    ".*\\/pool.*"
]

[network]
min_delay = 2.0
max_delay = 4.0
```

**Note:** For real protocol data, use blockchain APIs or subgraphs instead of web scraping.

### NFT Marketplaces

NFT platforms like OpenSea, Rarible, and Magic Eden display digital collectibles and marketplace data.

**Challenges:**
- Heavy reliance on JavaScript and Web3
- Wallet-gated content
- Large image assets
- Infinite scroll collections
- Rate limiting

**Configuration:**
```toml
[crawler]
name = "nft_marketplace"
start_urls = ["https://opensea.io/learn"]
max_depth = 2

[fetcher]
mode = "dynamic"
wait_time = 4.0

[patterns]
include = [
    ".*\\/learn/.*",
    ".*\\/blog/.*",
    ".*\\/resources/.*"
]
exclude = [
    ".*\\/assets/.*",  # Skip individual NFT pages
    ".*\\/collection/.*"  # Skip collection pages
]

[network]
min_delay = 3.0
max_delay = 6.0
```

### Layer 2 and Sidechain Sites

Layer 2 solutions (Optimism, Arbitrum, Polygon) and their documentation.

**Characteristics:**
- Similar to L1 documentation
- Bridge interfaces (avoid crawling interactive tools)
- Network status dashboards

**Example:**
```toml
[crawler]
name = "layer2_docs"
start_urls = ["https://docs.optimism.io/"]
max_depth = 4

[fetcher]
mode = "static"

[patterns]
include = ["^https://docs\\.optimism\\.io/.*"]
exclude = [".*\\/bridge.*", ".*\\/gateway.*"]
```

### Blockchain News and Media

CoinDesk, CoinTelegraph, The Block, and other crypto news sites.

**Considerations:**
- Standard news site structure
- Heavy advertising (may slow page loads)
- Paywalls on some content
- Newsletter signup modals

**Configuration:**
```toml
[crawler]
name = "crypto_news"
start_urls = ["https://www.coindesk.com/"]
max_depth = 3

[fetcher]
mode = "auto"
required_element = "article"

[patterns]
include = [".*\\/\\d{4}\\/\\d{2}\\/\\d{2}/.*"]  # Date-based URLs
exclude = [
    ".*\\/tag/.*",
    ".*\\/author/.*",
    ".*\\/newsletter.*",
    ".*\\/sponsored.*"
]

[network]
min_delay = 1.5
max_delay = 3.0
```

## Best Practices

### 1. Respect Rate Limits

Blockchain services often have strict rate limiting due to infrastructure costs and abuse prevention:

```toml
[network]
min_delay = 3.0
max_delay = 10.0
concurrent_requests = 1  # Avoid parallel requests to same domain
```

**Rate Limit Guidelines by Platform:**
- **Etherscan/BscScan**: 5 req/sec (free), 15 req/sec (premium)
- **Blockchain.com**: ~10 req/min recommended
- **Documentation sites**: 1-2 req/sec typically safe
- **News sites**: 2-5 req/sec usually acceptable

**Detecting Rate Limits:**
- HTTP 429 (Too Many Requests)
- HTTP 403 (Forbidden) - may indicate blocking
- Cloudflare challenge pages
- Empty/error responses

### 2. Use Appropriate Fetching Mode

- **Static mode**: Documentation, blogs, static content
  - Faster and more efficient
  - Lower resource usage
  - Works for server-rendered pages

- **Dynamic mode**: Explorers, dashboards, real-time data
  - Required for JavaScript-heavy SPAs
  - Waits for content to render
  - Higher resource usage (headless browser)

- **Auto mode**: Mixed content types
  - Automatically detects need for JavaScript
  - Good for diverse content
  - Balances speed and compatibility

**Decision Matrix:**
```
Static:  Docs sites, blogs, simple news sites
Dynamic: Block explorers, DeFi dashboards, NFT marketplaces
Auto:    Mixed content, unknown site structure
```

### 3. Pattern Matching

Blockchain URLs often contain hashes and addresses. Use careful pattern matching to avoid crawling millions of transaction/address pages:

```toml
[patterns]
# Include specific sections
include = [
    "^https://etherscan\\.io/blocks.*",
    "^https://etherscan\\.io/txs.*",
    "^https://etherscan\\.io/charts.*"
]

# Exclude specific transaction/address pages to avoid infinite crawling
exclude = [
    ".*\\/tx\\/0x[a-fA-F0-9]{64}$",  # Individual transactions
    ".*\\/address\\/0x[a-fA-F0-9]{40}$",  # Ethereum addresses
    ".*\\/block\\/\\d+$",  # Individual blocks
    ".*\\/token/0x[a-fA-F0-9]{40}$",  # Token contract pages
    ".*\\/nft/.*",  # Individual NFT pages
    ".*\\?.*page=\\d+$"  # Pagination (if you don't want all pages)
]
```

**Common Blockchain Address Formats:**
```toml
# Ethereum/EVM: 0x + 40 hex chars
".*\\/0x[a-fA-F0-9]{40}.*"

# Bitcoin: varies, often starts with 1, 3, or bc1
".*\\/[13][a-km-zA-HJ-NP-Z1-9]{25,34}.*"
".*\\/bc1[a-z0-9]{39,87}.*"

# Solana: base58, typically 32-44 chars
".*\\/[1-9A-HJ-NP-Za-km-z]{32,44}.*"

# Transaction hashes (64 hex chars)
".*\\/tx\\/0x[a-fA-F0-9]{64}.*"
```

### 4. Content Validation

Ensure critical elements are loaded before saving:

```toml
[fetcher]
required_element = "div.container"  # Adjust based on target site
wait_time = 3.0  # Wait for async content
```

**Common validation elements:**
- Block explorers: `"div.transaction-list"`, `"table.transactions"`
- Documentation: `"article"`, `"main.content"`
- News sites: `"article.post"`, `"div.article-body"`

### 5. Storage Considerations

Blockchain data can be large. Monitor disk space and organize output:

```toml
[storage]
output_dir = "/path/to/large/storage/blockchain_data"
compress = true  # Enable compression if supported
max_file_size = 10485760  # 10MB limit per file
```

**Storage Tips:**
- Separate crawls by blockchain (ethereum/, solana/, etc.)
- Use date-based directories for news content
- Consider database storage for structured data
- Regular cleanup of old/duplicate content

### 6. User Agent Identification

Use descriptive user agents to help site operators understand your bot:

```toml
[fetcher]
user_agent = "ResearchBot/1.0 (+https://yoursite.com/bot-info)"
```

**Best practices:**
- Include bot name and version
- Provide contact URL or email
- Be honest about your purpose
- Don't impersonate regular browsers for evasion

### 7. Handle JavaScript-Heavy Sites

Many blockchain sites are SPAs (Single Page Applications):

```toml
[fetcher]
mode = "dynamic"
wait_time = 5.0
wait_for_network_idle = true  # Wait for API calls to complete
```

**Additional considerations:**
- Increase timeouts for slow-loading content
- Watch for infinite scroll (may never complete)
- Some content may require user interaction (skip it)

### 8. Monitor and Log

Enable comprehensive logging to track issues:

```toml
[logging]
level = "INFO"
log_file = "blockchain_crawler.log"
log_errors = true
```

**What to monitor:**
- Rate limit errors (HTTP 429)
- Timeout errors
- Content validation failures
- Redirect chains
- Storage usage

## Advanced Crawling Techniques

### Handling Infinite Scroll

Many blockchain sites use infinite scroll for transaction lists and other data:

**Problem:** Page never "completes" loading as new content continuously loads.

**Solutions:**

1. **Set maximum scroll depth:**
```toml
[fetcher]
mode = "dynamic"
max_scroll_depth = 5  # Scroll 5 times maximum
scroll_delay = 2.0  # Wait 2s between scrolls
```

2. **Use pagination URLs instead:**
```toml
[patterns]
# Target paginated URLs instead of infinite scroll
include = [".*\\?page=\\d+$"]
```

3. **Extract API endpoints:**
- Inspect network traffic to find API endpoints
- Use API directly instead of web scraping

### Bypassing Anti-Bot Protections

**Cloudflare and Similar Services:**

Some blockchain sites use Cloudflare or other anti-bot services.

**Approaches:**
```toml
[fetcher]
mode = "dynamic"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
wait_time = 5.0  # Wait for Cloudflare check
```

**Important:**
- Respect site terms of service
- Don't attempt to evade protections for malicious purposes
- Consider using official APIs instead
- Some protections are legitimate abuse prevention

### Dealing with WebSocket Data

Real-time blockchain data often uses WebSockets for live updates.

**Reality:** Web crawlers can't effectively capture WebSocket streams.

**Alternatives:**
- Focus on historical/static data
- Use blockchain node APIs (web3, ethers.js)
- Query indexing services (The Graph, Alchemy)
- Use exchange APIs for price data

### Multi-Chain Crawling

Crawling multiple blockchain ecosystems simultaneously:

```toml
# ethereum_config.toml
[crawler]
name = "ethereum_crawler"
start_urls = ["https://ethereum.org/en/"]

# solana_config.toml
[crawler]
name = "solana_crawler"
start_urls = ["https://solana.com/docs"]

# Run multiple crawlers:
# ./crawler --config ethereum_config.toml &
# ./crawler --config solana_config.toml &
```

**Coordination strategies:**
- Separate configuration files per chain
- Separate output directories
- Shared rate limit pool if crawling same domain
- Centralized logging for monitoring

### API Integration

Many blockchain sites have APIs that are more reliable than web scraping:

**When to use APIs instead:**
- Transaction data (use block explorer APIs)
- Token prices (use CoinGecko, CoinMarketCap APIs)
- Protocol metrics (use protocol-specific APIs)
- DeFi data (use DeFi Llama, The Graph)

**Hybrid approach:**
```
Documentation → Web crawling
Real-time data → APIs
Historical data → APIs or web scraping
Static content → Web crawling
```

## Platform-Specific Guides

### Ethereum Ecosystem

**Key Sites:**
- ethereum.org (documentation)
- etherscan.io (block explorer)
- docs.soliditylang.org (Solidity docs)

**Configuration:**
```toml
[crawler]
name = "ethereum_ecosystem"
start_urls = [
    "https://ethereum.org/en/developers/",
    "https://docs.soliditylang.org/en/latest/"
]
max_depth = 4

[fetcher]
mode = "static"

[patterns]
include = [
    "^https://ethereum\\.org/en/developers/.*",
    "^https://docs\\.soliditylang\\.org/.*"
]
exclude = [".*\\/translations/.*"]

[network]
min_delay = 1.0
max_delay = 2.0
```

### Solana Ecosystem

**Key Sites:**
- solana.com/docs
- solscan.io (block explorer)
- docs.metaplex.com (NFT standard)

**Considerations:**
- Heavily JavaScript-dependent sites
- Use dynamic mode for explorers
- Documentation uses modern static generators

```toml
[crawler]
name = "solana_docs"
start_urls = ["https://solana.com/docs"]
max_depth = 4

[fetcher]
mode = "static"

[patterns]
include = ["^https://solana\\.com/docs/.*"]

[network]
min_delay = 1.0
max_delay = 2.0
```

### Bitcoin Ecosystem

**Key Sites:**
- bitcoin.org/en/developer-documentation
- blockchain.com
- developer.bitcoin.org

**Characteristics:**
- More static content than newer blockchains
- Well-established documentation
- Block explorers with traditional pagination

```toml
[crawler]
name = "bitcoin_docs"
start_urls = ["https://developer.bitcoin.org/"]
max_depth = 3

[fetcher]
mode = "static"

[patterns]
include = ["^https://developer\\.bitcoin\\.org/.*"]
```

### Polkadot/Substrate Ecosystem

**Key Sites:**
- wiki.polkadot.network
- docs.substrate.io
- polkadot.js.org

**Configuration:**
```toml
[crawler]
name = "polkadot_ecosystem"
start_urls = [
    "https://wiki.polkadot.network/",
    "https://docs.substrate.io/"
]
max_depth = 5

[fetcher]
mode = "static"

[patterns]
include = [
    "^https://wiki\\.polkadot\\.network/.*",
    "^https://docs\\.substrate\\.io/.*"
]
exclude = [".*\\/en/.*"]  # If limiting to English only
```

## Data Extraction Patterns

### Extracting Structured Data

Beyond just saving HTML, you may want to extract specific data.

**Common extractions:**

1. **Documentation code examples:**
```javascript
// Look for code blocks
document.querySelectorAll('pre code')
```

2. **Transaction lists (if crawling lists):**
```javascript
// Extract transaction data from tables
document.querySelectorAll('table.transactions tr')
```

3. **Token/Protocol metadata:**
```javascript
// Extract protocol stats from pages
document.querySelector('.protocol-stats')
```

**Post-processing:**
- Parse HTML with BeautifulSoup, lxml, or similar
- Extract text, links, code blocks
- Store in structured format (JSON, CSV, database)

### URL Pattern Analysis

Understanding blockchain URL structures:

**Block Explorer Patterns:**
```
Transactions: /tx/[hash]
Addresses:    /address/[address]
Blocks:       /block/[number]
Tokens:       /token/[address]
Charts:       /chart/[metric]
```

**Documentation Patterns:**
```
Guides:       /docs/guides/[topic]
API Ref:      /docs/api/[endpoint]
Tutorials:    /docs/tutorials/[tutorial]
```

**Strategy:** Focus on list/index pages, avoid individual item pages.

### Content Deduplication

Blockchain sites often have duplicate content:

**Common duplicates:**
- Multiple language versions
- Versioned documentation (v1, v2, latest)
- Mirror sites
- Archived content

**Solutions:**
```toml
[patterns]
exclude = [
    ".*\\/v[0-9]+/.*",  # Exclude old versions
    ".*\\/[a-z]{2}/.*",  # Exclude non-English (adjust as needed)
    ".*\\/archive/.*"
]
```

## Performance Optimization

### Crawl Speed vs. Politeness

Balance between speed and being a good web citizen:

**Aggressive (use cautiously):**
```toml
[network]
min_delay = 0.5
max_delay = 1.0
concurrent_requests = 3
```

**Polite (recommended):**
```toml
[network]
min_delay = 2.0
max_delay = 5.0
concurrent_requests = 1
```

**Very polite (for sensitive sites):**
```toml
[network]
min_delay = 5.0
max_delay = 10.0
concurrent_requests = 1
```

### Resource Management

**Memory optimization:**
- Limit maximum depth to prevent explosion
- Use pagination limits
- Clear browser cache periodically (dynamic mode)

**Disk optimization:**
- Enable compression
- Limit file sizes
- Periodic cleanup of old data

**Network optimization:**
- Reuse connections where possible
- Enable HTTP/2 if supported
- Use conditional requests (If-Modified-Since)

### Parallel Crawling

Crawl multiple domains in parallel:

```bash
# Terminal 1
./crawler --config ethereum.toml

# Terminal 2
./crawler --config solana.toml

# Terminal 3
./crawler --config bitcoin.toml
```

**Important:** Don't parallelize requests to the same domain (respect rate limits).

## Troubleshooting

### Common Issues

#### 1. Empty Pages Saved

**Symptom:** HTML files saved but content missing.

**Causes:**
- JavaScript not executed (using static mode on dynamic site)
- Content not loaded before save
- Required element not found

**Solutions:**
```toml
[fetcher]
mode = "dynamic"  # Switch to dynamic
wait_time = 5.0  # Increase wait time
required_element = "main"  # Adjust validation element
```

#### 2. Rate Limited / Blocked

**Symptom:** HTTP 429, 403, or Cloudflare challenges.

**Solutions:**
```toml
[network]
min_delay = 5.0  # Increase delays
max_delay = 10.0

[fetcher]
user_agent = "YourBot/1.0 (+contact@email.com)"  # Identify yourself
```

- Check robots.txt
- Review site terms of service
- Consider using official API

#### 3. Timeouts

**Symptom:** Requests timing out, incomplete pages.

**Causes:**
- Slow site or network
- Site is overloaded
- Infinite loading content

**Solutions:**
```toml
[network]
timeout = 60.0  # Increase timeout

[fetcher]
wait_time = 10.0  # Wait longer for content
```

#### 4. Too Many URLs

**Symptom:** Crawler finds millions of URLs.

**Cause:** Not excluding transaction/address pages.

**Solution:**
```toml
[patterns]
exclude = [
    ".*\\/tx\\/.*",
    ".*\\/address\\/.*",
    ".*\\/block\\/\\d+$"
]

[crawler]
max_depth = 3  # Limit depth
max_pages = 10000  # Set maximum pages
```

#### 5. Duplicate Content

**Symptom:** Same content saved multiple times.

**Causes:**
- Multiple URLs for same content
- Query parameters
- Trailing slashes

**Solutions:**
```toml
[crawler]
normalize_urls = true  # Remove trailing slashes, etc.

[patterns]
exclude = [".*\\?.*"]  # Exclude query strings if not needed
```

### Debugging Tips

1. **Start small:**
```toml
[crawler]
max_depth = 1  # Test with limited depth first
max_pages = 10
```

2. **Enable verbose logging:**
```toml
[logging]
level = "DEBUG"
log_file = "debug.log"
```

3. **Test pattern matching:**
```bash
# Check what URLs match your patterns
./crawler --test-patterns --config yourconfig.toml
```

4. **Inspect saved content:**
```bash
# Verify pages saved correctly
ls -lh output/
head -n 50 output/page1.html
```

5. **Monitor in real-time:**
```bash
# Watch log file
tail -f blockchain_crawler.log
```

## Security and Privacy

### Data Handling

**Sensitive information in blockchain data:**
- Wallet addresses (public but can be tracked)
- Transaction amounts and patterns
- User behavior on platforms
- IP addresses (in logs)

**Best practices:**
- Don't republish scraped wallet addresses without context
- Respect user privacy even for public blockchain data
- Secure your scraped data storage
- Follow GDPR and data protection regulations

### Scraping Ethics

**Do:**
- Respect robots.txt
- Use reasonable rate limits
- Identify your bot with user agent
- Provide contact information
- Honor site terms of service
- Use official APIs when available

**Don't:**
- Overwhelm sites with requests
- Evade anti-bot protections maliciously
- Scrape wallet-gated or authenticated content
- Republish copyrighted content without permission
- Use scraped data for harassment or tracking

### Legal Considerations

**Important notes:**
- Web scraping legality varies by jurisdiction
- Review site terms of service
- Public data != license to republish
- Consider copyright on content
- Some sites explicitly forbid scraping

**Safer alternatives:**
- Use official APIs
- Request data access from site owners
- Use publicly available datasets
- Access blockchain data directly via nodes

### Secure Configuration

**Protect your crawler configuration:**
```toml
[auth]
# Never commit API keys to version control
api_key = "${ENV_API_KEY}"  # Use environment variables

[network]
# Use secure connections
https_only = true
verify_ssl = true
```

**Secure data storage:**
- Encrypt sensitive scraped data
- Limit file permissions (chmod 600)
- Use secure storage locations
- Regular backups with encryption

## Example Configurations

### Ethereum Documentation Crawler

Comprehensive configuration for crawling Ethereum developer documentation:

```toml
[crawler]
name = "ethereum_docs"
start_urls = ["https://ethereum.org/en/"]
max_depth = 4
max_pages = 5000

[fetcher]
mode = "static"
user_agent = "EthereumDocsCrawler/1.0 (+https://yoursite.com)"

[patterns]
include = [
    "^https://ethereum\\.org/en/developers/.*",
    "^https://ethereum\\.org/en/whitepaper.*"
]
exclude = [
    ".*\\/translations/.*",
    ".*\\.pdf$",
    ".*\\/contributing/.*"
]

[network]
min_delay = 1.0
max_delay = 3.0
timeout = 30.0
retry_attempts = 3

[storage]
output_dir = "./ethereum_docs"
db_path = "./ethereum_docs.db"

[logging]
level = "INFO"
log_file = "ethereum_crawler.log"
```

### Multi-Chain Block Explorer (Read-Only Pages)

For crawling informational pages on block explorers (not transaction/address pages):

```toml
[crawler]
name = "explorer_info"
start_urls = [
    "https://etherscan.io/charts",
    "https://etherscan.io/apis"
]
max_depth = 2
max_pages = 500

[fetcher]
mode = "dynamic"
required_element = "main"
wait_time = 4.0

[patterns]
include = [
    "^https://etherscan\\.io/charts/.*",
    "^https://etherscan\\.io/apis.*",
    "^https://etherscan\\.io/tokencheck.*"
]
exclude = [
    ".*\\/tx\\/0x[a-fA-F0-9]{64}$",
    ".*\\/address\\/0x[a-fA-F0-9]{40}$",
    ".*\\/block\\/\\d+$",
    ".*\\/txs\\?.*"  # Exclude transaction list pages
]

[network]
min_delay = 3.0
max_delay = 6.0
timeout = 45.0

[storage]
output_dir = "./explorer_data"
```

### DeFi Documentation Aggregator

Crawling documentation from multiple DeFi protocols:

```toml
[crawler]
name = "defi_docs"
start_urls = [
    "https://docs.uniswap.org/",
    "https://docs.aave.com/",
    "https://docs.compound.finance/"
]
max_depth = 3

[fetcher]
mode = "static"

[patterns]
include = [
    "^https://docs\\.uniswap\\.org/.*",
    "^https://docs\\.aave\\.com/.*",
    "^https://docs\\.compound\\.finance/.*"
]
exclude = [
    ".*\\/api/.*",  # Exclude API reference if too large
    ".*\\/v1/.*"    # Exclude old versions
]

[network]
min_delay = 1.5
max_delay = 3.0

[storage]
output_dir = "./defi_docs"
```

### Blockchain News Archive

For archiving blockchain news articles:

```toml
[crawler]
name = "crypto_news"
start_urls = ["https://www.coindesk.com/"]
max_depth = 3
max_pages = 2000

[fetcher]
mode = "auto"
required_element = "article"

[patterns]
include = [
    ".*\\/\\d{4}\\/\\d{2}\\/\\d{2}/.*"  # Date-based article URLs
]
exclude = [
    ".*\\/tag/.*",
    ".*\\/author/.*",
    ".*\\/newsletter.*",
    ".*\\/sponsored.*",
    ".*\\/press-releases/.*"
]

[network]
min_delay = 2.0
max_delay = 5.0
timeout = 30.0

[storage]
output_dir = "./crypto_news"
db_path = "./crypto_news.db"

[logging]
level = "INFO"
log_file = "news_crawler.log"
```

### NFT Marketplace Educational Content

Crawling learning resources from NFT marketplaces (not marketplace listings):

```toml
[crawler]
name = "nft_education"
start_urls = [
    "https://opensea.io/learn",
    "https://support.opensea.io/"
]
max_depth = 2

[fetcher]
mode = "dynamic"
wait_time = 3.0

[patterns]
include = [
    "^https://opensea\\.io/learn/.*",
    "^https://support\\.opensea\\.io/.*"
]
exclude = [
    ".*\\/collection/.*",
    ".*\\/assets/.*",
    ".*\\/account/.*"
]

[network]
min_delay = 2.5
max_delay = 5.0

[storage]
output_dir = "./nft_education"
```

### Layer 2 Documentation Complete

Comprehensive Layer 2 documentation crawl:

```toml
[crawler]
name = "layer2_comprehensive"
start_urls = [
    "https://docs.optimism.io/",
    "https://docs.arbitrum.io/",
    "https://wiki.polygon.technology/"
]
max_depth = 5

[fetcher]
mode = "static"

[patterns]
include = [
    "^https://docs\\.optimism\\.io/.*",
    "^https://docs\\.arbitrum\\.io/.*",
    "^https://wiki\\.polygon\\.technology/.*"
]
exclude = [
    ".*\\/bridge/.*",  # Exclude bridge interfaces
    ".*\\/translations/.*"
]

[network]
min_delay = 1.0
max_delay = 2.0

[storage]
output_dir = "./layer2_docs"
db_path = "./layer2_docs.db"
```

### Research-Focused Configuration

For academic/research purposes with maximum detail:

```toml
[crawler]
name = "blockchain_research"
start_urls = [
    "https://ethereum.org/en/",
    "https://bitcoin.org/en/developer-documentation"
]
max_depth = 6  # Deeper crawl
max_pages = 10000

[fetcher]
mode = "auto"
save_screenshots = true  # Save screenshots for analysis

[patterns]
include = [
    "^https://ethereum\\.org/en/developers/.*",
    "^https://ethereum\\.org/en/whitepaper.*",
    "^https://bitcoin\\.org/en/developer-.*"
]

[network]
min_delay = 2.0
max_delay = 4.0
timeout = 60.0

[storage]
output_dir = "./blockchain_research"
save_metadata = true  # Save crawl metadata

[logging]
level = "DEBUG"  # Verbose logging for research
log_file = "research_crawler.log"
```

## Ethical Considerations

When crawling blockchain-related websites:

### 1. Respect robots.txt

Always check and respect robots.txt directives:

```bash
# Check robots.txt before crawling
curl https://etherscan.io/robots.txt
```

**Important:**
- Some sites disallow all crawlers
- Others specify allowed paths
- Ignoring robots.txt may violate terms of service
- Can lead to IP bans or legal issues

### 2. Review Terms of Service

Many blockchain sites have specific ToS regarding automated access:

**Common restrictions:**
- Maximum request rates
- Prohibited content usage
- Attribution requirements
- Commercial use limitations

**Before crawling:**
- Read the site's terms of service
- Check for "API Terms" or "Developer Terms"
- Look for explicit scraping policies
- Consider contacting site operators for permission

### 3. Rate Limiting and Server Load

Don't overload services, especially free public infrastructure:

**Impact of aggressive crawling:**
- Increased server costs for operators
- Degraded service for legitimate users
- Potential service outages
- IP blocking or legal action

**Best practices:**
- Use conservative delays (2-5 seconds minimum)
- Crawl during off-peak hours
- Limit concurrent requests
- Monitor for error responses (429, 503)

### 4. Data Usage and Copyright

Respect copyright and licensing of crawled content:

**Considerations:**
- Documentation may be copyrighted
- Code examples may have specific licenses
- Images and graphics have separate rights
- Commercial use may require permission

**Proper usage:**
- Attribute sources appropriately
- Respect license terms (MIT, GPL, etc.)
- Don't republish as your own work
- Link back to original sources

### 5. Privacy and Blockchain Data

Be cautious with addresses and transaction data:

**Privacy concerns:**
- Wallet addresses are pseudonymous, not anonymous
- Transaction patterns can reveal identities
- Aggregate data can deanonymize users
- GDPR and privacy laws may apply

**Responsible handling:**
- Don't republish address/transaction mappings
- Aggregate data to protect privacy
- Follow data protection regulations
- Consider ethical implications of deanonymization

### 6. Community Impact

Your crawling affects the broader blockchain community:

**Positive contributions:**
- Archive important documentation
- Enable research and analysis
- Improve search and discovery
- Preserve historical data

**Negative impacts:**
- Strain on community resources
- Potential abuse of scraped data
- Violation of community trust
- Reduced availability for others

### 7. Alternative Approaches

Consider alternatives to web scraping:

**Better options:**
- **Official APIs**: Most sites provide APIs (Etherscan API, CoinGecko API)
- **GraphQL endpoints**: The Graph protocol for blockchain data
- **Node access**: Direct blockchain node queries (Infura, Alchemy)
- **Data services**: Paid data providers with proper licensing
- **Public datasets**: Existing archived datasets (Kaggle, etc.)
- **Partnerships**: Contact site operators for data access

### 8. Transparency

Be transparent about your bot:

**User agent:**
```toml
[fetcher]
user_agent = "ResearchBot/1.0 (+https://yoursite.com/bot-info; contact@email.com)"
```

**Bot info page should include:**
- Purpose of crawling
- Contact information
- Crawl frequency and scope
- Data usage policy
- Opt-out instructions

## Limitations

This crawler is designed for **public web content** only. It cannot and should not be used to:

### Technical Limitations

- **Interact with blockchain networks directly** - Use web3 libraries instead
- **Access wallet-gated content** - Requires authentication not suitable for crawlers
- **Execute smart contracts** - Use blockchain SDKs (web3.js, ethers.js)
- **Crawl APIs** - Use proper API clients with authentication
- **Capture WebSocket streams** - Real-time data requires different tools
- **Bypass CAPTCHAs** - Don't attempt to evade security measures
- **Handle infinite scroll effectively** - May miss dynamically loaded content

### Ethical Limitations

- **Don't scrape private/gated content** - Respect access controls
- **Don't evade rate limits** - Respect site protections
- **Don't impersonate users** - Use honest user agents
- **Don't crawl during attacks** - Avoid adding load during incidents
- **Don't sell scraped data** - Check licensing terms

### Legal Limitations

- **Comply with local laws** - Web scraping legality varies by jurisdiction
- **Respect intellectual property** - Don't violate copyright
- **Follow data protection laws** - GDPR, CCPA, etc.
- **Honor contracts** - ToS are often legally binding

## Additional Resources

### Blockchain Documentation

- [Ethereum Developer Documentation](https://ethereum.org/en/developers/)
- [Bitcoin Developer Guide](https://developer.bitcoin.org/)
- [Solana Documentation](https://solana.com/docs)
- [Polkadot Wiki](https://wiki.polkadot.network/)
- [Cosmos Documentation](https://docs.cosmos.network/)

### APIs and Data Services

- [Etherscan API](https://docs.etherscan.io/)
- [The Graph Protocol](https://thegraph.com/)
- [Alchemy API](https://www.alchemy.com/)
- [Infura API](https://www.infura.io/)
- [CoinGecko API](https://www.coingecko.com/en/api)
- [DeFi Llama API](https://defillama.com/docs/api)

### Web Scraping Best Practices

- [Web Scraping Best Practices](https://www.scraperapi.com/blog/web-scraping-best-practices/)
- [Ethical Web Scraping Guide](https://towardsdatascience.com/ethics-in-web-scraping-b96b18136f01)
- [robots.txt Specification](https://www.robotstxt.org/)

### Legal and Ethical Resources

- [EFF on Web Scraping](https://www.eff.org/)
- [GDPR Compliance](https://gdpr.eu/)
- [Creative Commons Licenses](https://creativecommons.org/licenses/)

## Support

For issues specific to blockchain website crawling, please provide:

**Required information:**
1. **Target website URL** - Full URL you're trying to crawl
2. **Configuration file** - Your complete TOML configuration
3. **Error messages** - Exact error text and log output
4. **Content type** - Static or JavaScript-rendered
5. **Crawl scope** - How many pages, what depth

**Helpful additional info:**
- Browser DevTools network tab screenshot
- robots.txt content from target site
- Example URLs that fail/succeed
- Your crawler version
- Operating system and environment

**Getting help:**
- Check documentation first
- Search existing issues
- Provide minimal reproducible example
- Include relevant logs (not entire log dump)
- Describe what you've already tried

## Quick Reference

### Blockchain URL Patterns to Exclude

```toml
[patterns]
exclude = [
    # Ethereum-style addresses and transactions
    ".*\\/tx\\/0x[a-fA-F0-9]{64}$",
    ".*\\/address\\/0x[a-fA-F0-9]{40}$",
    ".*\\/token\\/0x[a-fA-F0-9]{40}$",
    ".*\\/block\\/\\d+$",

    # Bitcoin addresses
    ".*\\/address/[13][a-km-zA-HJ-NP-Z1-9]{25,34}$",
    ".*\\/address/bc1[a-z0-9]{39,87}$",

    # Solana addresses
    ".*\\/address/[1-9A-HJ-NP-Za-km-z]{32,44}$",

    # Common excludes
    ".*\\/search\\?.*",
    ".*\\/translations/.*",
    ".*\\#.*",  # Anchors
    ".*\\.pdf$"
]
```

### Recommended Delays by Site Type

| Site Type | Min Delay | Max Delay | Notes |
|-----------|-----------|-----------|-------|
| Documentation | 0.5s | 1.5s | Usually tolerant |
| News sites | 1.5s | 3.0s | Standard politeness |
| Block explorers | 3.0s | 6.0s | Heavily rate limited |
| DeFi platforms | 2.0s | 4.0s | Moderate protection |
| NFT marketplaces | 3.0s | 6.0s | Heavy JavaScript |

### Fetcher Mode Selection

| Site Type | Mode | Reason |
|-----------|------|--------|
| Docs (Docusaurus, GitBook) | static | Server-rendered |
| Block explorers | dynamic | React/Vue SPAs |
| News sites | auto | Mixed content |
| DeFi dashboards | dynamic | Heavy JavaScript |
| Blogs | static | Traditional HTML |

## Conclusion

Blockchain website crawling requires careful consideration of technical, ethical, and legal factors. Always prioritize:

1. **Respect** - For site operators, users, and community
2. **Transparency** - Identify your bot and purpose
3. **Moderation** - Use conservative rate limits
4. **Legality** - Comply with laws and terms of service
5. **Alternatives** - Consider APIs and official data sources

When in doubt, err on the side of caution and reach out to site operators for permission.

Happy (ethical) crawling!

