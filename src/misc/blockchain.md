# Blockchain Website Crawling Guide

This guide provides best practices and configuration examples for crawling blockchain-related websites using this crawler library.

## Overview

Blockchain websites (explorers, DeFi platforms, documentation sites) often have unique characteristics that require special consideration when crawling:

- Dynamic content loaded via JavaScript
- Real-time data updates
- Rate limiting and API quotas
- Complex URL structures
- Large datasets

## Common Blockchain Website Types

### Block Explorers

Block explorers (Etherscan, Blockchain.com, etc.) display blockchain data through web interfaces.

**Challenges:**
- Heavy JavaScript rendering
- Infinite scroll patterns
- Rate limiting
- Dynamic content updates

**Recommended Settings:**
```toml
[fetcher]
mode = "dynamic"  # Most explorers require JavaScript
required_element = "div.transaction-list"  # Validate content loaded

[network]
min_delay = 2.0
max_delay = 5.0  # Respect rate limits
```

### Blockchain Documentation Sites

Documentation for blockchain protocols, APIs, and development tools.

**Example Configuration:**
```toml
[crawler]
name = "blockchain_docs"
start_urls = ["https://ethereum.org/en/developers/docs/"]
max_depth = 5

[fetcher]
mode = "static"  # Docs often work with static fetching

[patterns]
include = ["^https://ethereum\\.org/en/developers/.*"]
exclude = [".*\\#.*"]  # Exclude anchor links
```

### DeFi Platforms

Decentralized finance platforms with protocol information and analytics.

**Considerations:**
- Often heavily JavaScript-dependent
- May require wallet connection (not crawlable)
- Focus on informational pages only

## Best Practices

### 1. Respect Rate Limits

Blockchain services often have strict rate limiting:

```toml
[network]
min_delay = 3.0
max_delay = 10.0
```

### 2. Use Appropriate Fetching Mode

- **Static mode**: Documentation, blogs, static content
- **Dynamic mode**: Explorers, dashboards, real-time data
- **Auto mode**: Mixed content types

### 3. Pattern Matching

Blockchain URLs often contain hashes and addresses. Use careful pattern matching:

```toml
[patterns]
# Include specific sections
include = [
    "^https://etherscan\\.io/blocks.*",
    "^https://etherscan\\.io/txs.*"
]

# Exclude specific transaction/address pages to avoid infinite crawling
exclude = [
    ".*\\/tx\\/0x[a-fA-F0-9]{64}$",
    ".*\\/address\\/0x[a-fA-F0-9]{40}$"
]
```

### 4. Content Validation

Ensure critical elements are loaded before saving:

```toml
[fetcher]
required_element = "div.container"  # Adjust based on target site
```

### 5. Storage Considerations

Blockchain data can be large. Monitor disk space:

```toml
[storage]
output_dir = "/path/to/large/storage/blockchain_data"
```

## Example Configurations

### Ethereum Documentation Crawler

```toml
[crawler]
name = "ethereum_docs"
start_urls = ["https://ethereum.org/en/"]
max_depth = 4

[fetcher]
mode = "static"

[patterns]
include = ["^https://ethereum\\.org/en/.*"]
exclude = [".*\\/translations/.*", ".*\\.pdf$"]

[network]
min_delay = 1.0
max_delay = 3.0

[storage]
output_dir = "./ethereum_docs"
db_path = "./ethereum_docs.db"
```

### Blockchain News Site Crawler

```toml
[crawler]
name = "crypto_news"
start_urls = ["https://example-crypto-news.com/"]
max_depth = 3

[fetcher]
mode = "auto"

[patterns]
include = ["^https://example-crypto-news\\.com/articles/.*"]
exclude = [".*\\/tag/.*", ".*\\/author/.*"]

[network]
min_delay = 2.0
max_delay = 5.0
```

## Ethical Considerations

When crawling blockchain-related websites:

1. **Check robots.txt**: Always respect robots.txt directives
2. **Terms of Service**: Review and comply with site ToS
3. **Rate Limiting**: Don't overload services, especially free public APIs
4. **Data Usage**: Respect copyright and licensing of crawled content
5. **Privacy**: Be cautious with addresses and transaction data

## Limitations

This crawler is designed for **public web content** only. It cannot:

- Interact with blockchain networks directly
- Access wallet-gated content
- Execute smart contracts
- Crawl APIs (use API clients instead)

## Additional Resources

- [Ethereum Developer Documentation](https://ethereum.org/en/developers/)
- [Bitcoin Developer Guide](https://developer.bitcoin.org/)
- [Web Scraping Best Practices](https://www.scraperapi.com/blog/web-scraping-best-practices/)

## Support

For issues specific to blockchain website crawling, please provide:
- Target website URL
- Configuration file
- Error messages or unexpected behavior
- Whether content is static or JavaScript-rendered

