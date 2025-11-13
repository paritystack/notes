# curl

curl (Client URL) is a command-line tool and library for transferring data with URLs. It supports a wide range of protocols including HTTP, HTTPS, FTP, FTPS, SCP, SFTP, TFTP, and more.

## Overview

curl is one of the most versatile tools for testing APIs, downloading files, and debugging network requests. It's available on virtually all platforms and is commonly used in scripts and automation.

**Key Features:**
- Support for numerous protocols (HTTP, HTTPS, FTP, SMTP, etc.)
- Authentication support (Basic, Digest, OAuth, etc.)
- SSL/TLS support
- Cookie handling
- Resume transfers
- Proxy support
- Rate limiting
- Custom headers and methods

## Basic Usage

### Simple GET Request

```bash
# Basic GET request
curl https://api.example.com

# GET with output to file
curl https://example.com -o output.html
curl https://example.com --output output.html

# Save with original filename
curl -O https://example.com/file.pdf

# Follow redirects
curl -L https://shortened-url.com
```

### Viewing Response Details

```bash
# Show response headers only
curl -I https://api.example.com
curl --head https://api.example.com

# Show response headers and body
curl -i https://api.example.com
curl --include https://api.example.com

# Verbose output (shows request/response details)
curl -v https://api.example.com
curl --verbose https://api.example.com

# Show only HTTP status code
curl -o /dev/null -s -w "%{http_code}\n" https://api.example.com
```

## HTTP Methods

### GET Request

```bash
# GET with query parameters
curl "https://api.example.com/users?page=1&limit=10"

# GET with URL-encoded parameters
curl -G https://api.example.com/search \
  -d "query=curl tutorial" \
  -d "limit=5"
```

### POST Request

```bash
# POST with form data
curl -X POST https://api.example.com/users \
  -d "name=John" \
  -d "email=john@example.com"

# POST with JSON data
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@example.com"}'

# POST with JSON from file
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d @data.json

# POST with form file upload
curl -X POST https://api.example.com/upload \
  -F "file=@document.pdf" \
  -F "description=My document"
```

### PUT Request

```bash
# PUT to update resource
curl -X PUT https://api.example.com/users/123 \
  -H "Content-Type: application/json" \
  -d '{"name":"John Updated","email":"john.new@example.com"}'
```

### PATCH Request

```bash
# PATCH to partially update resource
curl -X PATCH https://api.example.com/users/123 \
  -H "Content-Type: application/json" \
  -d '{"email":"newemail@example.com"}'
```

### DELETE Request

```bash
# DELETE a resource
curl -X DELETE https://api.example.com/users/123

# DELETE with authentication
curl -X DELETE https://api.example.com/users/123 \
  -H "Authorization: Bearer token123"
```

## Headers

### Custom Headers

```bash
# Single custom header
curl -H "X-Custom-Header: value" https://api.example.com

# Multiple headers
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer token123" \
     -H "X-Request-ID: abc123" \
     https://api.example.com

# User-Agent header
curl -A "MyApp/1.0" https://api.example.com
curl --user-agent "MyApp/1.0" https://api.example.com

# Referer header
curl -e "https://referrer.com" https://api.example.com
curl --referer "https://referrer.com" https://api.example.com
```

### Accept Headers

```bash
# Request JSON response
curl -H "Accept: application/json" https://api.example.com

# Request XML response
curl -H "Accept: application/xml" https://api.example.com

# Request specific API version
curl -H "Accept: application/vnd.api+json; version=2" https://api.example.com
```

## Authentication

### Basic Authentication

```bash
# Basic auth (username:password)
curl -u username:password https://api.example.com

# Basic auth with prompt for password
curl -u username https://api.example.com

# Basic auth in URL (not recommended for production)
curl https://username:password@api.example.com
```

### Bearer Token

```bash
# Bearer token authentication
curl -H "Authorization: Bearer your_token_here" https://api.example.com

# Using environment variable
export TOKEN="your_token_here"
curl -H "Authorization: Bearer $TOKEN" https://api.example.com
```

### API Key

```bash
# API key in header
curl -H "X-API-Key: your_api_key" https://api.example.com

# API key in query parameter
curl "https://api.example.com/data?api_key=your_api_key"
```

### OAuth 2.0

```bash
# OAuth 2.0 with access token
curl -H "Authorization: Bearer access_token" https://api.example.com

# Get OAuth token
curl -X POST https://auth.example.com/token \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_client_secret"
```

## Cookies

### Managing Cookies

```bash
# Save cookies to file
curl -c cookies.txt https://example.com/login \
  -d "username=user&password=pass"

# Load cookies from file
curl -b cookies.txt https://example.com/profile

# Send cookies directly
curl -b "session=abc123; user=john" https://example.com

# Save and load cookies in same request
curl -b cookies.txt -c cookies.txt https://example.com
```

## File Operations

### Downloading Files

```bash
# Download single file
curl -O https://example.com/file.zip

# Download with custom name
curl -o myfile.zip https://example.com/file.zip

# Download multiple files
curl -O https://example.com/file1.zip \
     -O https://example.com/file2.zip

# Resume interrupted download
curl -C - -O https://example.com/largefile.zip

# Download with progress bar
curl -# -O https://example.com/file.zip
```

### Uploading Files

```bash
# Upload file with PUT
curl -X PUT https://api.example.com/files/document.pdf \
  --upload-file document.pdf

# Upload with POST multipart
curl -F "file=@document.pdf" https://api.example.com/upload

# Upload multiple files
curl -F "file1=@doc1.pdf" \
     -F "file2=@doc2.pdf" \
     https://api.example.com/upload
```

### FTP Operations

```bash
# Download from FTP
curl ftp://ftp.example.com/file.txt -u username:password

# Upload to FTP
curl -T localfile.txt ftp://ftp.example.com/ -u username:password

# List FTP directory
curl ftp://ftp.example.com/ -u username:password
```

## Advanced Options

### Timeouts

```bash
# Connection timeout (seconds)
curl --connect-timeout 10 https://api.example.com

# Maximum time for entire operation
curl --max-time 30 https://api.example.com
curl -m 30 https://api.example.com

# Keepalive time
curl --keepalive-time 60 https://api.example.com
```

### Retry Logic

```bash
# Retry on failure
curl --retry 3 https://api.example.com

# Retry with delay
curl --retry 3 --retry-delay 5 https://api.example.com

# Retry on specific errors
curl --retry 3 --retry-connrefused https://api.example.com
```

### Rate Limiting

```bash
# Limit download speed (K = kilobytes, M = megabytes)
curl --limit-rate 100K https://example.com/largefile.zip

# Limit upload speed
curl --limit-rate 50K -T file.zip https://example.com/upload
```

### Proxy

```bash
# Use HTTP proxy
curl -x http://proxy.example.com:8080 https://api.example.com

# Use SOCKS5 proxy
curl --socks5 proxy.example.com:1080 https://api.example.com

# Proxy with authentication
curl -x http://user:pass@proxy.example.com:8080 https://api.example.com

# Bypass proxy for specific hosts
curl --noproxy "localhost,127.0.0.1" -x proxy.example.com:8080 https://api.example.com
```

### SSL/TLS Options

```bash
# Ignore SSL certificate validation (unsafe - use only for testing)
curl -k https://self-signed.example.com
curl --insecure https://self-signed.example.com

# Specify SSL version
curl --tlsv1.2 https://api.example.com

# Use client certificate
curl --cert client.pem --key key.pem https://api.example.com

# Use CA certificate
curl --cacert ca-bundle.crt https://api.example.com
```

## Response Formatting

### Format Output

```bash
# Pretty print JSON response (with jq)
curl https://api.example.com/users | jq '.'

# Extract specific field from JSON
curl https://api.example.com/users | jq '.data[].name'

# Silent mode (no progress bar)
curl -s https://api.example.com

# Show only errors
curl -S -s https://api.example.com

# Output format string
curl -w "\nTime: %{time_total}s\nStatus: %{http_code}\n" https://api.example.com
```

### Custom Output Variables

```bash
# Show timing information
curl -w "
    time_namelookup:  %{time_namelookup}
    time_connect:     %{time_connect}
    time_appconnect:  %{time_appconnect}
    time_pretransfer: %{time_pretransfer}
    time_redirect:    %{time_redirect}
    time_starttransfer: %{time_starttransfer}
    time_total:       %{time_total}
    http_code:        %{http_code}
" -o /dev/null -s https://api.example.com

# Save format to file
curl -w "@curl-format.txt" -o /dev/null -s https://api.example.com
```

## Debugging

### Verbose Output

```bash
# Show detailed request/response
curl -v https://api.example.com

# Even more verbose (includes SSL info)
curl -vv https://api.example.com

# Trace ASCII
curl --trace-ascii debug.txt https://api.example.com

# Trace binary
curl --trace debug.bin https://api.example.com
```

### Testing APIs

```bash
# Test API endpoint
curl -I https://api.example.com/health

# Test with timeout
curl -m 5 https://api.example.com

# Check response time
time curl -o /dev/null -s https://api.example.com

# Test with different methods
for method in GET POST PUT DELETE; do
  echo "Testing $method:"
  curl -X $method -I https://api.example.com/test
done
```

## Common Patterns

### API Testing Script

```bash
#!/bin/bash
BASE_URL="https://api.example.com"
TOKEN="your_token_here"

# GET request
curl -H "Authorization: Bearer $TOKEN" "$BASE_URL/users"

# POST request
curl -X POST "$BASE_URL/users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@example.com"}'

# Check status
STATUS=$(curl -o /dev/null -s -w "%{http_code}" "$BASE_URL/health")
if [ "$STATUS" -eq 200 ]; then
  echo "API is healthy"
else
  echo "API returned status $STATUS"
fi
```

### Download with Progress

```bash
# Download with progress bar
curl -# -L -o file.zip https://example.com/download

# Download with custom progress
curl --progress-bar -o file.zip https://example.com/download
```

### REST API CRUD Operations

```bash
# Create
curl -X POST https://api.example.com/items \
  -H "Content-Type: application/json" \
  -d '{"name":"Item1","price":99.99}'

# Read
curl https://api.example.com/items/1

# Update
curl -X PUT https://api.example.com/items/1 \
  -H "Content-Type: application/json" \
  -d '{"name":"Item1 Updated","price":89.99}'

# Delete
curl -X DELETE https://api.example.com/items/1
```

## Configuration File

Create `~/.curlrc` for default options:

```bash
# Always follow redirects
-L

# Show error messages
--show-error

# Retry on failure
--retry 3

# Set user agent
user-agent = "MyApp/1.0"

# Always use HTTP/2 if available
--http2
```

## Best Practices

1. **Use verbose mode for debugging**
   ```bash
   curl -v https://api.example.com
   ```

2. **Always handle errors in scripts**
   ```bash
   if ! curl -f https://api.example.com; then
     echo "Request failed"
     exit 1
   fi
   ```

3. **Use environment variables for sensitive data**
   ```bash
   export API_TOKEN="secret"
   curl -H "Authorization: Bearer $API_TOKEN" https://api.example.com
   ```

4. **Set appropriate timeouts**
   ```bash
   curl --connect-timeout 10 --max-time 60 https://api.example.com
   ```

5. **Save and reuse cookies for session management**
   ```bash
   curl -c cookies.txt -d "user=john&pass=secret" https://example.com/login
   curl -b cookies.txt https://example.com/profile
   ```

## Common Use Cases

### Health Check Monitoring

```bash
#!/bin/bash
# Check if service is up
while true; do
  STATUS=$(curl -o /dev/null -s -w "%{http_code}" https://api.example.com/health)
  if [ "$STATUS" -eq 200 ]; then
    echo "$(date): Service is up"
  else
    echo "$(date): Service returned $STATUS"
  fi
  sleep 60
done
```

### API Load Testing

```bash
# Simple load test
for i in {1..100}; do
  curl -o /dev/null -s -w "%{time_total}\n" https://api.example.com &
done
wait
```

### Web Scraping

```bash
# Download webpage and extract links
curl -s https://example.com | grep -oP 'href="\K[^"]*'
```

### Testing Webhooks

```bash
# Send webhook payload
curl -X POST https://webhook.site/unique-url \
  -H "Content-Type: application/json" \
  -d '{"event":"user.created","data":{"id":123,"name":"John"}}'
```

## Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias curljson='curl -H "Content-Type: application/json"'
alias curlpost='curl -X POST -H "Content-Type: application/json"'
alias curltime='curl -w "\nTotal time: %{time_total}s\n"'
alias curlstatus='curl -o /dev/null -s -w "%{http_code}\n"'
```

## Common Options Reference

| Option | Description |
|--------|-------------|
| `-X, --request` | HTTP method (GET, POST, etc.) |
| `-H, --header` | Custom header |
| `-d, --data` | POST data |
| `-F, --form` | Multipart form data |
| `-o, --output` | Write to file |
| `-O, --remote-name` | Save with remote name |
| `-L, --location` | Follow redirects |
| `-i, --include` | Include headers in output |
| `-I, --head` | Fetch headers only |
| `-v, --verbose` | Verbose output |
| `-s, --silent` | Silent mode |
| `-u, --user` | Username:password |
| `-b, --cookie` | Cookie string or file |
| `-c, --cookie-jar` | Save cookies to file |
| `-A, --user-agent` | User-Agent string |
| `-e, --referer` | Referer URL |
| `-k, --insecure` | Ignore SSL errors |
| `-x, --proxy` | Use proxy |
| `-m, --max-time` | Maximum time in seconds |
| `--retry` | Number of retries |

## Troubleshooting

### Common Errors

```bash
# SSL certificate problem
curl --cacert /path/to/ca-bundle.crt https://example.com

# Connection timeout
curl --connect-timeout 30 https://example.com

# DNS resolution issues
curl --dns-servers 8.8.8.8 https://example.com

# Test specific IP
curl --resolve example.com:443:1.2.3.4 https://example.com
```

### Debug SSL Issues

```bash
# Show SSL certificate details
curl -vv https://example.com 2>&1 | grep -A 10 "SSL certificate"

# Test SSL handshake
openssl s_client -connect example.com:443

# Use specific TLS version
curl --tlsv1.2 https://example.com
```

curl is an incredibly powerful tool for working with APIs, testing endpoints, and automating HTTP requests. Master these patterns and you'll be able to handle almost any HTTP-related task from the command line.
