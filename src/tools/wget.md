# wget

wget is a free command-line utility for non-interactive downloading of files from the web. It supports HTTP, HTTPS, and FTP protocols, and can work through proxies, resume downloads, and handle various network conditions.

## Overview

wget is designed for robustness over slow or unstable network connections. If a download fails, it will keep retrying until the whole file has been retrieved. It's ideal for downloading files in scripts and automated tasks.

**Key Features:**
- Non-interactive operation (works in background)
- Resume interrupted downloads
- Recursive downloads (entire websites)
- Multiple protocol support (HTTP, HTTPS, FTP)
- Proxy support
- Timestamping and mirroring
- Convert links for offline viewing
- Bandwidth limiting

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install wget

# macOS
brew install wget

# CentOS/RHEL
sudo yum install wget

# Arch Linux
sudo pacman -S wget

# Verify installation
wget --version
```

## Basic Usage

### Simple Downloads

```bash
# Download a file
wget https://example.com/file.zip

# Download and save with different name
wget -O myfile.zip https://example.com/file.zip
wget --output-document=myfile.zip https://example.com/file.zip

# Download to specific directory
wget -P /path/to/directory https://example.com/file.zip
wget --directory-prefix=/path/to/directory https://example.com/file.zip

# Download in background
wget -b https://example.com/largefile.zip
wget --background https://example.com/largefile.zip

# Continue interrupted download
wget -c https://example.com/largefile.zip
wget --continue https://example.com/largefile.zip
```

### Multiple Files

```bash
# Download multiple files
wget https://example.com/file1.zip https://example.com/file2.zip

# Download from file list
cat urls.txt
# https://example.com/file1.zip
# https://example.com/file2.zip
# https://example.com/file3.zip

wget -i urls.txt
wget --input-file=urls.txt

# Download from URLs with wildcards
wget https://example.com/file{1..10}.zip
```

### Download Options

```bash
# Limit download speed (K, M, G)
wget --limit-rate=200k https://example.com/file.zip
wget --limit-rate=1M https://example.com/file.zip

# Set number of retries
wget --tries=10 https://example.com/file.zip
wget -t 10 https://example.com/file.zip

# Infinite retries
wget --tries=0 https://example.com/file.zip

# Timeout settings
wget --timeout=30 https://example.com/file.zip
wget --dns-timeout=10 --connect-timeout=10 --read-timeout=30 https://example.com/file.zip

# Wait between downloads
wget --wait=5 -i urls.txt  # Wait 5 seconds
wget --random-wait -i urls.txt  # Random wait 0.5-1.5x wait time
```

## Recursive Downloads

### Mirror Websites

```bash
# Mirror entire website
wget --mirror --convert-links --page-requisites --no-parent https://example.com

# Shorter version
wget -mkEpnp https://example.com

# Flags explained:
# -m, --mirror: mirror (recursive + timestamping + infinite depth)
# -k, --convert-links: convert links for offline viewing
# -E, --adjust-extension: save HTML with .html extension
# -p, --page-requisites: get all images, CSS, etc.
# -np, --no-parent: don't ascend to parent directory

# Limit recursion depth
wget -r -l 2 https://example.com  # 2 levels deep
wget --recursive --level=2 https://example.com

# Download specific file types only
wget -r -A pdf,jpg,png https://example.com
wget --recursive --accept=pdf,jpg,png https://example.com

# Exclude specific file types
wget -r -R gif,svg https://example.com
wget --recursive --reject=gif,svg https://example.com
```

### Download Directories

```bash
# Download entire directory
wget -r -np -nH --cut-dirs=2 https://example.com/files/documents/

# Flags explained:
# -r: recursive
# -np: no parent (stay in directory)
# -nH: no host directory
# --cut-dirs=2: skip 2 directory levels

# Example:
# URL: https://example.com/files/documents/pdf/file.pdf
# Without flags: example.com/files/documents/pdf/file.pdf
# With flags: pdf/file.pdf
```

## Authentication

### HTTP Authentication

```bash
# Basic authentication
wget --user=username --password=password https://example.com/file.zip

# Prompt for password
wget --user=username --ask-password https://example.com/file.zip

# HTTP authentication via .wgetrc
cat << EOF > ~/.wgetrc
http_user = username
http_password = password
EOF
```

### FTP Authentication

```bash
# FTP download with credentials
wget ftp://username:password@ftp.example.com/file.zip

# Anonymous FTP
wget ftp://ftp.example.com/file.zip
```

### Cookies

```bash
# Send cookies
wget --header="Cookie: session=abc123" https://example.com/file.zip

# Load cookies from file
wget --load-cookies=cookies.txt https://example.com/file.zip

# Save cookies to file
wget --save-cookies=cookies.txt --keep-session-cookies https://example.com/login

# Use cookies for authenticated download
wget --save-cookies=cookies.txt --keep-session-cookies \
     --post-data='user=john&pass=secret' \
     https://example.com/login
wget --load-cookies=cookies.txt https://example.com/protected/file.zip
```

## Headers and User Agent

### Custom Headers

```bash
# Set user agent
wget --user-agent="Mozilla/5.0" https://example.com/file.zip
wget -U "Mozilla/5.0" https://example.com/file.zip

# Custom headers
wget --header="Accept: application/json" https://api.example.com/data
wget --header="Authorization: Bearer token123" https://api.example.com/file.zip

# Multiple headers
wget --header="Accept: application/json" \
     --header="X-API-Key: abc123" \
     https://api.example.com/data

# Referer header
wget --referer=https://example.com https://example.com/file.zip
```

### POST Requests

```bash
# POST data
wget --post-data='name=John&email=john@example.com' https://example.com/api

# POST from file
wget --post-file=data.json https://example.com/api

# POST with headers
wget --post-data='{"name":"John"}' \
     --header="Content-Type: application/json" \
     https://example.com/api
```

## SSL/TLS Options

```bash
# Ignore SSL certificate check (unsafe)
wget --no-check-certificate https://self-signed.example.com/file.zip

# Specify CA certificate
wget --ca-certificate=/path/to/ca-cert.pem https://example.com/file.zip

# Use client certificate
wget --certificate=/path/to/client-cert.pem \
     --certificate-type=PEM \
     https://example.com/file.zip

# Use private key
wget --private-key=/path/to/key.pem https://example.com/file.zip

# Specify SSL protocol
wget --secure-protocol=TLSv1_2 https://example.com/file.zip
```

## Proxy Support

```bash
# Use HTTP proxy
wget -e use_proxy=yes -e http_proxy=http://proxy.example.com:8080 https://example.com/file.zip

# Use proxy with authentication
wget -e use_proxy=yes \
     -e http_proxy=http://user:pass@proxy.example.com:8080 \
     https://example.com/file.zip

# HTTPS proxy
wget -e https_proxy=http://proxy.example.com:8080 https://example.com/file.zip

# FTP proxy
wget -e ftp_proxy=http://proxy.example.com:8080 ftp://ftp.example.com/file.zip

# No proxy for specific domains
wget -e no_proxy=localhost,127.0.0.1 https://example.com/file.zip

# Configure in .wgetrc
cat << EOF > ~/.wgetrc
use_proxy = on
http_proxy = http://proxy.example.com:8080
https_proxy = http://proxy.example.com:8080
ftp_proxy = http://proxy.example.com:8080
no_proxy = localhost,127.0.0.1
EOF
```

## Output Control

### Verbosity

```bash
# Quiet mode (no output)
wget -q https://example.com/file.zip
wget --quiet https://example.com/file.zip

# Verbose output
wget -v https://example.com/file.zip
wget --verbose https://example.com/file.zip

# Debug output
wget -d https://example.com/file.zip
wget --debug https://example.com/file.zip

# Show progress bar only
wget --progress=bar https://example.com/file.zip
wget --progress=dot https://example.com/file.zip

# No verbose but show errors
wget -nv https://example.com/file.zip
wget --no-verbose https://example.com/file.zip
```

### Logging

```bash
# Log to file
wget -o download.log https://example.com/file.zip
wget --output-file=download.log https://example.com/file.zip

# Append to log
wget -a download.log https://example.com/file.zip
wget --append-output=download.log https://example.com/file.zip

# Background download with logging
wget -b -o wget.log https://example.com/largefile.zip
```

## Advanced Features

### Timestamping

```bash
# Only download if newer than local file
wget -N https://example.com/file.zip
wget --timestamping https://example.com/file.zip

# Check if file has been modified
wget --spider --server-response https://example.com/file.zip
```

### Spider Mode

```bash
# Check if file exists without downloading
wget --spider https://example.com/file.zip

# Check if URL is valid
if wget --spider https://example.com/file.zip 2>&1 | grep -q '200 OK'; then
    echo "URL is valid"
else
    echo "URL is invalid"
fi

# Get response headers only
wget --spider --server-response https://example.com/file.zip
```

### Quota and Limits

```bash
# Limit total download size
wget --quota=100M -i urls.txt

# Reject files larger than size
wget --reject-size=10M https://example.com/

# Accept files within size range
wget --accept-size=1M-10M https://example.com/
```

### Filtering

```bash
# Include only specific directories
wget -r -I /docs,/guides https://example.com

# Exclude specific directories
wget -r -X /private,/admin https://example.com

# Include only specific domains
wget -r -D example.com,cdn.example.com https://example.com

# Follow only relative links
wget -r --relative https://example.com
```

## Configuration File

### .wgetrc

```bash
# Create ~/.wgetrc
cat << 'EOF' > ~/.wgetrc
# Retry settings
tries = 10
retry_connrefused = on

# Timeout settings
timeout = 30
dns_timeout = 10
connect_timeout = 10
read_timeout = 30

# Wait between downloads
wait = 2
random_wait = on

# Download settings
continue = on
timestamping = on

# User agent
user_agent = Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36

# Proxy settings
# use_proxy = on
# http_proxy = http://proxy.example.com:8080
# https_proxy = http://proxy.example.com:8080

# Directories
dir_prefix = ~/Downloads/

# Output
verbose = off
quiet = off
EOF
```

## Common Use Cases

### Download Large Files

```bash
# Download with resume support
wget -c -t 0 --timeout=120 https://example.com/largefile.iso

# Download in background with logging
wget -b -c -o download.log https://example.com/largefile.iso

# Monitor background download
tail -f download.log
```

### Backup Website

```bash
#!/bin/bash
# backup-website.sh

SITE="https://example.com"
BACKUP_DIR="/backup/website"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR/$DATE"
cd "$BACKUP_DIR/$DATE"

wget --mirror \
     --convert-links \
     --adjust-extension \
     --page-requisites \
     --no-parent \
     --no-clobber \
     --wait=1 \
     --random-wait \
     "$SITE"

echo "Backup completed: $BACKUP_DIR/$DATE"
```

### Download All PDFs from Site

```bash
# Download all PDFs
wget -r -A pdf https://example.com

# Download PDFs from specific directory
wget -r -np -nd -A pdf https://example.com/documents/

# Download PDFs with original structure
wget -r -np -A pdf https://example.com/documents/
```

### API File Downloads

```bash
# Download with authentication token
wget --header="Authorization: Bearer $API_TOKEN" \
     https://api.example.com/files/report.pdf

# Download with API key
wget --header="X-API-Key: $API_KEY" \
     https://api.example.com/download/file.zip
```

### Batch Downloads

```bash
# Create URL list
for i in {1..100}; do
    echo "https://example.com/images/img${i}.jpg"
done > urls.txt

# Download with rate limiting
wget -i urls.txt --wait=1 --random-wait --limit-rate=500k

# Download with progress tracking
wget -i urls.txt -o download.log &
tail -f download.log | grep -E "saved|failed"
```

## Scripting Examples

### Download with Retry Logic

```bash
#!/bin/bash
URL="https://example.com/file.zip"
OUTPUT="file.zip"
MAX_ATTEMPTS=5

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "Attempt $i of $MAX_ATTEMPTS"

    if wget -c -O "$OUTPUT" "$URL"; then
        echo "Download successful"
        exit 0
    else
        echo "Download failed, retrying..."
        sleep 5
    fi
done

echo "Download failed after $MAX_ATTEMPTS attempts"
exit 1
```

### Parallel Downloads

```bash
#!/bin/bash
# Download multiple files in parallel

URLS=(
    "https://example.com/file1.zip"
    "https://example.com/file2.zip"
    "https://example.com/file3.zip"
)

for url in "${URLS[@]}"; do
    wget -c "$url" &
done

# Wait for all downloads to complete
wait

echo "All downloads completed"
```

### Monitor Website Changes

```bash
#!/bin/bash
# Check if website has been updated

URL="https://example.com/news.html"
OUTPUT="/tmp/news.html"

if [ -f "$OUTPUT" ]; then
    wget -N -o /tmp/wget.log "$URL"

    if grep -q "not retrieving" /tmp/wget.log; then
        echo "No changes detected"
    else
        echo "Website has been updated"
        # Send notification or perform action
    fi
else
    wget -O "$OUTPUT" "$URL"
    echo "Initial download completed"
fi
```

## Best Practices

1. **Always use resume support** for large files: `wget -c`
2. **Be respectful** with recursive downloads: use `--wait` and `--random-wait`
3. **Set appropriate timeout** values for unreliable connections
4. **Use timestamping** to avoid re-downloading unchanged files: `wget -N`
5. **Log downloads** for troubleshooting: `wget -o logfile`
6. **Limit bandwidth** if needed: `--limit-rate`
7. **Use .wgetrc** for common settings
8. **Check robots.txt**: `wget --execute robots=off` to override (use responsibly)

## Troubleshooting

### Common Issues

```bash
# SSL certificate verification failed
wget --no-check-certificate https://example.com/file.zip
# Better: Install proper CA certificates

# Connection timeout
wget --timeout=60 --tries=5 https://example.com/file.zip

# 403 Forbidden error
wget --user-agent="Mozilla/5.0" https://example.com/file.zip

# Cannot write to file (permission denied)
sudo wget -P /protected/directory https://example.com/file.zip

# Resume failed download
wget -c https://example.com/file.zip

# Check download status in background
tail -f wget-log

# Verify download integrity
wget https://example.com/file.zip
wget https://example.com/file.zip.sha256
sha256sum -c file.zip.sha256
```

### Debug Issues

```bash
# Enable debug output
wget -d https://example.com/file.zip 2>&1 | tee debug.log

# Check DNS resolution
wget --dns-timeout=10 https://example.com/file.zip

# Test connection only
wget --spider --server-response https://example.com/file.zip

# Show headers
wget -S https://example.com/file.zip
```

## Quick Reference

| Option | Description |
|--------|-------------|
| `-O file` | Save as file |
| `-P dir` | Save to directory |
| `-c` | Continue/resume download |
| `-b` | Background download |
| `-i file` | Download URLs from file |
| `-r` | Recursive download |
| `-l N` | Recursion depth |
| `-A list` | Accept file types |
| `-R list` | Reject file types |
| `-np` | No parent directory |
| `-m` | Mirror website |
| `-k` | Convert links |
| `-p` | Page requisites |
| `-q` | Quiet mode |
| `-v` | Verbose mode |
| `-N` | Timestamping |
| `--limit-rate=N` | Limit speed |
| `--tries=N` | Number of retries |
| `--timeout=N` | Timeout seconds |

wget is a versatile tool for reliable file downloads, website mirroring, and automated download tasks, essential for system administrators and developers.
