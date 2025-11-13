# Essential Linux Commands Reference

A comprehensive guide to essential Linux commands with examples, use cases, and practical tips.

## Table of Contents

1. [File System Navigation](#file-system-navigation)
2. [File Operations](#file-operations)
3. [Text Processing](#text-processing)
4. [Search and Find](#search-and-find)
5. [Process Management](#process-management)
6. [System Monitoring](#system-monitoring)
7. [User Management](#user-management)
8. [Permissions](#permissions)
9. [Package Management](#package-management)
10. [Network Commands](#network-commands)
11. [Service Management](#service-management)
12. [Compression](#compression)
13. [Disk Management](#disk-management)
14. [System Information](#system-information)

---

## File System Navigation

### ls - List Directory Contents

```bash
# Basic listing
ls                         # List files in current directory
ls -l                      # Long format with details
ls -a                      # Show hidden files
ls -lh                     # Human-readable sizes
ls -lah                    # Combine all above options
ls -R                      # Recursive listing
ls -lt                     # Sort by modification time
ls -lS                     # Sort by size

# Advanced usage
ls -i                      # Show inode numbers
ls -d */                   # List only directories
ls --color=auto            # Colored output
ls -ltr                    # Reverse time sort (oldest first)

# Examples
ls *.txt                   # List all .txt files
ls -l /var/log/            # List files in specific directory
ls -lh --sort=size         # Sort by size, human-readable
```

**Use Cases:**
- Quick directory overview
- Check file permissions and ownership
- Find recently modified files
- Disk usage analysis

### cd - Change Directory

```bash
cd /path/to/directory      # Absolute path
cd relative/path           # Relative path
cd ..                      # Parent directory
cd ../..                   # Two levels up
cd -                       # Previous directory
cd ~                       # Home directory
cd                         # Home directory (shorthand)
cd ~username               # Another user's home

# Examples
cd /var/log                # Go to log directory
cd ~/Documents             # Go to Documents in home
cd -                       # Toggle between two directories
```

### pwd - Print Working Directory

```bash
pwd                        # Show current directory
pwd -P                     # Show physical directory (resolve symlinks)
```

---

## File Operations

### cp - Copy Files

```bash
# Basic copying
cp source.txt dest.txt     # Copy file
cp file1 file2 dir/        # Copy multiple files to directory
cp -r dir1/ dir2/          # Copy directory recursively
cp -i file dest            # Interactive (prompt before overwrite)
cp -v file dest            # Verbose output
cp -u file dest            # Update (copy only if newer)

# Advanced options
cp -p file dest            # Preserve attributes (mode, ownership, timestamps)
cp -a dir1/ dir2/          # Archive mode (recursive + preserve)
cp --backup file dest      # Create backup before overwriting

# Examples
cp /etc/config ~/.config/  # Copy config file to home
cp -r /var/www/* /backup/  # Backup web directory
cp -av src/ dest/          # Full directory copy with attributes
```

### mv - Move/Rename Files

```bash
# Basic move/rename
mv old.txt new.txt         # Rename file
mv file.txt dir/           # Move file to directory
mv file1 file2 dir/        # Move multiple files
mv -i file dest            # Interactive mode
mv -v file dest            # Verbose output

# Examples
mv *.log /var/log/         # Move all log files
mv -n file dest            # No overwrite
mv --backup=numbered f d   # Numbered backups
```

### rm - Remove Files

```bash
# Basic removal
rm file.txt                # Remove file
rm -r directory/           # Remove directory recursively
rm -f file                 # Force removal (no confirmation)
rm -rf directory/          # Force remove directory
rm -i file                 # Interactive (ask before removal)
rm -v file                 # Verbose output

# Safe practices
rm -I files*               # Prompt once before removing many files
rm -d emptydir/            # Remove empty directory only

# Examples
rm *.tmp                   # Remove all .tmp files
rm -rf /tmp/session*       # Force remove temp sessions
find . -name "*.bak" -delete  # Alternative: safer removal
```

**Warning:** Use `rm -rf` with extreme caution!

### mkdir - Make Directories

```bash
mkdir newdir               # Create directory
mkdir -p path/to/dir       # Create parent directories
mkdir -m 755 dir           # Set permissions
mkdir -v dir               # Verbose output

# Examples
mkdir -p project/{src,bin,doc}  # Create multiple directories
mkdir -p ~/backup/$(date +%Y-%m-%d)  # Date-based backup dir
```

### touch - Create/Update Files

```bash
touch file.txt             # Create empty file or update timestamp
touch -c file              # No create (only update if exists)
touch -t 202301011200 file # Set specific timestamp
touch -d "2023-01-01" file # Set date

# Examples
touch {1..10}.txt          # Create multiple files
touch -r ref.txt new.txt   # Copy timestamp from reference
```

---

## Text Processing

### cat - Concatenate and Display

```bash
cat file.txt               # Display file contents
cat file1 file2            # Concatenate multiple files
cat > file.txt             # Create file from stdin (Ctrl+D to end)
cat >> file.txt            # Append to file
cat -n file.txt            # Number all lines
cat -b file.txt            # Number non-blank lines
cat -s file.txt            # Squeeze multiple blank lines

# Examples
cat /etc/passwd            # View user accounts
cat file1 file2 > combined # Combine files
cat /dev/null > file.txt   # Empty a file
```

### grep - Search Text Patterns

```bash
# Basic search
grep "pattern" file.txt    # Search for pattern
grep -i "pattern" file     # Case-insensitive
grep -v "pattern" file     # Invert match (exclude)
grep -r "pattern" dir/     # Recursive search
grep -n "pattern" file     # Show line numbers
grep -c "pattern" file     # Count matches

# Advanced options
grep -w "word" file        # Match whole words only
grep -A 3 "pattern" file   # Show 3 lines after match
grep -B 3 "pattern" file   # Show 3 lines before match
grep -C 3 "pattern" file   # Show 3 lines context
grep -l "pattern" files*   # List filenames only
grep -E "regex" file       # Extended regex (or egrep)

# Regular expressions
grep "^start" file         # Lines starting with "start"
grep "end$" file           # Lines ending with "end"
grep "^$" file             # Empty lines
grep "[0-9]\{3\}" file     # Three consecutive digits

# Examples
grep -r "TODO" ~/code/     # Find all TODOs in code
grep -i "error" /var/log/*.log  # Find errors in logs
ps aux | grep nginx        # Find nginx processes
grep -v "^#" config.txt    # Show non-comment lines
netstat -tulpn | grep :80  # Find what's using port 80
```

**Use Cases:**
- Log file analysis
- Finding specific code patterns
- Filtering command output
- Configuration file parsing

### sed - Stream Editor

```bash
# Basic substitution
sed 's/old/new/' file      # Replace first occurrence per line
sed 's/old/new/g' file     # Replace all occurrences
sed 's/old/new/gi' file    # Case-insensitive global replace
sed -i 's/old/new/g' file  # In-place editing
sed -i.bak 's/old/new/g' file  # In-place with backup

# Line operations
sed -n '5p' file           # Print line 5
sed -n '1,5p' file         # Print lines 1-5
sed '5d' file              # Delete line 5
sed '/pattern/d' file      # Delete lines matching pattern
sed '1,3d' file            # Delete lines 1-3

# Advanced usage
sed '/pattern/s/old/new/' file  # Replace only in matching lines
sed 's/^/  /' file         # Add 2 spaces at start of each line
sed 's/$/\r/' file         # Convert to DOS line endings
sed '/^$/d' file           # Remove empty lines

# Examples
sed 's/localhost/127.0.0.1/g' config  # Replace hostname
sed -n '/ERROR/,/END/p' log    # Print between patterns
sed '/#/d' file                # Remove comment lines
sed 's/\t/ /g' file            # Replace tabs with spaces
```

### awk - Text Processing Language

```bash
# Basic usage
awk '{print}' file         # Print all lines
awk '{print $1}' file      # Print first column
awk '{print $1,$3}' file   # Print columns 1 and 3
awk '{print $NF}' file     # Print last column
awk '{print NR,$0}' file   # Print line numbers

# Field separator
awk -F: '{print $1}' /etc/passwd  # Custom delimiter
awk -F',' '{print $2}' data.csv   # CSV parsing

# Patterns and conditions
awk '/pattern/' file       # Print lines matching pattern
awk '$3 > 100' file        # Print if column 3 > 100
awk 'NR==5' file           # Print line 5
awk 'NR>=5 && NR<=10' file # Print lines 5-10
awk 'length($0) > 80' file # Print lines longer than 80 chars

# Calculations
awk '{sum+=$1} END {print sum}' file  # Sum first column
awk '{print $1*$2}' file   # Multiply columns 1 and 2

# Examples
awk -F: '{print $1}' /etc/passwd  # List usernames
ps aux | awk '{print $2,$11}'  # Print PID and command
df -h | awk '$5+0 > 80 {print $0}'  # Disk usage > 80%
netstat -an | awk '/ESTABLISHED/ {print $5}'  # Connected IPs
awk '{sum+=$1} END {print sum/NR}' data  # Average of column 1
```

**Use Cases:**
- Log parsing and analysis
- Data extraction from structured text
- Quick calculations on columns
- Report generation

### head - Display Beginning of File

```bash
head file.txt              # First 10 lines
head -n 20 file.txt        # First 20 lines
head -c 100 file.txt       # First 100 bytes
head -n -5 file.txt        # All but last 5 lines

# Examples
head -n 1 *.txt            # First line of each file
head /var/log/syslog       # Quick log preview
```

### tail - Display End of File

```bash
tail file.txt              # Last 10 lines
tail -n 20 file.txt        # Last 20 lines
tail -f file.txt           # Follow file (live updates)
tail -F file.txt           # Follow with retry (if rotated)
tail -n +5 file.txt        # From line 5 to end

# Examples
tail -f /var/log/syslog    # Monitor system log
tail -n 100 -f app.log     # Follow last 100 lines
tail -f log | grep ERROR   # Filter live log stream
```

### sort - Sort Lines

```bash
sort file.txt              # Alphabetical sort
sort -r file.txt           # Reverse sort
sort -n file.txt           # Numeric sort
sort -u file.txt           # Unique lines only
sort -k 2 file.txt         # Sort by column 2
sort -t: -k3 -n /etc/passwd  # Numeric sort by field 3

# Examples
sort -t',' -k2 -n data.csv # Sort CSV by second column
ls -l | sort -k 5 -n       # Sort files by size
history | sort | uniq -c   # Find most used commands
```

### uniq - Report Unique Lines

```bash
uniq file.txt              # Remove adjacent duplicates
uniq -c file.txt           # Count occurrences
uniq -d file.txt           # Show only duplicates
uniq -u file.txt           # Show only unique lines
uniq -i file.txt           # Case-insensitive

# Examples (usually with sort)
sort file.txt | uniq       # Remove all duplicates
sort file.txt | uniq -c | sort -rn  # Frequency count
```

---

## Search and Find

### find - Search for Files

```bash
# By name
find . -name "file.txt"    # Find by exact name
find . -iname "*.txt"      # Case-insensitive name
find /var -name "*.log"    # Find in specific directory

# By type
find . -type f             # Find files
find . -type d             # Find directories
find . -type l             # Find symbolic links

# By size
find . -size +100M         # Files larger than 100MB
find . -size -1k           # Files smaller than 1KB
find . -empty              # Empty files/directories

# By time
find . -mtime -7           # Modified in last 7 days
find . -atime +30          # Accessed more than 30 days ago
find . -ctime -1           # Changed in last 24 hours
find . -mmin -60           # Modified in last 60 minutes

# By permissions
find . -perm 777           # Exactly 777 permissions
find . -perm -644          # At least 644 permissions
find . -user root          # Owned by root
find . -group www-data     # Owned by www-data group

# Actions
find . -name "*.tmp" -delete  # Delete found files
find . -name "*.sh" -exec chmod +x {} \;  # Execute command
find . -type f -exec wc -l {} +  # Count lines

# Examples
find /home -user john -name "*.pdf"  # User's PDF files
find . -name "*.log" -mtime +30 -delete  # Delete old logs
find /var/www -type f -perm 777  # Find world-writable files
find . -size +50M -size -100M    # Files between 50-100MB
find . -name "*.js" -exec grep -l "TODO" {} \;  # Find TODOs
```

### locate - Quick File Search

```bash
locate filename            # Quick search in database
locate -i filename         # Case-insensitive
locate -c pattern          # Count matches
locate -b '\filename'      # Exact basename match

# Update database
sudo updatedb              # Refresh locate database

# Examples
locate nginx.conf          # Find nginx config
locate -r '\.conf$'        # All .conf files
```

### which - Locate Command

```bash
which python               # Find command path
which -a python            # Show all matches

# Examples
which docker               # Find Docker binary
type python                # Alternative (bash builtin)
```

### whereis - Locate Binary/Source/Manual

```bash
whereis ls                 # Find binary, source, man page
whereis -b ls              # Binary only
whereis -m ls              # Manual only
whereis -s ls              # Source only
```

---

## Process Management

### ps - Process Status

```bash
# Basic usage
ps                         # Current shell processes
ps aux                     # All processes (BSD style)
ps -ef                     # All processes (System V style)
ps -u username             # User's processes
ps -p 1234                 # Specific process by PID

# Detailed view
ps aux | grep nginx        # Find specific process
ps auxf                    # Process tree (forest)
ps -eo pid,user,%cpu,%mem,cmd  # Custom columns
ps --sort=-%mem            # Sort by memory usage
ps -C nginx                # Processes by command name

# Examples
ps aux | head              # Top processes
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head  # CPU hogs
ps -U www-data             # Web server processes
```

### top - Interactive Process Viewer

```bash
top                        # Launch interactive viewer
top -u username            # Show user's processes
top -p 1234                # Monitor specific PID
top -b -n 1                # Batch mode (one iteration)
top -d 5                   # Update every 5 seconds

# Interactive commands (while running)
# k - kill process
# r - renice (change priority)
# M - sort by memory
# P - sort by CPU
# q - quit
# h - help

# Examples
top -o %MEM                # Sort by memory (macOS)
top | head -20             # First 20 lines
```

### htop - Enhanced Process Viewer

```bash
htop                       # Launch htop (if installed)
htop -u username           # Show user's processes
htop -p PID,PID            # Monitor specific PIDs

# Interactive features
# F9 - kill process
# F7/F8 - adjust priority
# F5 - tree view
# F3 - search
# F4 - filter
```

### kill - Terminate Process

```bash
# By PID
kill 1234                  # Graceful termination (SIGTERM)
kill -9 1234               # Force kill (SIGKILL)
kill -15 1234              # Explicit SIGTERM
kill -HUP 1234             # Hangup signal (reload config)

# Signal list
kill -l                    # List all signals

# Examples
kill $(pidof firefox)      # Kill by process name
killall nginx              # Kill all nginx processes
pkill -u username          # Kill user's processes
```

### pkill/killall - Kill by Name

```bash
pkill firefox              # Kill by process name
pkill -u username          # Kill user's processes
pkill -9 python            # Force kill all python processes
pkill -f "script.py"       # Kill by full command line

killall nginx              # Kill all nginx processes
killall -u username bash   # Kill user's bash sessions
```

### jobs/bg/fg - Job Control

```bash
# Job control
command &                  # Run in background
jobs                       # List background jobs
fg %1                      # Bring job 1 to foreground
bg %1                      # Resume job 1 in background
Ctrl+Z                     # Suspend current job
disown %1                  # Detach job from shell

# Examples
find / -name "*.log" > /tmp/logs.txt &  # Background search
sleep 100 &                # Background sleep
jobs -l                    # List with PIDs
```

### nohup - Run Immune to Hangups

```bash
nohup command &            # Run detached from terminal
nohup ./script.sh &        # Script continues after logout
nohup command > output.log 2>&1 &  # Redirect output

# Examples
nohup python server.py > server.log 2>&1 &
nohup long_running_task.sh &
```

### systemctl - Service Management

```bash
# Service operations
systemctl start nginx      # Start service
systemctl stop nginx       # Stop service
systemctl restart nginx    # Restart service
systemctl reload nginx     # Reload configuration
systemctl status nginx     # Service status
systemctl enable nginx     # Enable at boot
systemctl disable nginx    # Disable at boot

# System operations
systemctl reboot           # Reboot system
systemctl poweroff         # Shutdown system
systemctl suspend          # Suspend system

# Information
systemctl list-units       # List active units
systemctl list-unit-files  # List all unit files
systemctl --failed         # Show failed services
systemctl is-enabled nginx # Check if enabled
systemctl is-active nginx  # Check if running

# Examples
systemctl status sshd      # Check SSH status
systemctl restart apache2  # Restart web server
systemctl list-dependencies nginx  # Show dependencies
```

---

## System Monitoring

### df - Disk Free Space

```bash
df                         # Show disk usage
df -h                      # Human-readable sizes
df -i                      # Inode usage
df -T                      # Show filesystem type
df /home                   # Specific mount point

# Examples
df -h | grep -v tmpfs      # Exclude temporary filesystems
df -h --total              # Show total summary
```

### du - Disk Usage

```bash
du                         # Directory space usage
du -h                      # Human-readable
du -sh *                   # Summary for each item
du -sh directory           # Total for directory
du -ah                     # All files (not just directories)
du --max-depth=1           # Limit directory depth

# Examples
du -sh /var/log            # Log directory size
du -h | sort -rh | head -10  # Top 10 largest directories
du -ch *.log | tail -1     # Total size of log files
```

### free - Memory Usage

```bash
free                       # Show memory usage
free -h                    # Human-readable
free -m                    # In megabytes
free -g                    # In gigabytes
free -s 5                  # Update every 5 seconds

# Examples
free -h                    # Quick memory check
watch -n 1 free -h         # Monitor continuously
```

### vmstat - Virtual Memory Statistics

```bash
vmstat                     # Memory, process, paging stats
vmstat 1                   # Update every second
vmstat 1 10                # 10 samples, 1 second apart
vmstat -s                  # Memory statistics
vmstat -d                  # Disk statistics

# Examples
vmstat 5                   # Monitor system stats
```

### iostat - I/O Statistics

```bash
iostat                     # CPU and disk I/O stats
iostat -x                  # Extended statistics
iostat -d 1                # Disk stats every second
iostat -p sda              # Specific disk

# Examples
iostat -xz 1               # Extended, skip zero-activity
```

### netstat - Network Statistics

```bash
netstat -tulpn             # Listening ports with programs
netstat -an                # All connections, numeric
netstat -r                 # Routing table
netstat -i                 # Network interfaces
netstat -s                 # Protocol statistics

# Examples
netstat -tulpn | grep :80  # Check port 80
netstat -ant | grep ESTABLISHED  # Active connections
```

### ss - Socket Statistics (newer netstat)

```bash
ss -tulpn                  # Listening TCP/UDP ports
ss -ta                     # All TCP sockets
ss -ua                     # All UDP sockets
ss -s                      # Summary statistics
ss dst :80                 # Connections to port 80

# Examples
ss -t state established    # Established TCP connections
ss -o state established    # With timer info
ss -p | grep ssh           # SSH connections
```

### lsof - List Open Files

```bash
lsof                       # All open files
lsof -u username           # User's open files
lsof -i :80                # Processes using port 80
lsof -i TCP:1-1024         # Processes on ports 1-1024
lsof /path/to/file         # What's accessing a file
lsof -c nginx              # Files opened by nginx
lsof -p 1234               # Files opened by PID

# Examples
lsof -i -P -n              # Network connections (no DNS)
lsof +D /var/log           # Everything under directory
lsof -t -i :8080           # PIDs using port 8080
```

---

## User Management

### useradd - Create User

```bash
useradd username           # Create user
useradd -m username        # Create with home directory
useradd -m -s /bin/bash username  # Specify shell
useradd -m -G group1,group2 user  # Add to groups
useradd -m -e 2024-12-31 user     # With expiry date

# Examples
useradd -m -s /bin/bash john
useradd -m -G sudo,docker admin
```

### usermod - Modify User

```bash
usermod -aG sudo username  # Add to sudo group
usermod -s /bin/zsh user   # Change shell
usermod -L username        # Lock account
usermod -U username        # Unlock account
usermod -e 2024-12-31 user # Set expiry date

# Examples
usermod -aG docker username  # Add to docker group
usermod -d /new/home -m user # Change home directory
```

### userdel - Delete User

```bash
userdel username           # Delete user
userdel -r username        # Delete user and home directory
```

### passwd - Change Password

```bash
passwd                     # Change your password
passwd username            # Change user's password (as root)
passwd -l username         # Lock password
passwd -u username         # Unlock password
passwd -e username         # Expire password (force change)

# Examples
passwd john                # Set password for john
passwd -S john             # Show password status
```

### su - Switch User

```bash
su                         # Switch to root
su username                # Switch to user
su - username              # Switch with environment
su -c "command" username   # Run command as user

# Examples
su - postgres              # Switch to postgres user
su -c "systemctl restart nginx" root
```

### sudo - Execute as Superuser

```bash
sudo command               # Run command as root
sudo -u user command       # Run as specific user
sudo -i                    # Interactive root shell
sudo -s                    # Shell as root
sudo -l                    # List allowed commands
sudo -k                    # Invalidate cached credentials

# Examples
sudo apt update            # Update package lists
sudo -u www-data touch /var/www/file
sudo !!                    # Run last command with sudo
```

---

## Permissions

### chmod - Change File Mode

```bash
# Numeric mode
chmod 644 file             # rw-r--r--
chmod 755 file             # rwxr-xr-x
chmod 777 file             # rwxrwxrwx
chmod 600 file             # rw-------

# Symbolic mode
chmod u+x file             # Add execute for user
chmod g-w file             # Remove write for group
chmod o=r file             # Set others to read only
chmod a+r file             # Add read for all
chmod u+x,g+x file         # Multiple changes

# Recursive
chmod -R 755 directory     # Apply recursively

# Examples
chmod +x script.sh         # Make executable
chmod -R 755 /var/www      # Web directory permissions
chmod u+s file             # Set SUID bit
chmod g+s directory        # Set SGID bit
chmod +t directory         # Set sticky bit
```

**Permission numbers:**
- 4 = read (r)
- 2 = write (w)
- 1 = execute (x)
- Sum for each user/group/others

### chown - Change Ownership

```bash
chown user file            # Change owner
chown user:group file      # Change owner and group
chown -R user:group dir    # Recursive change
chown --reference=ref file # Copy ownership from reference

# Examples
chown www-data:www-data /var/www/html
chown -R mysql:mysql /var/lib/mysql
chown john:developers project/
```

### chgrp - Change Group

```bash
chgrp group file           # Change group
chgrp -R group directory   # Recursive change

# Examples
chgrp www-data website/
chgrp -R developers /opt/project
```

### umask - Default Permissions

```bash
umask                      # Show current umask
umask 022                  # Set umask (755 for dirs, 644 for files)
umask 002                  # Set umask (775 for dirs, 664 for files)

# Examples
umask 077                  # Private by default (700/600)
```

---

## Package Management

### APT (Debian/Ubuntu)

```bash
# Update
apt update                 # Update package lists
apt upgrade                # Upgrade packages
apt full-upgrade           # Upgrade + handle dependencies
apt dist-upgrade           # Distribution upgrade

# Install/Remove
apt install package        # Install package
apt install package1 package2  # Multiple packages
apt remove package         # Remove package
apt purge package          # Remove package and config
apt autoremove             # Remove unused dependencies

# Search and Info
apt search keyword         # Search packages
apt show package           # Package information
apt list --installed       # List installed packages
apt list --upgradable      # List upgradable packages

# Examples
apt install nginx          # Install web server
apt remove --purge apache2 # Complete removal
apt install build-essential git curl
```

### DNF/YUM (RHEL/Fedora/CentOS)

```bash
# Update
dnf update                 # Update packages
dnf upgrade                # Synonym for update

# Install/Remove
dnf install package        # Install package
dnf remove package         # Remove package
dnf autoremove             # Remove orphaned dependencies

# Search and Info
dnf search keyword         # Search packages
dnf info package           # Package information
dnf list installed         # List installed packages

# Examples
dnf install httpd          # Install Apache
dnf groupinstall "Development Tools"
```

### Snap (Universal)

```bash
snap install package       # Install snap package
snap remove package        # Remove package
snap refresh               # Update all snaps
snap list                  # List installed snaps
snap find keyword          # Search snaps

# Examples
snap install docker
snap install --classic code  # Classic confinement
```

---

## Network Commands

### ip - Network Configuration

```bash
# Address management
ip addr show               # Show all IP addresses
ip addr show eth0          # Show specific interface
ip addr add IP/MASK dev eth0  # Add IP address
ip addr del IP/MASK dev eth0  # Delete IP address

# Link management
ip link show               # Show network interfaces
ip link set eth0 up        # Bring interface up
ip link set eth0 down      # Bring interface down

# Route management
ip route show              # Show routing table
ip route add default via GATEWAY  # Add default route
ip route del default       # Delete default route

# Neighbor (ARP)
ip neigh show              # Show ARP table

# Examples
ip addr show               # Quick network overview
ip route get 8.8.8.8       # Show route to destination
ip link set eth0 mtu 9000  # Set MTU
```

### ping - Test Connectivity

```bash
ping host                  # Ping host
ping -c 4 host             # Send 4 packets
ping -i 2 host             # 2 second interval
ping -s 1000 host          # 1000 byte packets
ping -W 1 host             # 1 second timeout

# Examples
ping -c 4 google.com       # Test internet connectivity
ping 192.168.1.1           # Test local gateway
```

### curl - Transfer Data

```bash
# Basic requests
curl URL                   # GET request
curl -O URL                # Download file (keep name)
curl -o file.txt URL       # Download with custom name
curl -I URL                # Headers only
curl -L URL                # Follow redirects

# HTTP methods
curl -X POST URL           # POST request
curl -X PUT URL            # PUT request
curl -X DELETE URL         # DELETE request

# Data and headers
curl -d "param=value" URL  # POST data
curl -H "Header: Value" URL  # Custom header
curl -u user:pass URL      # Basic authentication
curl -b cookies.txt URL    # Send cookies
curl -c cookies.txt URL    # Save cookies

# Examples
curl -I https://google.com  # Check headers
curl -o page.html https://example.com
curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' API_URL
curl -u admin:password http://api.example.com
```

### wget - Download Files

```bash
wget URL                   # Download file
wget -O filename URL       # Save with custom name
wget -c URL                # Continue interrupted download
wget -b URL                # Background download
wget -r URL                # Recursive download
wget --limit-rate=200k URL # Limit download speed
wget -i urls.txt           # Download multiple URLs

# Examples
wget https://example.com/file.iso
wget -c https://mirrors.kernel.org/ubuntu/ubuntu-22.04.iso
wget -r -np -k http://example.com  # Mirror website
```

### ssh - Secure Shell

```bash
ssh user@host              # Connect to host
ssh -p 2222 user@host      # Custom port
ssh -i key.pem user@host   # Use specific key
ssh user@host command      # Run remote command
ssh -L 8080:localhost:80 user@host  # Local port forwarding
ssh -R 8080:localhost:80 user@host  # Remote port forwarding

# Examples
ssh john@192.168.1.100
ssh -i ~/.ssh/aws-key.pem ubuntu@ec2-instance
ssh user@host 'df -h'      # Check remote disk space
```

### scp - Secure Copy

```bash
scp file user@host:/path   # Copy to remote
scp user@host:/path/file . # Copy from remote
scp -r directory user@host:/path  # Copy directory
scp -P 2222 file user@host:/path  # Custom port
scp -i key.pem file user@host:/path  # Specific key

# Examples
scp backup.tar.gz user@backup-server:/backups/
scp -r website/ user@server:/var/www/
scp user@server:/var/log/app.log ./logs/
```

### rsync - Sync Files

```bash
rsync -av source/ dest/    # Archive and verbose
rsync -avz source/ user@host:dest/  # With compression
rsync -av --delete src/ dst/  # Delete in destination
rsync -av --progress src/ dst/  # Show progress
rsync -av --exclude="*.log" src/ dst/  # Exclude pattern

# Examples
rsync -avz ~/project/ backup-server:/backups/project/
rsync -av --delete /var/www/ /backup/www/
rsync -avz -e "ssh -p 2222" src/ user@host:dest/
```

---

## Service Management

### journalctl - Query Systemd Journal

```bash
journalctl                 # Show all logs
journalctl -f              # Follow logs (tail -f)
journalctl -u nginx        # Service logs
journalctl -u nginx -f     # Follow service logs
journalctl --since today   # Today's logs
journalctl --since "1 hour ago"  # Last hour
journalctl -p err          # Error priority and above
journalctl -k              # Kernel messages
journalctl -b              # Current boot logs
journalctl --disk-usage    # Disk usage by logs

# Examples
journalctl -u sshd -n 100  # Last 100 SSH log entries
journalctl --since "2024-01-01" --until "2024-01-31"
journalctl -u nginx --since yesterday
```

---

## Compression

### tar - Archive Files

```bash
# Create archives
tar -cvf archive.tar files # Create tar archive
tar -czvf archive.tar.gz files  # Create gzipped archive
tar -cjvf archive.tar.bz2 files # Create bzip2 archive
tar -cJvf archive.tar.xz files  # Create xz archive

# Extract archives
tar -xvf archive.tar       # Extract tar
tar -xzvf archive.tar.gz   # Extract gzipped
tar -xjvf archive.tar.bz2  # Extract bzip2
tar -xJvf archive.tar.xz   # Extract xz
tar -xzvf archive.tar.gz -C /dest  # Extract to directory

# List contents
tar -tvf archive.tar       # List contents
tar -tzvf archive.tar.gz   # List gzipped archive

# Examples
tar -czvf backup-$(date +%Y%m%d).tar.gz /home/user/
tar -xzvf website.tar.gz -C /var/www/
tar -czvf project.tar.gz --exclude='*.log' project/
```

### gzip/gunzip - Compress Files

```bash
gzip file.txt              # Compress (creates file.txt.gz)
gzip -k file.txt           # Keep original
gzip -9 file.txt           # Maximum compression
gunzip file.txt.gz         # Decompress
gzip -l file.txt.gz        # List compression info

# Examples
gzip -r directory/         # Compress all files in directory
gzip -c file.txt > file.txt.gz  # Keep original
```

### zip/unzip - Zip Archives

```bash
zip archive.zip files      # Create zip
zip -r archive.zip dir/    # Recursive zip
unzip archive.zip          # Extract zip
unzip -l archive.zip       # List contents
unzip archive.zip -d /dest # Extract to directory

# Examples
zip -r backup.zip /home/user/Documents
unzip file.zip
zip -e secure.zip file     # Password protected
```

---

## Disk Management

### fdisk - Partition Disk

```bash
fdisk -l                   # List all disks and partitions
fdisk /dev/sda             # Open disk for partitioning

# Interactive commands (in fdisk):
# n - new partition
# d - delete partition
# p - print partition table
# w - write changes
# q - quit without saving
```

### mount/umount - Mount Filesystems

```bash
mount                      # Show mounted filesystems
mount /dev/sda1 /mnt       # Mount partition
mount -t nfs server:/share /mnt  # Mount NFS
mount -o loop disk.iso /mnt # Mount ISO
umount /mnt                # Unmount
umount -l /mnt             # Lazy unmount

# Examples
mount /dev/sdb1 /media/usb
mount -t cifs //server/share /mnt -o username=user
mount --bind /source /dest # Bind mount
```

### mkfs - Make Filesystem

```bash
mkfs.ext4 /dev/sda1        # Create ext4 filesystem
mkfs.xfs /dev/sda1         # Create XFS filesystem
mkfs.vfat /dev/sda1        # Create FAT filesystem

# Examples
mkfs.ext4 -L MyDisk /dev/sdb1  # With label
mkfs.ext4 -m 1 /dev/sdb1   # Reserve 1% for root
```

---

## System Information

### uname - System Information

```bash
uname -a                   # All information
uname -r                   # Kernel release
uname -m                   # Machine hardware
uname -o                   # Operating system
```

### lscpu - CPU Information

```bash
lscpu                      # Detailed CPU info
lscpu | grep "CPU(s)"      # Number of CPUs
```

### lsblk - Block Devices

```bash
lsblk                      # List block devices
lsblk -f                   # Show filesystems
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT  # Custom columns
```

### lspci - PCI Devices

```bash
lspci                      # List PCI devices
lspci -v                   # Verbose output
lspci | grep VGA           # Graphics card info
```

### lsusb - USB Devices

```bash
lsusb                      # List USB devices
lsusb -v                   # Verbose output
```

### hostname - System Hostname

```bash
hostname                   # Show hostname
hostname -I                # Show IP addresses
hostnamectl                # Detailed host information
hostnamectl set-hostname new-name  # Change hostname
```

### date - Date and Time

```bash
date                       # Current date and time
date +%Y-%m-%d             # Custom format (2024-01-15)
date +%s                   # Unix timestamp
date -d "yesterday"        # Yesterday's date
date -d "next Monday"      # Next Monday

# Examples
date +%Y%m%d-%H%M%S        # 20240115-143025
date -d @1704067200        # Convert timestamp
```

### uptime - System Uptime

```bash
uptime                     # How long system is running
uptime -p                  # Pretty format
uptime -s                  # Since when
```

---

## Practical Tips and Best Practices

### Command Chaining

```bash
# Sequential execution
command1 ; command2        # Run both regardless
command1 && command2       # Run command2 if command1 succeeds
command1 || command2       # Run command2 if command1 fails

# Examples
apt update && apt upgrade  # Update then upgrade
make || echo "Build failed"
cd /tmp && rm -rf old_files
```

### Redirection and Pipes

```bash
# Output redirection
command > file             # Overwrite file
command >> file            # Append to file
command 2> file            # Redirect stderr
command > file 2>&1        # Redirect both stdout and stderr
command &> file            # Redirect both (shorthand)

# Input redirection
command < file             # Read from file
command << EOF             # Here document
multiline input
