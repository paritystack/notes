# Ansible

Ansible is an open-source IT automation engine that automates provisioning, configuration management, application deployment, orchestration, and many other IT processes. It uses SSH for communication and requires no agents on managed nodes.

## Overview

Ansible uses a simple, human-readable language (YAML) to describe automation jobs. It's agentless, using OpenSSH for transport, making it secure and easy to set up.

**Key Concepts:**
- **Inventory**: List of managed nodes (hosts)
- **Playbook**: YAML files defining tasks to execute
- **Module**: Reusable code units for specific tasks
- **Role**: Organized collection of playbooks and files
- **Task**: Single action to be performed
- **Handler**: Tasks triggered by notifications
- **Facts**: System information gathered from hosts

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ansible

# macOS
brew install ansible

# CentOS/RHEL
sudo yum install epel-release
sudo yum install ansible

# Using pip
pip install ansible

# Verify installation
ansible --version
```

## Basic Configuration

### Ansible Config

```bash
# Create ansible.cfg
cat << 'EOF' > ansible.cfg
[defaults]
inventory = ./inventory
host_key_checking = False
remote_user = ansible
private_key_file = ~/.ssh/id_rsa
retry_files_enabled = False
gathering = smart
fact_caching = jsonfile
fact_caching_connection = /tmp/ansible_facts
fact_caching_timeout = 3600

[privilege_escalation]
become = True
become_method = sudo
become_user = root
become_ask_pass = False
EOF
```

### Inventory File

```ini
# inventory or hosts file

# Single host
web1.example.com

# Group of hosts
[webservers]
web1.example.com
web2.example.com
192.168.1.10

# Multiple groups
[databases]
db1.example.com
db2.example.com

[app:children]
webservers
databases

# Host with variables
[webservers]
web1.example.com ansible_user=admin ansible_port=2222

# Group variables
[webservers:vars]
ansible_user=deploy
ansible_python_interpreter=/usr/bin/python3
http_port=80
```

### Dynamic Inventory (YAML)

```yaml
# inventory.yml
all:
  hosts:
    web1.example.com:
    web2.example.com:
  children:
    webservers:
      hosts:
        web1.example.com:
          ansible_user: deploy
        web2.example.com:
          ansible_user: deploy
      vars:
        http_port: 80
    databases:
      hosts:
        db1.example.com:
        db2.example.com:
      vars:
        db_port: 5432
```

## Ad-hoc Commands

```bash
# Ping all hosts
ansible all -m ping

# Ping specific group
ansible webservers -m ping

# Run shell command
ansible all -m shell -a "uptime"
ansible webservers -a "df -h"  # shell module is default

# Copy file
ansible all -m copy -a "src=/local/file dest=/remote/file"

# Install package
ansible webservers -m apt -a "name=nginx state=present" --become

# Start service
ansible webservers -m service -a "name=nginx state=started" --become

# Gather facts
ansible all -m setup

# Specific fact
ansible all -m setup -a "filter=ansible_distribution*"

# Execute with sudo
ansible all -a "systemctl restart nginx" --become

# Execute as specific user
ansible all -a "whoami" --become-user=www-data
```

## Playbooks

### Basic Playbook

```yaml
# playbook.yml
---
- name: Configure web servers
  hosts: webservers
  become: yes

  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes

    - name: Start nginx service
      service:
        name: nginx
        state: started
        enabled: yes

    - name: Copy index.html
      copy:
        src: files/index.html
        dest: /var/www/html/index.html
        owner: www-data
        group: www-data
        mode: '0644'
```

### Running Playbooks

```bash
# Run playbook
ansible-playbook playbook.yml

# Dry run (check mode)
ansible-playbook playbook.yml --check

# Show differences
ansible-playbook playbook.yml --check --diff

# Limit to specific hosts
ansible-playbook playbook.yml --limit web1.example.com
ansible-playbook playbook.yml --limit webservers

# Tags
ansible-playbook playbook.yml --tags "install"
ansible-playbook playbook.yml --skip-tags "config"

# Start at specific task
ansible-playbook playbook.yml --start-at-task="Install nginx"

# Verbose output
ansible-playbook playbook.yml -v    # verbose
ansible-playbook playbook.yml -vv   # more verbose
ansible-playbook playbook.yml -vvv  # very verbose
```

### Variables in Playbooks

```yaml
---
- name: Configure application
  hosts: webservers
  vars:
    app_name: myapp
    app_version: "1.0"
    app_port: 8080

  tasks:
    - name: Create app directory
      file:
        path: "/opt/{{ app_name }}"
        state: directory
        owner: "{{ ansible_user }}"

    - name: Display variables
      debug:
        msg: "Deploying {{ app_name }} version {{ app_version }} on port {{ app_port }}"
```

### Variables from Files

```yaml
# vars.yml
---
app_name: myapp
app_version: "1.0"
app_port: 8080
database:
  host: db.example.com
  name: myapp_db
  user: myapp_user

# playbook.yml
---
- name: Configure application
  hosts: webservers
  vars_files:
    - vars.yml

  tasks:
    - name: Display app info
      debug:
        msg: "App: {{ app_name }}, DB: {{ database.host }}"
```

## Common Modules

### System Modules

```yaml
# User management
- name: Create user
  user:
    name: deploy
    state: present
    groups: sudo
    shell: /bin/bash
    create_home: yes

# Group management
- name: Create group
  group:
    name: developers
    state: present

# File operations
- name: Create file
  file:
    path: /tmp/test.txt
    state: touch
    mode: '0644'
    owner: deploy

- name: Create directory
  file:
    path: /opt/myapp
    state: directory
    mode: '0755'
    recurse: yes

# Copy files
- name: Copy file
  copy:
    src: files/config.conf
    dest: /etc/myapp/config.conf
    backup: yes

# Template files
- name: Deploy template
  template:
    src: templates/nginx.conf.j2
    dest: /etc/nginx/nginx.conf
    validate: 'nginx -t -c %s'
  notify: restart nginx
```

### Package Management

```yaml
# APT (Debian/Ubuntu)
- name: Install packages
  apt:
    name:
      - nginx
      - postgresql
      - python3-pip
    state: present
    update_cache: yes

# YUM/DNF (RedHat/CentOS)
- name: Install packages
  yum:
    name:
      - httpd
      - mariadb-server
    state: present

# Package from URL
- name: Install deb package
  apt:
    deb: https://example.com/package.deb

# Remove package
- name: Remove package
  apt:
    name: apache2
    state: absent
    purge: yes
```

### Service Management

```yaml
- name: Manage service
  service:
    name: nginx
    state: started
    enabled: yes

- name: Restart service
  service:
    name: apache2
    state: restarted

- name: Reload service
  service:
    name: nginx
    state: reloaded
```

### Command Execution

```yaml
# Shell module
- name: Run shell command
  shell: echo $HOME
  register: home_dir

- name: Display output
  debug:
    var: home_dir.stdout

# Command module (no shell features)
- name: Run command
  command: /usr/bin/uptime
  register: uptime_result

# Script execution
- name: Run script
  script: scripts/setup.sh

# Execute with conditions
- name: Check file exists
  stat:
    path: /etc/config.conf
  register: config_file

- name: Run if file exists
  command: /usr/bin/process_config
  when: config_file.stat.exists
```

## Handlers

```yaml
---
- name: Configure nginx
  hosts: webservers
  become: yes

  tasks:
    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify:
        - restart nginx
        - reload nginx

    - name: Copy site config
      template:
        src: site.conf.j2
        dest: /etc/nginx/sites-available/default
      notify: reload nginx

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted

    - name: reload nginx
      service:
        name: nginx
        state: reloaded
```

## Roles

### Creating a Role

```bash
# Create role structure
ansible-galaxy init myrole

# Directory structure
myrole/
├── defaults/          # Default variables
│   └── main.yml
├── files/             # Static files
├── handlers/          # Handlers
│   └── main.yml
├── meta/              # Role metadata
│   └── main.yml
├── tasks/             # Main tasks
│   └── main.yml
├── templates/         # Jinja2 templates
├── tests/             # Test playbooks
│   └── test.yml
└── vars/              # Role variables
    └── main.yml
```

### Role Example

```yaml
# roles/nginx/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present
    update_cache: yes

- name: Copy nginx config
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: restart nginx

- name: Start nginx
  service:
    name: nginx
    state: started
    enabled: yes

# roles/nginx/handlers/main.yml
---
- name: restart nginx
  service:
    name: nginx
    state: restarted

# roles/nginx/defaults/main.yml
---
nginx_port: 80
nginx_user: www-data

# Using the role
---
- name: Setup web server
  hosts: webservers
  become: yes
  roles:
    - nginx
    - { role: mysql, mysql_port: 3306 }
```

## Conditionals and Loops

### Conditionals

```yaml
---
- name: Conditional tasks
  hosts: all
  tasks:
    - name: Install on Ubuntu
      apt:
        name: nginx
        state: present
      when: ansible_distribution == "Ubuntu"

    - name: Install on CentOS
      yum:
        name: httpd
        state: present
      when: ansible_distribution == "CentOS"

    - name: Multiple conditions (AND)
      apt:
        name: nginx
        state: present
      when:
        - ansible_distribution == "Ubuntu"
        - ansible_distribution_version == "20.04"

    - name: Multiple conditions (OR)
      apt:
        name: nginx
        state: present
      when: ansible_distribution == "Ubuntu" or ansible_distribution == "Debian"
```

### Loops

```yaml
---
- name: Loop examples
  hosts: all
  tasks:
    # Simple loop
    - name: Install multiple packages
      apt:
        name: "{{ item }}"
        state: present
      loop:
        - nginx
        - postgresql
        - redis-server

    # Loop with dictionary
    - name: Create users
      user:
        name: "{{ item.name }}"
        groups: "{{ item.groups }}"
        state: present
      loop:
        - { name: 'alice', groups: 'developers' }
        - { name: 'bob', groups: 'admins' }

    # Loop with complex data
    - name: Create directories
      file:
        path: "{{ item.path }}"
        state: directory
        owner: "{{ item.owner }}"
        mode: "{{ item.mode }}"
      loop:
        - { path: '/opt/app1', owner: 'deploy', mode: '0755' }
        - { path: '/opt/app2', owner: 'www-data', mode: '0750' }
```

## Templates

```jinja2
{# templates/nginx.conf.j2 #}
user {{ nginx_user }};
worker_processes {{ ansible_processor_vcpus }};

events {
    worker_connections 1024;
}

http {
    server {
        listen {{ nginx_port }};
        server_name {{ ansible_hostname }};

        location / {
            root /var/www/html;
            index index.html;
        }
    }
}

{# Conditional content #}
{% if enable_ssl %}
    ssl on;
    ssl_certificate {{ ssl_cert_path }};
{% endif %}

{# Loop in template #}
{% for server in backend_servers %}
    upstream backend_{{ loop.index }} {
        server {{ server.host }}:{{ server.port }};
    }
{% endfor %}
```

## Vault (Encryption)

```bash
# Create encrypted file
ansible-vault create secrets.yml

# Edit encrypted file
ansible-vault edit secrets.yml

# Encrypt existing file
ansible-vault encrypt vars.yml

# Decrypt file
ansible-vault decrypt vars.yml

# View encrypted file
ansible-vault view secrets.yml

# Change password
ansible-vault rekey secrets.yml

# Use vault in playbook
ansible-playbook playbook.yml --ask-vault-pass

# Use password file
ansible-playbook playbook.yml --vault-password-file ~/.vault_pass

# Multiple vaults
ansible-playbook playbook.yml --vault-id prod@prompt --vault-id dev@~/.vault_pass_dev
```

### Vault Example

```yaml
# secrets.yml (encrypted)
db_password: "super_secret_password"
api_key: "abc123xyz789"

# playbook.yml
---
- name: Deploy with secrets
  hosts: webservers
  vars_files:
    - secrets.yml
  tasks:
    - name: Configure database
      template:
        src: db_config.j2
        dest: /etc/app/db_config.conf
```

## Best Practices

### Playbook Organization

```bash
# Recommended directory structure
site.yml                # Master playbook
webservers.yml          # Webserver playbook
dbservers.yml           # Database playbook

inventory/
├── production/
│   ├── hosts
│   └── group_vars/
│       ├── all.yml
│       ├── webservers.yml
│       └── dbservers.yml
└── staging/
    ├── hosts
    └── group_vars/

roles/
├── common/
├── nginx/
├── postgresql/
└── app/

group_vars/
├── all.yml
├── webservers.yml
└── dbservers.yml

host_vars/
└── web1.example.com.yml
```

### Best Practices

```yaml
# 1. Use names for all tasks
- name: Install nginx
  apt:
    name: nginx
    state: present

# 2. Use become appropriately
- name: System task
  become: yes
  apt:
    name: nginx
    state: present

# 3. Validate configurations
- name: Deploy nginx config
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
    validate: 'nginx -t -c %s'

# 4. Use check mode compatible tasks
- name: Check if service exists
  stat:
    path: /etc/systemd/system/myapp.service
  register: service_file
  check_mode: no

# 5. Add tags
- name: Install packages
  apt:
    name: nginx
    state: present
  tags: ['install', 'packages']

# 6. Use blocks for error handling
- block:
    - name: Risky operation
      command: /usr/bin/risky_command
  rescue:
    - name: Handle error
      debug:
        msg: "Command failed, handling gracefully"
  always:
    - name: Cleanup
      file:
        path: /tmp/temp_file
        state: absent
```

## Common Patterns

### Complete Web Server Setup

```yaml
---
- name: Configure web servers
  hosts: webservers
  become: yes
  vars:
    app_name: myapp
    app_user: www-data

  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600

    - name: Install packages
      apt:
        name:
          - nginx
          - python3-pip
          - git
        state: present

    - name: Create app directory
      file:
        path: "/var/www/{{ app_name }}"
        state: directory
        owner: "{{ app_user }}"
        mode: '0755'

    - name: Deploy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
      notify: reload nginx

    - name: Enable site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}
        dest: /etc/nginx/sites-enabled/{{ app_name }}
        state: link
      notify: reload nginx

    - name: Start nginx
      service:
        name: nginx
        state: started
        enabled: yes

  handlers:
    - name: reload nginx
      service:
        name: nginx
        state: reloaded
```

### Multi-stage Deployment

```yaml
---
- name: Deploy application
  hosts: webservers
  serial: 1  # Rolling update
  max_fail_percentage: 25

  pre_tasks:
    - name: Remove from load balancer
      haproxy:
        state: disabled
        host: "{{ ansible_hostname }}"

  tasks:
    - name: Deploy application
      git:
        repo: https://github.com/user/app.git
        dest: /opt/app
        version: "{{ app_version }}"

    - name: Install dependencies
      pip:
        requirements: /opt/app/requirements.txt

    - name: Restart app service
      service:
        name: myapp
        state: restarted

    - name: Wait for app to start
      wait_for:
        port: 8080
        delay: 5
        timeout: 30

  post_tasks:
    - name: Add to load balancer
      haproxy:
        state: enabled
        host: "{{ ansible_hostname }}"
```

## Troubleshooting

```bash
# Check syntax
ansible-playbook playbook.yml --syntax-check

# List tasks
ansible-playbook playbook.yml --list-tasks

# List hosts
ansible-playbook playbook.yml --list-hosts

# Dry run
ansible-playbook playbook.yml --check

# Debug mode
ansible-playbook playbook.yml -vvv

# Start at specific task
ansible-playbook playbook.yml --start-at-task="Install nginx"

# Step through playbook
ansible-playbook playbook.yml --step

# Gather facts only
ansible all -m setup --tree /tmp/facts
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `ansible all -m ping` | Ping all hosts |
| `ansible-playbook playbook.yml` | Run playbook |
| `ansible-playbook --check` | Dry run |
| `ansible-playbook --tags TAG` | Run specific tags |
| `ansible-playbook --limit HOST` | Limit to hosts |
| `ansible-vault create FILE` | Create encrypted file |
| `ansible-galaxy init ROLE` | Create role |
| `ansible-inventory --list` | Show inventory |

Ansible simplifies IT automation with its agentless architecture and simple YAML syntax, making infrastructure management efficient and reproducible.
