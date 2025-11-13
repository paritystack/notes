# tmux

tmux (terminal multiplexer) is a powerful tool that allows you to create, access, and control multiple terminal sessions from a single window. It enables session persistence, split panes, and window management.

## Overview

tmux allows you to:
- Run multiple terminal sessions in a single window
- Split your terminal into multiple panes
- Detach and reattach sessions (sessions persist after disconnection)
- Share sessions between users
- Script and automate terminal workflows

**Key Concepts:**
- **Session**: A collection of windows, managed independently
- **Window**: A single screen within a session (like a tab)
- **Pane**: A split section within a window
- **Prefix Key**: Default `Ctrl+b`, used before tmux commands
- **Detach**: Disconnect from session (keeps running in background)
- **Attach**: Reconnect to an existing session

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tmux

# macOS
brew install tmux

# CentOS/RHEL
sudo yum install tmux

# Arch Linux
sudo pacman -S tmux

# Verify installation
tmux -V
```

## Basic Usage

### Session Management

```bash
# Start new session
tmux

# Start new session with name
tmux new -s mysession
tmux new-session -s mysession

# List sessions
tmux ls
tmux list-sessions

# Attach to session
tmux attach
tmux a

# Attach to specific session
tmux attach -t mysession
tmux a -t mysession

# Detach from session (inside tmux)
# Press: Ctrl+b, then d

# Kill session
tmux kill-session -t mysession

# Kill all sessions
tmux kill-server

# Rename session (inside tmux)
# Press: Ctrl+b, then $
```

### Window Management

```bash
# Inside tmux, press Ctrl+b then:

# c - Create new window
# , - Rename current window
# w - List windows
# n - Next window
# p - Previous window
# 0-9 - Switch to window number
# l - Last active window
# & - Kill current window
# f - Find window by name
```

### Pane Management

```bash
# Inside tmux, press Ctrl+b then:

# % - Split pane vertically
# " - Split pane horizontally
# Arrow keys - Navigate between panes
# o - Switch to next pane
# ; - Toggle between current and previous pane
# x - Kill current pane
# z - Toggle pane zoom (fullscreen)
# Space - Toggle between layouts
# { - Move pane left
# } - Move pane right
# Ctrl+Arrow - Resize pane
# q - Show pane numbers (then press number to switch)
```

## Configuration

### Basic .tmux.conf

```bash
# Create configuration file
cat << 'EOF' > ~/.tmux.conf
# Change prefix from Ctrl+b to Ctrl+a
set-option -g prefix C-a
unbind-key C-b
bind-key C-a send-prefix

# Enable mouse support
set -g mouse on

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows when one is closed
set -g renumber-windows on

# Increase scrollback buffer size
set -g history-limit 10000

# Enable 256 colors
set -g default-terminal "screen-256color"

# Reload config file
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Split panes with | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# Switch panes using Alt+Arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Set status bar
set -g status-bg black
set -g status-fg white
set -g status-interval 60
set -g status-left-length 30
set -g status-left '#[fg=green](#S) #(whoami) '
set -g status-right '#[fg=yellow]#(cut -d " " -f 1-3 /proc/loadavg)#[default] #[fg=white]%H:%M#[default]'
EOF

# Reload tmux configuration
tmux source-file ~/.tmux.conf
```

### Advanced Configuration

```bash
cat << 'EOF' > ~/.tmux.conf
# ===== Basic Settings =====
set-option -g prefix C-a
unbind-key C-b
bind-key C-a send-prefix

# Enable mouse
set -g mouse on

# Start numbering at 1
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows
set -g renumber-windows on

# History
set -g history-limit 50000

# Terminal settings
set -g default-terminal "screen-256color"
set -ga terminal-overrides ",*256col*:Tc"

# No delay for escape key
set -sg escape-time 0

# Monitor activity
setw -g monitor-activity on
set -g visual-activity off

# ===== Key Bindings =====

# Reload config
bind r source-file ~/.tmux.conf \; display "Reloaded!"

# Split panes
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
bind c new-window -c "#{pane_current_path}"

# Pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Pane resizing
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# Window navigation
bind -r C-h select-window -t :-
bind -r C-l select-window -t :+

# Copy mode with vi keys
setw -g mode-keys vi
bind-key -T copy-mode-vi 'v' send -X begin-selection
bind-key -T copy-mode-vi 'y' send -X copy-selection-and-cancel

# ===== Appearance =====

# Status bar
set -g status-position bottom
set -g status-justify left
set -g status-style 'bg=colour234 fg=colour137'
set -g status-left ''
set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '
set -g status-right-length 50
set -g status-left-length 20

# Window status
setw -g window-status-current-style 'fg=colour1 bg=colour19 bold'
setw -g window-status-current-format ' #I#[fg=colour249]:#[fg=colour255]#W#[fg=colour249]#F '
setw -g window-status-style 'fg=colour9 bg=colour18'
setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '

# Pane borders
set -g pane-border-style 'fg=colour238'
set -g pane-active-border-style 'fg=colour51'

# Message text
set -g message-style 'fg=colour232 bg=colour166 bold'
EOF
```

## Key Bindings Reference

### Default Prefix: `Ctrl+b`

#### Session Commands

```bash
Ctrl+b d       # Detach from session
Ctrl+b s       # List sessions
Ctrl+b $       # Rename session
Ctrl+b (       # Switch to previous session
Ctrl+b )       # Switch to next session
Ctrl+b L       # Switch to last session
```

#### Window Commands

```bash
Ctrl+b c       # Create new window
Ctrl+b ,       # Rename current window
Ctrl+b &       # Kill current window
Ctrl+b w       # List windows
Ctrl+b n       # Next window
Ctrl+b p       # Previous window
Ctrl+b 0-9     # Switch to window by number
Ctrl+b l       # Switch to last active window
Ctrl+b f       # Find window
Ctrl+b .       # Move window (prompts for index)
```

#### Pane Commands

```bash
Ctrl+b %       # Split vertically
Ctrl+b "       # Split horizontally
Ctrl+b o       # Switch to next pane
Ctrl+b ;       # Toggle between current and previous pane
Ctrl+b x       # Kill current pane
Ctrl+b !       # Break pane into window
Ctrl+b z       # Toggle pane zoom
Ctrl+b Space   # Toggle between pane layouts
Ctrl+b q       # Show pane numbers
Ctrl+b {       # Move pane left
Ctrl+b }       # Move pane right
Ctrl+b Ctrl+o  # Rotate panes
Ctrl+b Arrow   # Navigate panes
```

#### Copy Mode

```bash
Ctrl+b [       # Enter copy mode
Ctrl+b ]       # Paste buffer
Space          # Start selection (in copy mode)
Enter          # Copy selection (in copy mode)
q              # Exit copy mode

# With vi mode enabled:
v              # Begin selection
y              # Copy selection
```

#### Other Commands

```bash
Ctrl+b ?       # List all key bindings
Ctrl+b :       # Enter command mode
Ctrl+b t       # Show time
Ctrl+b ~       # Show messages
```

## Common Workflows

### Development Environment

```bash
# Create development session
tmux new -s dev

# Inside tmux:
# Split into 3 panes
Ctrl+b %    # Split vertically
Ctrl+b "    # Split right pane horizontally

# Now you have:
# - Left pane: Editor (vim/emacs)
# - Top right: Run server
# - Bottom right: Git/commands

# Navigate between panes
Ctrl+b Arrow keys
```

### Remote Server Session

```bash
# SSH to server
ssh user@server

# Start tmux session
tmux new -s work

# Do work...
# Connection drops or intentional detach
Ctrl+b d

# Reconnect later
ssh user@server
tmux attach -t work
# Your session is exactly as you left it
```

### Pair Programming

```bash
# User 1: Create session
tmux new -s pair

# User 2: Attach to same session (read-only)
tmux attach -t pair -r

# User 2: Attach with full control
tmux attach -t pair
```

### Multiple Projects

```bash
# Create sessions for different projects
tmux new -s project1 -d
tmux new -s project2 -d
tmux new -s project3 -d

# List all sessions
tmux ls

# Attach to specific project
tmux attach -t project1

# Switch between sessions (inside tmux)
Ctrl+b s    # Shows session list
Ctrl+b (    # Previous session
Ctrl+b )    # Next session
```

## Advanced Features

### Copy and Paste

```bash
# Enter copy mode
Ctrl+b [

# Navigate with vi keys (if vi mode enabled)
# Or use arrow keys

# Start selection
Space

# Copy selection
Enter

# Paste
Ctrl+b ]

# View paste buffers
Ctrl+b #

# Choose buffer to paste
Ctrl+b =
```

### Synchronized Panes

```bash
# Enable synchronized panes (type in all panes at once)
Ctrl+b :
:setw synchronize-panes on

# Disable
:setw synchronize-panes off

# Toggle with binding (add to .tmux.conf)
bind S setw synchronize-panes
```

### Save and Restore Sessions

```bash
# Save session layout
Ctrl+b :
:save-buffer /tmp/tmux-session.txt

# Create script to restore layout
cat << 'EOF' > ~/restore-session.sh
#!/bin/bash
tmux new-session -d -s dev
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux send-keys 'vim' C-m
tmux select-pane -t 1
tmux send-keys 'npm run dev' C-m
tmux select-pane -t 2
tmux attach -t dev
EOF

chmod +x ~/restore-session.sh
```

### Tmux Plugins (TPM)

```bash
# Install Tmux Plugin Manager
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# Add to .tmux.conf
cat << 'EOF' >> ~/.tmux.conf

# List of plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# Initialize TPM (keep at bottom of .tmux.conf)
run '~/.tmux/plugins/tpm/tpm'
EOF

# Reload config
tmux source ~/.tmux.conf

# Install plugins (inside tmux)
Ctrl+b I
```

### Custom Scripts

```bash
# Create reusable session layout
cat << 'EOF' > ~/tmux-dev.sh
#!/bin/bash

SESSION="dev"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

if [ "$SESSIONEXISTS" = "" ]
then
    # Create new session
    tmux new-session -d -s $SESSION

    # Create windows
    tmux rename-window -t 0 'Editor'
    tmux send-keys -t 'Editor' 'cd ~/project && vim' C-m

    tmux new-window -t $SESSION:1 -n 'Server'
    tmux send-keys -t 'Server' 'cd ~/project && npm run dev' C-m

    tmux new-window -t $SESSION:2 -n 'Git'
    tmux send-keys -t 'Git' 'cd ~/project && git status' C-m

    # Split panes
    tmux select-window -t $SESSION:2
    tmux split-window -h
    tmux send-keys -t 1 'cd ~/project' C-m
fi

# Attach to session
tmux attach-session -t $SESSION:0
EOF

chmod +x ~/tmux-dev.sh
```

## Command Mode

```bash
# Enter command mode
Ctrl+b :

# Common commands
:new-window -n mywindow
:kill-window
:split-window -h
:resize-pane -D 10
:setw synchronize-panes on
:set mouse on
:source-file ~/.tmux.conf
:list-keys
:list-commands
```

## Scripting tmux

### Create Complex Layouts

```bash
#!/bin/bash

# Create session with specific layout
tmux new-session -d -s complex

# Split into 4 panes
tmux split-window -h -t complex
tmux split-window -v -t complex:0.0
tmux split-window -v -t complex:0.2

# Send commands to each pane
tmux send-keys -t complex:0.0 'htop' C-m
tmux send-keys -t complex:0.1 'tail -f /var/log/syslog' C-m
tmux send-keys -t complex:0.2 'vim' C-m
tmux send-keys -t complex:0.3 'echo "Ready for commands"' C-m

# Attach to session
tmux attach -t complex
```

### Automation Script

```bash
#!/bin/bash

# Monitor multiple servers
SERVERS=("server1" "server2" "server3")
SESSION="monitoring"

tmux new-session -d -s $SESSION

for i in "${!SERVERS[@]}"; do
    if [ $i -eq 0 ]; then
        tmux rename-window -t $SESSION:0 "${SERVERS[$i]}"
    else
        tmux new-window -t $SESSION:$i -n "${SERVERS[$i]}"
    fi

    tmux send-keys -t $SESSION:$i "ssh ${SERVERS[$i]}" C-m
done

tmux select-window -t $SESSION:0
tmux attach -t $SESSION
```

## Best Practices

### Recommended .tmux.conf Settings

```bash
# Essential settings
set -g mouse on                      # Enable mouse
set -g history-limit 50000           # Large scrollback
set -sg escape-time 0                # No escape delay
set -g base-index 1                  # Start windows at 1
setw -g pane-base-index 1            # Start panes at 1
set -g renumber-windows on           # Renumber windows

# Visual settings
set -g default-terminal "screen-256color"
set -g status-position bottom
setw -g monitor-activity on

# Key bindings
bind r source-file ~/.tmux.conf \; display "Reloaded!"
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
setw -g mode-keys vi
```

### Workflow Tips

1. **Use named sessions** for different projects
2. **Create restore scripts** for complex layouts
3. **Enable mouse support** for easier navigation
4. **Use vi key bindings** in copy mode
5. **Set up custom key bindings** for frequent actions
6. **Use tmux with SSH** for persistent remote sessions
7. **Share sessions** for collaboration
8. **Create aliases** for common commands

### Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias tm='tmux'
alias tma='tmux attach -t'
alias tms='tmux new-session -s'
alias tml='tmux list-sessions'
alias tmk='tmux kill-session -t'
```

## Troubleshooting

### Common Issues

```bash
# Prefix key not working
# Check if prefix is correct in .tmux.conf
tmux show-options -g | grep prefix

# Colors not displaying correctly
set -g default-terminal "screen-256color"

# Mouse not working
set -g mouse on

# Sessions not persisting
# Make sure you detach (Ctrl+b d) instead of exiting

# Can't attach to session
# Check if session exists
tmux ls

# Configuration not loading
# Reload config
tmux source-file ~/.tmux.conf

# Reset tmux to defaults
tmux kill-server
rm ~/.tmux.conf
```

### Debug Mode

```bash
# Start tmux in verbose mode
tmux -v

# Show current settings
tmux show-options -g
tmux show-window-options -g

# Check key bindings
tmux list-keys

# Show messages
Ctrl+b ~
```

## Integration with Tools

### Vim Integration

```bash
# Add to .vimrc for seamless navigation
if exists('$TMUX')
    " Use same keybindings for vim and tmux
    let g:tmux_navigator_no_mappings = 1
endif
```

### Shell Integration

```bash
# Auto-attach or create session
if command -v tmux &> /dev/null && [ -z "$TMUX" ]; then
    tmux attach -t default || tmux new -s default
fi
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `tmux` | Start new session |
| `tmux new -s name` | Start named session |
| `tmux ls` | List sessions |
| `tmux attach -t name` | Attach to session |
| `Ctrl+b d` | Detach from session |
| `Ctrl+b c` | Create window |
| `Ctrl+b ,` | Rename window |
| `Ctrl+b %` | Split vertically |
| `Ctrl+b "` | Split horizontally |
| `Ctrl+b Arrow` | Navigate panes |
| `Ctrl+b z` | Zoom pane |
| `Ctrl+b [` | Copy mode |
| `Ctrl+b ?` | List keybindings |

tmux is an essential tool for managing terminal workflows, especially valuable for remote server management, development environments, and maintaining persistent sessions.
