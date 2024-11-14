# tmux

## tmux.conf

```bash
cat << 'EOF' > ~/.tmux.conf
set-option -g prefix C-a
unbind-key C-b
bind-key C-a last-window
EOF