# tmux

tmux is a terminal multiplexer that allows you to create multiple terminal sessions inside a single window. It is a powerful tool that can be used to manage multiple terminals, run multiple commands, and more.
## tmux.conf

```bash
cat << 'EOF' > ~/.tmux.conf
set-option -g prefix C-a
unbind-key C-b
bind-key C-a last-window
EOF