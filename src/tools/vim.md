# Vim

Vim is a powerful, highly configurable text editor built to enable efficient text editing. It is the improved version of the vi editor distributed with most UNIX systems. Vim is designed for use both from a command line interface and as a standalone application in a graphical user interface.

## Philosophy & Core Concepts

Vim's power comes from its unique approach to text editing:

### Modal Editing

Unlike traditional editors, Vim is a **modal editor** with multiple modes, each optimized for specific tasks:

- **Normal Mode**: Navigate and manipulate text (default mode)
- **Insert Mode**: Type and insert text
- **Visual Mode**: Select and manipulate text regions
- **Command-line Mode**: Execute commands, search, and perform operations
- **Replace Mode**: Overwrite existing text

This separation allows the same keys to perform different functions depending on the mode, dramatically increasing the number of available commands without requiring complex key combinations.

### Composability

Vim's commands follow a grammar-like structure:

```
[count] operator [count] motion/text-object
```

**Examples:**
- `d2w` - Delete 2 words
- `c3j` - Change 3 lines down
- `y$` - Yank to end of line
- `3dd` - Delete 3 lines
- `di"` - Delete inside quotes
- `gUap` - Uppercase a paragraph

This composability means learning a few operators and motions gives you hundreds of combinations.

### Efficiency & Speed

Vim is designed to keep your hands on the home row of the keyboard, minimizing movement and maximizing speed. Once mastered, Vim allows for text manipulation at the speed of thought.

### Repeatability

The `.` command repeats the last change, and macros allow recording complex operations for replay. This makes repetitive tasks trivial.


## Vim Modes

Understanding Vim's modes is fundamental to mastering the editor.

### Normal Mode (Command Mode)

The default mode for navigation and text manipulation. Press `Esc` from any mode to return to Normal mode.

**Entering Normal Mode:**
- `Esc` - From any mode
- `Ctrl+[` - Alternative to Esc
- `Ctrl+c` - Alternative (may skip some autocmds)

### Insert Mode

Mode for typing and inserting text.

**Entering Insert Mode:**
- `i` - Insert before cursor
- `I` - Insert at beginning of line
- `a` - Append after cursor
- `A` - Append at end of line
- `o` - Open new line below and insert
- `O` - Open new line above and insert
- `s` - Substitute character (delete char and insert)
- `S` - Substitute line (delete line and insert)
- `C` - Change to end of line (delete to end and insert)
- `gi` - Insert at last insert position

### Visual Mode

Mode for selecting and manipulating text regions.

**Entering Visual Mode:**
- `v` - Character-wise visual mode
- `V` - Line-wise visual mode
- `Ctrl+v` - Block-wise visual mode (vertical selection)
- `gv` - Reselect last visual selection

**Visual Mode Operations:**
Once text is selected:
- `d` - Delete selection
- `y` - Yank (copy) selection
- `c` - Change selection (delete and enter insert mode)
- `>` - Indent selection right
- `<` - Indent selection left
- `=` - Auto-indent selection
- `u` - Lowercase selection
- `U` - Uppercase selection
- `~` - Toggle case

### Command-line Mode

Mode for executing Ex commands.

**Entering Command-line Mode:**
- `:` - Enter command-line for Ex commands
- `/` - Search forward
- `?` - Search backward
- `!` - Execute external command

### Replace Mode

Mode for overwriting existing text.

**Entering Replace Mode:**
- `R` - Enter replace mode (continuous overwrite)
- `r` - Replace single character and return to normal mode
- `gr` - Virtual replace (respects tabs and preserves layout)

## Navigation

Efficient navigation is key to Vim mastery. Avoid using arrow keys - embrace Vim's powerful motion commands.

### Basic Motion (Character & Line)

**Character Movement:**
- `h` - Left
- `j` - Down
- `k` - Up
- `l` - Right
- `gj` - Down (screen line, not file line - useful for wrapped text)
- `gk` - Up (screen line)

**Horizontal (within line):**
- `0` - To first character of line
- `^` - To first non-blank character of line
- `$` - To end of line
- `g_` - To last non-blank character of line
- `|` - To column 0
- `{n}|` - To column n

### Word Motion

**Forward:**
- `w` - Start of next word
- `W` - Start of next WORD (space-separated)
- `e` - End of current/next word
- `E` - End of current/next WORD
- `ge` - End of previous word
- `gE` - End of previous WORD

**Backward:**
- `b` - Start of previous word
- `B` - Start of previous WORD

**Word vs WORD:**
- **word**: Delimited by non-keyword characters (a-zA-Z0-9_)
- **WORD**: Delimited by whitespace only

Example: `foo-bar` is 3 words (`foo`, `-`, `bar`) but 1 WORD

### Line Motion

- `j` / `k` - Down / Up one line
- `{n}j` / `{n}k` - Down / Up n lines
- `+` / `-` - Down / Up to first non-blank character
- `G` - Go to last line
- `gg` - Go to first line
- `{n}G` or `:{n}` - Go to line n
- `{n}%` - Go to n% through file

### Paragraph & Block Motion

- `{` - Move to previous paragraph (or block)
- `}` - Move to next paragraph (or block)
- `[[` - Move to previous section (or function start in code)
- `]]` - Move to next section (or function start in code)
- `[]` - Move to previous section end
- `][` - Move to next section end

### Screen Motion

**Relative to screen:**
- `H` - Move to top of screen (High)
- `M` - Move to middle of screen (Middle)
- `L` - Move to bottom of screen (Low)
- `{n}H` - Move to n lines from top
- `{n}L` - Move to n lines from bottom

**Scrolling:**
- `Ctrl+f` - Scroll forward (full screen)
- `Ctrl+b` - Scroll backward (full screen)
- `Ctrl+d` - Scroll down (half screen)
- `Ctrl+u` - Scroll up (half screen)
- `Ctrl+e` - Scroll down one line
- `Ctrl+y` - Scroll up one line
- `zz` - Center cursor on screen
- `zt` - Move cursor to top of screen
- `zb` - Move cursor to bottom of screen

### Character Search (within line)

- `f{char}` - Find next occurrence of {char} (forward)
- `F{char}` - Find previous occurrence of {char} (backward)
- `t{char}` - Till next occurrence of {char} (stop before)
- `T{char}` - Till previous occurrence of {char} (stop after)
- `;` - Repeat last f, t, F, or T
- `,` - Repeat last f, t, F, or T in opposite direction

**Example:** `df,` - Delete up to and including next comma

### Marks & Jumps

**Setting Marks:**
- `m{a-z}` - Set local mark (local to file)
- `m{A-Z}` - Set global mark (across files)

**Jumping to Marks:**
- `'{mark}` - Jump to line of mark
- `` `{mark}`` - Jump to exact position of mark
- `''` - Jump to position before last jump
- ``` `` ``` - Jump to exact position before last jump

**Special Marks:**
- `'.` - Jump to last change
- `` `^ `` - Jump to last insert position
- `` `[ `` - Jump to beginning of last change
- `` `] `` - Jump to end of last change
- `` `< `` - Jump to beginning of last visual selection
- `` `> `` - Jump to end of last visual selection

**Jump List:**
- `Ctrl+o` - Jump to older position in jump list
- `Ctrl+i` - Jump to newer position in jump list
- `:jumps` - Show jump list

### Pattern Search

- `/pattern` - Search forward for pattern
- `?pattern` - Search backward for pattern
- `n` - Repeat search in same direction
- `N` - Repeat search in opposite direction
- `*` - Search forward for word under cursor
- `#` - Search backward for word under cursor
- `g*` - Search forward for word under cursor (partial match)
- `g#` - Search backward for word under cursor (partial match)
- `/` - Repeat last forward search
- `?` - Repeat last backward search

**Search Options:**
- `/pattern/e` - Search and place cursor at end of match
- `/pattern/+n` - Search and move n lines down from match
- `/pattern/-n` - Search and move n lines up from match

## Text Objects

Text objects are one of Vim's most powerful features. They allow you to operate on semantic units of text.

### Syntax

Text objects are used with operators:
```
operator + a/i + text-object
```

- `a` - "a" or "around" (includes surrounding whitespace/delimiters)
- `i` - "inner" or "inside" (excludes surrounding whitespace/delimiters)

### Word Text Objects

- `aw` - A word (includes surrounding whitespace)
- `iw` - Inner word (excludes surrounding whitespace)
- `aW` - A WORD (space-separated)
- `iW` - Inner WORD

**Examples:**
- `diw` - Delete inner word
- `ciw` - Change inner word
- `yaw` - Yank a word (with space)

### Sentence & Paragraph Text Objects

- `as` - A sentence
- `is` - Inner sentence
- `ap` - A paragraph
- `ip` - Inner paragraph

### Quote Text Objects

- `a"` - A double-quoted string (including quotes)
- `i"` - Inner double-quoted string (excluding quotes)
- `a'` - A single-quoted string (including quotes)
- `i'` - Inner single-quoted string (excluding quotes)
- ``` a` ``` - A back-quoted string (including back-quotes)
- ``` i` ``` - Inner back-quoted string (excluding back-quotes)

**Examples:**
- `di"` - Delete inside quotes
- `ci'` - Change inside single quotes
- `ya"` - Yank around quotes (including quotes)

### Bracket/Parenthesis Text Objects

- `a)` or `ab` - A block () (including parentheses)
- `i)` or `ib` - Inner block () (excluding parentheses)
- `a]` - A block [] (including brackets)
- `i]` - Inner block [] (excluding brackets)
- `a}` or `aB` - A block {} (including braces)
- `i}` or `iB` - Inner block {} (excluding braces)
- `a>` - A block <> (including angle brackets)
- `i>` - Inner block <> (excluding angle brackets)

**Examples:**
- `di(` - Delete inside parentheses
- `da{` - Delete around braces (including braces)
- `ci]` - Change inside brackets
- `ya}` - Yank around braces

### Tag Text Objects (XML/HTML)

- `at` - A tag block (including tags)
- `it` - Inner tag block (excluding tags)

**Example:**
```html
<div>Hello World</div>
```
- Cursor on "Hello", `dit` → deletes "Hello World"
- Cursor on "Hello", `dat` → deletes `<div>Hello World</div>`

## Operators

Operators perform actions on text. Combined with motions or text objects, they form powerful commands.

### Common Operators

- `d` - Delete
- `c` - Change (delete and enter insert mode)
- `y` - Yank (copy)
- `p` - Put (paste) after cursor/line
- `P` - Put (paste) before cursor/line
- `~` - Toggle case
- `gu` - Make lowercase
- `gU` - Make uppercase
- `g~` - Toggle case
- `>` - Indent right
- `<` - Indent left
- `=` - Auto-indent
- `!` - Filter through external command

### Operator-Motion Combinations

The power of Vim comes from combining operators with motions:

**Delete:**
- `dw` - Delete word
- `d$` or `D` - Delete to end of line
- `d0` - Delete to beginning of line
- `dd` - Delete line
- `dj` - Delete current and next line
- `d/pattern` - Delete to pattern

**Change:**
- `cw` - Change word
- `c$` or `C` - Change to end of line
- `cc` or `S` - Change entire line
- `ct{char}` - Change till {char}
- `ci{` - Change inside braces

**Yank (Copy):**
- `yw` - Yank word
- `y$` - Yank to end of line
- `yy` or `Y` - Yank entire line
- `yap` - Yank a paragraph
- `yi"` - Yank inside quotes

**Case Change:**
- `guw` - Lowercase word
- `gUw` - Uppercase word
- `g~w` - Toggle case of word
- `guap` - Lowercase paragraph
- `gUap` - Uppercase paragraph

### Doubling an Operator

Many operators can be doubled to operate on the current line:
- `dd` - Delete line
- `yy` - Yank line
- `cc` - Change line
- `>>` - Indent line right
- `<<` - Indent line left
- `==` - Auto-indent line
- `g~~` - Toggle case of line

## Editing Operations

### Basic Editing

**Insert/Append:**
- `i` / `a` - Insert before / after cursor
- `I` / `A` - Insert at beginning / end of line
- `o` / `O` - Open line below / above

**Delete:**
- `x` - Delete character under cursor
- `X` - Delete character before cursor
- `s` - Substitute character (delete and insert)
- `D` - Delete to end of line
- `dd` - Delete line

**Replace:**
- `r{char}` - Replace single character
- `R` - Enter replace mode
- `~` - Toggle case of character

**Join Lines:**
- `J` - Join current line with next (remove newline, add space)
- `gJ` - Join without adding space

**Increment/Decrement Numbers:**
- `Ctrl+a` - Increment number under cursor
- `Ctrl+x` - Decrement number under cursor
- `{n}Ctrl+a` - Increment by n

### Undo & Redo

- `u` - Undo last change
- `Ctrl+r` - Redo last undone change
- `U` - Undo all changes on line
- `:earlier {time}` - Go to earlier text state (e.g., `:earlier 10m`)
- `:later {time}` - Go to later text state
- `:undolist` - Show undo tree
- `g+` / `g-` - Navigate undo tree (newer/older)

### Copy & Paste

**Copy (Yank):**
- `yy` or `Y` - Yank line
- `yw` - Yank word
- `y$` - Yank to end of line
- `yiw` - Yank inner word
- `yi"` - Yank inside quotes

**Paste (Put):**
- `p` - Paste after cursor/line
- `P` - Paste before cursor/line
- `gp` - Paste and move cursor after pasted text
- `gP` - Paste before and move cursor

**Paste in Insert Mode:**
- `Ctrl+r {register}` - Paste from register in insert mode
- `Ctrl+r "` - Paste from default register
- `Ctrl+r 0` - Paste from yank register

### Indentation

- `>>` - Indent line right
- `<<` - Indent line left
- `==` - Auto-indent line
- `>{motion}` - Indent motion right (e.g., `>ap` indent paragraph)
- `<{motion}` - Indent motion left
- `={motion}` - Auto-indent motion
- `gg=G` - Auto-indent entire file

**Visual Mode Indentation:**
- Select lines with `V`, then `>` or `<`
- Press `.` to repeat indentation

### Line Manipulation

- `:m {line}` - Move current line to after {line}
- `:m +1` - Move line down
- `:m -2` - Move line up
- `ddp` - Swap current line with next
- `ddkP` - Swap current line with previous

## Search and Replace

### Search

**Basic Search:**
- `/pattern` - Search forward
- `?pattern` - Search backward
- `n` - Next match
- `N` - Previous match
- `*` - Search word under cursor (forward)
- `#` - Search word under cursor (backward)

**Search Options:**
- `:set hlsearch` - Highlight search matches
- `:set incsearch` - Incremental search (search as you type)
- `:noh` or `:nohlsearch` - Clear search highlighting
- `/pattern\c` - Case-insensitive search
- `/pattern\C` - Case-sensitive search
- `/\<word\>` - Search for exact word (whole word match)

**Search History:**
- `/` then `↑`/`↓` - Browse search history
- `q/` - Open search history window
- `q?` - Open backward search history window

### Substitution (Find and Replace)

**Syntax:**
```
:[range]s/pattern/replacement/[flags]
```

**Basic Examples:**
- `:s/foo/bar/` - Replace first occurrence of "foo" with "bar" on current line
- `:s/foo/bar/g` - Replace all occurrences of "foo" with "bar" on current line
- `:%s/foo/bar/g` - Replace all occurrences in entire file
- `:5,12s/foo/bar/g` - Replace in lines 5-12
- `:'<,'>s/foo/bar/g` - Replace in visual selection

**Common Flags:**
- `g` - Global (all occurrences in line)
- `c` - Confirm each substitution
- `i` - Case-insensitive
- `I` - Case-sensitive
- `n` - Report number of matches, don't substitute

**Advanced Examples:**
- `:%s/foo/bar/gc` - Replace all with confirmation
- `:%s/\<foo\>/bar/g` - Replace whole word "foo"
- `:%s/foo\|baz/bar/g` - Replace "foo" or "baz" with "bar"
- `:%s/\(pattern\)/\1_suffix/g` - Capture and reuse (\1 is captured group)
- `:%s/old/\=@"/g` - Replace with contents of register

**Special Characters in Replacement:**
- `&` - Entire matched pattern
- `\0` - Entire matched pattern
- `\1`, `\2`, etc. - Captured groups
- `\u` - Uppercase next character
- `\l` - Lowercase next character
- `\U` - Uppercase until `\E`
- `\L` - Lowercase until `\E`

### Global Commands

Execute command on lines matching pattern:

```
:[range]g/pattern/command
```

**Examples:**
- `:g/pattern/d` - Delete all lines containing pattern
- `:g!/pattern/d` or `:v/pattern/d` - Delete lines NOT containing pattern
- `:g/TODO/p` - Print all lines containing "TODO"
- `:g/^$/d` - Delete all empty lines
- `:g/pattern/normal @a` - Execute macro `a` on matching lines
- `:g/pattern/t$` - Copy matching lines to end of file

## Registers

Registers are storage locations for text. Vim has multiple registers for different purposes.

### Register Types

**Named Registers (a-z):**
- `"ayy` - Yank line to register `a`
- `"ap` - Paste from register `a`
- `"Ayy` - Append line to register `a` (uppercase appends)

**Numbered Registers (0-9):**
- `"0` - Last yank
- `"1-"9` - Last 9 deletes (delete history)

**Special Registers:**
- `""` - Default (unnamed) register
- `"+` - System clipboard (requires +clipboard)
- `"*` - Primary selection (X11, requires +clipboard)
- `".` - Last inserted text
- `"%` - Current file name
- `":` - Last command
- `"/` - Last search pattern
- `"_` - Black hole register (doesn't store anything)

**Using Registers:**
- `:reg` - Show all registers
- `:reg a b c` - Show specific registers
- `"ayy` - Yank to register a
- `"ap` - Paste from register a
- `Ctrl+r a` - Paste register a in insert/command mode

## Macros

Macros allow recording and replaying sequences of commands.

**Recording:**
- `q{register}` - Start recording to register (a-z)
- ... perform operations ...
- `q` - Stop recording

**Playback:**
- `@{register}` - Execute macro from register
- `@@` - Repeat last executed macro
- `{n}@{register}` - Execute macro n times

**Examples:**
```
qa          " Start recording to register 'a'
^           " Go to start of line
i"          " Insert quote
<Esc>       " Exit insert mode
$           " Go to end of line
a"          " Append quote
<Esc>       " Exit insert mode
j           " Move down
q           " Stop recording

10@a        " Execute macro 10 times
```

**Editing Macros:**
- `"ap` - Paste macro from register a
- Edit the text
- `"ayy` - Yank back to register a

**Recursive Macros:**
```
qaqa        " Clear register a
qa          " Start recording
...
@a          " Call macro recursively
q           " Stop recording
```

## Buffers, Windows, and Tabs

### Buffers

Buffers are in-memory representations of files.

**Buffer Commands:**
- `:e filename` - Edit file in new buffer
- `:bn` - Next buffer
- `:bp` - Previous buffer
- `:b{n}` - Jump to buffer number n
- `:b filename` - Jump to buffer by name (supports tab-completion)
- `:bd` - Delete (close) current buffer
- `:buffers` or `:ls` - List all buffers
- `:ball` - Open all buffers in windows

**Buffer States:**
- `a` - Active (loaded and visible)
- `h` - Hidden (loaded but not visible)
- `%` - Current buffer
- `#` - Alternate buffer (toggle with `Ctrl+^`)
- `+` - Modified

### Windows

Windows are viewports into buffers.

**Splitting:**
- `:split` or `:sp` - Split horizontally
- `:vsplit` or `:vs` - Split vertically
- `:new` - New horizontal split with empty buffer
- `:vnew` - New vertical split with empty buffer
- `Ctrl+w s` - Split horizontally
- `Ctrl+w v` - Split vertically

**Navigation:**
- `Ctrl+w h/j/k/l` - Move to left/down/up/right window
- `Ctrl+w w` - Cycle through windows
- `Ctrl+w p` - Move to previous window
- `Ctrl+w t` - Move to top-left window
- `Ctrl+w b` - Move to bottom-right window

**Resizing:**
- `Ctrl+w =` - Equalize window sizes
- `Ctrl+w +` - Increase height
- `Ctrl+w -` - Decrease height
- `Ctrl+w >` - Increase width
- `Ctrl+w <` - Decrease width
- `Ctrl+w |` - Maximize width
- `Ctrl+w _` - Maximize height
- `:resize {n}` - Set height to n
- `:vertical resize {n}` - Set width to n

**Moving/Rotating:**
- `Ctrl+w r` - Rotate windows
- `Ctrl+w x` - Exchange windows
- `Ctrl+w H/J/K/L` - Move window to far left/bottom/top/right
- `Ctrl+w T` - Move window to new tab

**Closing:**
- `Ctrl+w q` or `:q` - Close current window
- `Ctrl+w o` or `:only` - Close all windows except current

### Tabs

Tabs are collections of windows.

**Tab Commands:**
- `:tabnew` - New tab
- `:tabe filename` - Edit file in new tab
- `:tabc` - Close current tab
- `:tabo` - Close all other tabs
- `gt` or `:tabn` - Next tab
- `gT` or `:tabp` - Previous tab
- `{n}gt` - Go to tab n
- `:tabs` - List all tabs
- `:tabm {n}` - Move tab to position n

## Command-line Mode

### File Operations

- `:e filename` - Edit file
- `:w` - Write (save) file
- `:w filename` - Save as filename
- `:w!` - Force write
- `:q` - Quit
- `:q!` - Quit without saving
- `:wq` or `:x` or `ZZ` - Write and quit
- `:qa` - Quit all windows
- `:wqa` - Write and quit all
- `:saveas filename` - Save as and continue editing new file

### Range Commands

Ranges specify lines for commands:

**Syntax:**
- `{start},{end}command`
- `.` - Current line
- `$` - Last line
- `%` - All lines (equivalent to `1,$`)
- `'<,'>` - Visual selection

**Examples:**
- `:10,20d` - Delete lines 10-20
- `:.,+5d` - Delete current line and next 5
- `:%y` - Yank all lines
- `:5,10s/foo/bar/g` - Substitute in lines 5-10
- `:.,$d` - Delete from current line to end

### External Commands

- `:!command` - Execute external command
- `:r !command` - Read command output into buffer
- `:.!command` - Filter current line through command
- `:%!command` - Filter entire file through command
- `:'<,'>!command` - Filter visual selection through command

**Examples:**
- `:!ls` - List directory
- `:r !date` - Insert current date
- `:%!sort` - Sort entire file
- `:%!python -m json.tool` - Format JSON
- `:'<,'>!sort -u` - Sort and unique selected lines

### Settings and Configuration

**View Settings:**
- `:set` - Show all non-default options
- `:set all` - Show all options
- `:set option?` - Query value of option
- `:set option` - Enable boolean option
- `:set nooption` - Disable boolean option
- `:set option=value` - Set value option

**Common Settings:**
- `:set number` or `:set nu` - Show line numbers
- `:set relativenumber` or `:set rnu` - Show relative line numbers
- `:set nonumber` or `:set nonu` - Hide line numbers
- `:set wrap` / `:set nowrap` - Enable/disable line wrapping
- `:set expandtab` / `:set noexpandtab` - Use spaces/tabs for indentation
- `:set tabstop=4` - Set tab width to 4
- `:set shiftwidth=4` - Set indent width to 4
- `:set autoindent` / `:set ai` - Enable auto-indent
- `:set smartindent` / `:set si` - Enable smart indent
- `:set hlsearch` / `:set hls` - Highlight search results
- `:set incsearch` / `:set is` - Incremental search
- `:set ignorecase` / `:set ic` - Ignore case in search
- `:set smartcase` / `:set scs` - Smart case (case-sensitive if uppercase used)

## Common Patterns and Workflows

### Changing Surrounding Characters

**Change quotes:**
- `ci"` → type new content → delete inside quotes and insert
- `ca"` → delete around quotes (including quotes) and insert

**Change brackets:**
- `ci(` → change inside parentheses
- `ci{` → change inside braces
- `ci[` → change inside brackets
- `cit` → change inside HTML/XML tags

**Delete surrounding:**
- `di"` → delete inside quotes
- `da"` → delete around quotes (including quotes)
- `di(` → delete inside parentheses

### Duplicating Lines/Blocks

- `yyp` - Duplicate current line
- `Yp` - Duplicate current line (alternative)
- `V{motion}y` → select lines → `P` - Duplicate block

### Swapping Characters/Words/Lines

- `xp` - Swap two characters
- `ddp` - Swap current line with next
- `dawwP` - Swap two words

### Comment/Uncomment Lines

**Visual block mode method:**
```
Ctrl+v          " Enter visual block mode
{motion}        " Select lines
I#<Esc>         " Insert # at beginning
```

**To uncomment:**
```
Ctrl+v          " Enter visual block mode
{motion}        " Select comment characters
x               " Delete
```

### Sorting Lines

- `:sort` - Sort lines
- `:sort!` - Reverse sort
- `:sort u` - Sort and remove duplicates
- `:'<,'>sort` - Sort visual selection

### Working with Multiple Files

**Split windows and compare:**
```
:vs other_file.txt      " Vertical split
:diffthis               " In both windows
:diffoff                " Turn off diff
```

**Quick file switching:**
- `Ctrl+^` - Toggle between current and alternate buffer
- `:b#` - Jump to alternate buffer

### Refactoring Patterns

**Rename variable (simple):**
```
*               " Search for word under cursor
cgn             " Change next occurrence
{new_name}
<Esc>
.               " Repeat for next occurrence
n.n.n.          " Continue for more occurrences
```

**Rename variable (all in file):**
```
:%s/\<old_name\>/new_name/gc
```

**Extract to variable:**
```
vi"             " Select inside quotes (or other text object)
y               " Yank
O               " Open line above
const name =    " Type variable name
<Esc>p          " Paste
```

### Repeating Operations

- `.` - Repeat last change
- `@:` - Repeat last command-line command
- `@@` - Repeat last macro
- `&` - Repeat last substitute

### Efficient Editing Patterns

**Delete around cursor:**
- `daw` - Delete a word (including space)
- `das` - Delete a sentence
- `dap` - Delete a paragraph
- `da"` - Delete around quotes
- `da(` - Delete around parentheses

**Change around cursor:**
- `caw` - Change a word
- `cas` - Change a sentence
- `cit` - Change inside tag
- `ci"` - Change inside quotes

**Delete to character:**
- `dt{char}` - Delete till character
- `df{char}` - Delete up to and including character
- `dT{char}` - Delete backwards till character
- `dF{char}` - Delete backwards including character

## Configuration and Customization

### The .vimrc File

The `.vimrc` file (located at `~/.vimrc` on Unix or `$HOME/_vimrc` on Windows) contains Vim configuration.

**Basic .vimrc Example:**
```vim
" Enable syntax highlighting
syntax on

" Show line numbers
set number
set relativenumber

" Indentation settings
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set smartindent

" Search settings
set hlsearch
set incsearch
set ignorecase
set smartcase

" UI enhancements
set showcmd
set showmatch
set ruler
set wildmenu
set cursorline

" Performance
set lazyredraw

" Enable mouse support
set mouse=a

" Better split behavior
set splitbelow
set splitright

" Persistent undo
set undofile
set undodir=~/.vim/undo

" Clipboard
set clipboard=unnamedplus

" Key mappings
let mapleader = " "
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>h :noh<CR>

" Quick window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Move lines up/down
nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv
```

### Key Mappings

**Mapping Modes:**
- `nnoremap` - Normal mode
- `inoremap` - Insert mode
- `vnoremap` - Visual mode
- `cnoremap` - Command-line mode
- `tnoremap` - Terminal mode

**Mapping Syntax:**
```vim
{mode}map {lhs} {rhs}
```

**Examples:**
```vim
" Map jk to escape
inoremap jk <Esc>

" Save with Ctrl+S
nnoremap <C-s> :w<CR>
inoremap <C-s> <Esc>:w<CR>a

" Toggle line numbers
nnoremap <leader>n :set number!<CR>

" Open vimrc
nnoremap <leader>ev :e $MYVIMRC<CR>

" Source vimrc
nnoremap <leader>sv :source $MYVIMRC<CR>
```

**Special Keys in Mappings:**
- `<CR>` - Enter
- `<Esc>` - Escape
- `<Space>` - Space
- `<Tab>` - Tab
- `<Leader>` - Leader key (default `\`)
- `<C-x>` - Ctrl+x
- `<A-x>` or `<M-x>` - Alt+x
- `<S-x>` - Shift+x
- `<F1>`-`<F12>` - Function keys

### Plugin Management

**Popular Plugin Managers:**
- **vim-plug** - Minimalist plugin manager
- **Vundle** - Classic plugin manager
- **Pathogen** - Simple runtime path manager
- **dein.vim** - Fast plugin manager

**vim-plug Example:**
```vim
" Install vim-plug if not already installed
if empty(glob('~/.vim/autoload/plug.vim'))
  silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
endif

" Plugin section
call plug#begin('~/.vim/plugged')

" File explorer
Plug 'preservim/nerdtree'

" Fuzzy finder
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" Status line
Plug 'vim-airline/vim-airline'

" Git integration
Plug 'tpope/vim-fugitive'

" Surround text objects
Plug 'tpope/vim-surround'

" Auto pairs
Plug 'jiangmiao/auto-pairs'

" Commentary
Plug 'tpope/vim-commentary'

" Color scheme
Plug 'morhetz/gruvbox'

call plug#end()

" Plugin configuration
colorscheme gruvbox
set background=dark

" NERDTree
nnoremap <leader>e :NERDTreeToggle<CR>

" FZF
nnoremap <leader>f :Files<CR>
nnoremap <leader>b :Buffers<CR>
nnoremap <leader>/ :Rg<CR>
```

**Essential Plugins:**
- **NERDTree** - File system explorer
- **fzf.vim** - Fuzzy file finding
- **vim-surround** - Manipulate surrounding characters
- **vim-commentary** - Easy commenting
- **vim-fugitive** - Git wrapper
- **coc.nvim** - IntelliSense/LSP support
- **vim-airline** - Status line
- **vim-easymotion** - Enhanced motion
- **auto-pairs** - Auto close brackets/quotes

## Tips, Tricks & Best Practices

### Efficiency Tips

1. **Stay in Normal mode** - Normal mode is home; insert mode is a temporary visit
2. **Think in operators + motions** - `ciw`, `dap`, `yi"` are more powerful than visual selection
3. **Use text objects** - `ci"`, `da{`, `yap` are game-changers
4. **Embrace the dot command** - Make changes repeatable with `.`
5. **Learn one new thing per week** - Vim has a steep learning curve; pace yourself
6. **Use relative line numbers** - `set relativenumber` makes jumping with `{count}j/k` easier
7. **Master search** - `*`, `#`, `/`, `?` navigation is faster than scrolling
8. **Use marks for long jumps** - `mA` to set, `` `A`` to jump back
9. **Keep `.vimrc` organized** - Comment your config for future reference
10. **Practice regularly** - Muscle memory is key

### Avoiding Anti-patterns

**Don't:**
- ❌ Hold down `j/k` to scroll through file (use `/{pattern}`, `{n}j`, `}`, `Ctrl+d`)
- ❌ Use arrow keys (use `h/j/k/l`)
- ❌ Use mouse for text selection (use visual mode or text objects)
- ❌ Use visual mode for everything (operators + motions are often better)
- ❌ Exit insert mode just to move one character (use `Ctrl+o {motion}`)
- ❌ Manually delete each character with `x` (use `dw`, `diw`, `D`, etc.)
- ❌ Repeat the same edit manually (use `.`, macros, or `:g//`)
- ❌ Navigate without search (searching is faster than scrolling)

**Do:**
- ✅ Use `*` to search word under cursor
- ✅ Use `ci"` instead of selecting and deleting
- ✅ Use `>>` for indenting instead of inserting spaces
- ✅ Use `d$` instead of holding `x`
- ✅ Use `/pattern` to jump instead of scrolling
- ✅ Record macros for repetitive tasks
- ✅ Use `cgn` pattern for incremental replacements
- ✅ Learn regex for powerful search/replace

### Muscle Memory Builders

**Practice these daily:**
1. Navigation: `w`, `b`, `e`, `{`, `}`, `%`, `*`, `#`
2. Text objects: `ciw`, `di"`, `da{`, `yap`, `vi)`
3. Delete/Change: `dd`, `cc`, `D`, `C`, `dt{char}`, `df{char}`
4. Combinations: `ci"`, `da)`, `yi{`, `va}`, `ci]`
5. Repeat operations: `.`, `@@`, `&`, `@:`

### Quick Reference - Most Useful Commands

**Top 20 commands to master first:**
1. `i` / `a` - Insert mode before/after cursor
2. `Esc` - Return to normal mode
3. `h/j/k/l` - Navigation
4. `w` / `b` - Word forward/backward
5. `0` / `$` - Line start/end
6. `dd` - Delete line
7. `yy` - Yank (copy) line
8. `p` - Paste
9. `u` - Undo
10. `Ctrl+r` - Redo
11. `/pattern` - Search
12. `n` / `N` - Next/previous search result
13. `ciw` - Change inner word
14. `di"` - Delete inside quotes
15. `.` - Repeat last change
16. `:w` - Save
17. `:q` - Quit
18. `v` - Visual mode
19. `gg` / `G` - File start/end
20. `*` - Search word under cursor

### Vim Cheat Sheet

**Modes:**
- `Esc` → Normal | `i` → Insert | `v` → Visual | `:` → Command

**Navigation:**
- `hjkl` → ←↓↑→ | `w/b` → word forward/back | `0/$` → line start/end
- `gg/G` → file start/end | `{/}` → paragraph | `%` → matching bracket

**Editing:**
- `x` → delete char | `dd` → delete line | `yy` → yank line | `p` → paste
- `u` → undo | `Ctrl+r` → redo | `.` → repeat | `J` → join lines

**Operators + Motions:**
- `d{motion}` → delete | `c{motion}` → change | `y{motion}` → yank
- `>{motion}` → indent | `gu{motion}` → lowercase

**Text Objects:**
- `iw/aw` → inner/around word | `i"/a"` → inner/around quotes
- `i(/a(` → inner/around () | `i{/a{` → inner/around {}
- `it/at` → inner/around tag | `ip/ap` → inner/around paragraph

**Search & Replace:**
- `/pattern` → search | `n/N` → next/prev | `*/#` → word under cursor
- `:%s/old/new/g` → replace all | `:%s/old/new/gc` → replace with confirm

**Files & Windows:**
- `:w` → save | `:q` → quit | `:wq` → save & quit | `:e file` → edit file
- `:sp/:vs` → split | `Ctrl+w hjkl` → navigate windows | `:tabnew` → new tab

## Conclusion

Vim is more than a text editor—it's a highly efficient, composable language for manipulating text. Its modal nature and powerful command combinations enable editing at the speed of thought.

**Key Takeaways:**

1. **Modal editing** separates navigation from insertion, making each mode optimized for its purpose.

2. **Composability** through the operator + motion/text-object grammar creates hundreds of commands from a dozen primitives.

3. **Text objects** (`iw`, `i"`, `a{`, `ap`) are one of Vim's most powerful features—master them early.

4. **Repeatability** via `.`, macros, and `:g//` commands makes repetitive tasks trivial.

5. **Efficiency** comes from keeping hands on home row and thinking in terms of semantic units (words, sentences, paragraphs) rather than characters.

6. **Customization** through `.vimrc` and plugins allows tailoring Vim to your workflow.

7. **Learning curve** is steep but worthwhile—invest time in deliberate practice and muscle memory.

**Learning Path:**

1. **Week 1**: Master modes, basic navigation (`hjkl`, `w/b`, `0/$`), insert/append (`i/a/o`)
2. **Week 2**: Operators (`d/c/y`) + basic motions, undo/redo, basic search
3. **Week 3**: Text objects (`iw`, `i"`, `i{`), dot command, basic visual mode
4. **Week 4**: Advanced navigation (marks, jumps, `f/t`), search/replace
5. **Month 2**: Macros, registers, windows/buffers, plugins
6. **Month 3+**: Advanced patterns, custom configuration, language-specific setups

**Resources:**
- `:help user-manual` - Built-in comprehensive manual
- `:Tutor` or `vimtutor` - Interactive tutorial
- Practice regularly - Muscle memory is essential
- Avoid using Vim for everything immediately—gradually increase usage

Vim proficiency is a journey, not a destination. Each technique you master compounds with others, making you exponentially more efficient. Happy Vimming!
