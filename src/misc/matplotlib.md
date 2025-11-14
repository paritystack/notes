# Matplotlib: Complete Guide for Data Visualization

Matplotlib is the foundational plotting library for Python, providing publication-quality visualizations and serving as the basis for many other plotting libraries (Seaborn, Pandas plotting, etc.).

## Table of Contents
- [Architecture & Core Concepts](#architecture--core-concepts)
- [Basic Plotting](#basic-plotting)
- [Figure and Axes Management](#figure-and-axes-management)
- [Customization Deep Dive](#customization-deep-dive)
- [Advanced Plot Types](#advanced-plot-types)
- [Styling and Themes](#styling-and-themes)
- [ML/Data Science Visualizations](#mldata-science-visualizations)
- [Working with Images](#working-with-images)
- [Animations](#animations)
- [Integration Patterns](#integration-patterns)
- [Performance & Best Practices](#performance--best-practices)
- [Common Patterns & Recipes](#common-patterns--recipes)

---

## Architecture & Core Concepts

### The Matplotlib Hierarchy

Matplotlib has a hierarchical structure that's essential to understand:

```
Figure (entire window)
  └── Axes (plot area, NOT axis!)
        ├── Axis (x-axis, y-axis)
        ├── Spines (plot boundaries)
        ├── Artists (everything you see)
        └── Legend, Title, Labels
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Understanding the hierarchy
fig = plt.figure(figsize=(10, 6))  # Figure: the whole window
ax = fig.add_subplot(111)           # Axes: a plot area

# Everything drawn is an "Artist"
line, = ax.plot([1, 2, 3], [1, 4, 2])  # Line2D artist
text = ax.text(2, 3, 'Point')          # Text artist
```

### Two Interfaces: pyplot vs Object-Oriented

```python
# PYPLOT INTERFACE (MATLAB-style, stateful)
plt.plot([1, 2, 3], [1, 4, 2])
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# OBJECT-ORIENTED INTERFACE (Recommended for complex plots)
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
plt.show()
```

**When to use which:**
- **pyplot**: Quick exploratory plots, simple scripts
- **OO interface**: Complex figures, multiple subplots, functions that create plots, production code

### Key Design Principle

```python
# Everything in matplotlib is customizable
# General pattern:
fig, ax = plt.subplots()

# Plot data
artist = ax.plot(x, y)

# Customize
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Title')

# Display or save
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Basic Plotting

### Line Plots

```python
import numpy as np
import matplotlib.pyplot as plt

# Single line
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('sin(X)')
ax.set_title('Sine Wave')
plt.show()

# Multiple lines
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots()
ax.plot(x, y1, label='sin(x)')
ax.plot(x, y2, label='cos(x)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# Customized line styles
fig, ax = plt.subplots()
ax.plot(x, y1, 'r-', linewidth=2, label='solid')
ax.plot(x, y2, 'b--', linewidth=2, label='dashed')
ax.plot(x, y1 + 0.5, 'g-.', linewidth=2, label='dash-dot')
ax.plot(x, y2 + 0.5, 'k:', linewidth=2, label='dotted')
ax.legend()
plt.show()
```

### Scatter Plots

```python
# Basic scatter
x = np.random.randn(100)
y = np.random.randn(100)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# Customized scatter with size and color
sizes = np.random.rand(100) * 100
colors = np.random.rand(100)

fig, ax = plt.subplots()
scatter = ax.scatter(x, y, s=sizes, c=colors,
                     cmap='viridis', alpha=0.6,
                     edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='Color Value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# Multiple scatter series
x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)
x2 = np.random.normal(3, 1, 100)
y2 = np.random.normal(3, 1, 100)

fig, ax = plt.subplots()
ax.scatter(x1, y1, label='Class 1', alpha=0.6)
ax.scatter(x2, y2, label='Class 2', alpha=0.6)
ax.legend()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.show()
```

### Bar Charts

```python
# Vertical bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 55, 45]

fig, ax = plt.subplots()
bars = ax.bar(categories, values, color='steelblue',
              edgecolor='black', linewidth=1.2)
ax.set_ylabel('Values')
ax.set_title('Bar Chart')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}', ha='center', va='bottom')
plt.show()

# Horizontal bar chart
fig, ax = plt.subplots()
ax.barh(categories, values, color='coral')
ax.set_xlabel('Values')
plt.show()

# Grouped bar chart
x = np.arange(len(categories))
values1 = [25, 40, 30, 55, 45]
values2 = [30, 35, 45, 40, 50]
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values1, width, label='Group 1')
bars2 = ax.bar(x + width/2, values2, width, label='Group 2')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.show()

# Stacked bar chart
fig, ax = plt.subplots()
ax.bar(categories, values1, label='Part 1')
ax.bar(categories, values2, bottom=values1, label='Part 2')
ax.legend()
plt.show()
```

### Histograms

```python
# Basic histogram
data = np.random.randn(1000)

fig, ax = plt.subplots()
ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram')
plt.show()

# Multiple histograms
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

fig, ax = plt.subplots()
ax.hist(data1, bins=30, alpha=0.5, label='Distribution 1')
ax.hist(data2, bins=30, alpha=0.5, label='Distribution 2')
ax.legend()
plt.show()

# Normalized histogram (density)
fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True, alpha=0.7,
        edgecolor='black', label='Data')

# Overlay theoretical distribution
mu, sigma = 0, 1
x = np.linspace(data.min(), data.max(), 100)
ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp(-0.5 * ((x - mu)/sigma)**2),
        'r-', linewidth=2, label='Theoretical')
ax.legend()
plt.show()

# 2D histogram (hexbin)
x = np.random.randn(10000)
y = np.random.randn(10000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist2d(x, y, bins=50, cmap='Blues')
ax1.set_title('2D Histogram')

hexbin = ax2.hexbin(x, y, gridsize=30, cmap='Reds')
ax2.set_title('Hexbin')
plt.colorbar(hexbin, ax=ax2)
plt.show()
```

### Pie Charts

```python
# Basic pie chart
sizes = [25, 35, 20, 20]
labels = ['A', 'B', 'C', 'D']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       startangle=90)
ax.axis('equal')  # Equal aspect ratio
plt.show()

# Exploded pie chart with custom colors
explode = (0.1, 0, 0, 0)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=labels,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    explode=explode,
                                    colors=colors,
                                    shadow=True)
# Customize text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
plt.show()

# Donut chart
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       wedgeprops=dict(width=0.5))  # Creates donut
ax.axis('equal')
plt.show()
```

---

## Figure and Axes Management

### Creating Figures and Subplots

```python
# Method 1: plt.subplots() (Recommended)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3], [1, 4, 2])

# Method 2: Multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot([1, 2, 3])
ax2.plot([3, 2, 1])

# Method 3: Grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.plot(np.random.randn(10))
    ax.set_title(f'Subplot {i+1}')

# Method 4: Figure first, then add axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)  # 1 row, 1 col, index 1
```

### Complex Layouts with GridSpec

```python
import matplotlib.gridspec as gridspec

# GridSpec for flexible layouts
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Span multiple cells
ax1 = fig.add_subplot(gs[0, :])   # First row, all columns
ax2 = fig.add_subplot(gs[1, :-1]) # Second row, first 2 columns
ax3 = fig.add_subplot(gs[1:, -1]) # Last 2 rows, last column
ax4 = fig.add_subplot(gs[-1, 0])  # Last row, first column
ax5 = fig.add_subplot(gs[-1, 1])  # Last row, second column

ax1.plot(np.random.randn(100))
ax1.set_title('Wide Top Panel')

ax2.plot(np.random.randn(100))
ax2.set_title('Middle Left')

ax3.plot(np.random.randn(100))
ax3.set_title('Right Panel')

ax4.plot(np.random.randn(100))
ax5.plot(np.random.randn(100))

plt.tight_layout()
plt.show()

# Unequal spacing
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[2, 1],
                       height_ratios=[1, 2],
                       hspace=0.3, wspace=0.3)

for i in range(4):
    ax = fig.add_subplot(gs[i])
    ax.plot(np.random.randn(100))
    ax.set_title(f'Subplot {i+1}')
plt.show()
```

### Subplot Sharing and Linking

```python
# Shared axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_ylabel('sin(x)')
ax2.plot(x, np.cos(x))
ax2.set_ylabel('cos(x)')
ax2.set_xlabel('x')
plt.show()

# Grid with shared axes
fig, axes = plt.subplots(2, 2, sharex='col', sharey='row',
                         figsize=(10, 8))
for i in range(2):
    for j in range(2):
        axes[i, j].plot(np.random.randn(100).cumsum())
plt.show()
```

### Inset Axes and Zooming

```python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(10, 6))

# Main plot
x = np.linspace(0, 10, 1000)
y = np.sin(x) * np.exp(-x/10)
ax.plot(x, y)

# Inset axes
axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
axins.plot(x, y)
axins.set_xlim(2, 3)
axins.set_ylim(0.3, 0.5)
axins.set_xticks([])
axins.set_yticks([])

# Mark the inset region
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()
```

### Twin Axes (Two Y-axes)

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x/5)

# First y-axis
color = 'tab:blue'
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Second y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('exp(x/5)', color=color)
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
```

---

## Customization Deep Dive

### Colors

```python
# Named colors
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

# Hex colors
colors = ['#FF5733', '#33FF57', '#3357FF']

# RGB tuples (0-1)
colors = [(0.8, 0.2, 0.1), (0.1, 0.8, 0.2)]

# RGBA with transparency
colors = [(0.8, 0.2, 0.1, 0.5)]

# Colormaps
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()
for i in range(10):
    color = plt.cm.viridis(i / 10)  # Get color from colormap
    ax.plot(x, np.sin(x + i/5), color=color)
plt.show()

# Popular colormaps
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',  # Perceptually uniform
         'coolwarm', 'RdYlBu', 'RdYlGn',  # Diverging
         'Greys', 'Blues', 'Reds',  # Sequential
         'tab10', 'tab20', 'Set1']  # Qualitative

# Custom colormap
from matplotlib.colors import LinearSegmentedColormap
colors_list = ['blue', 'white', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)

# Using colormap
data = np.random.rand(10, 10)
fig, ax = plt.subplots()
im = ax.imshow(data, cmap=cmap)
plt.colorbar(im, ax=ax)
plt.show()
```

### Markers and Line Styles

```python
# Markers
markers = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H',
           '+', 'x', 'D', 'd', '|', '_']

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(markers))
for i, marker in enumerate(markers):
    ax.plot(i, i, marker=marker, markersize=10, label=marker)
ax.legend(ncol=6)
plt.show()

# Line styles
linestyles = ['-', '--', '-.', ':']
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
for i, ls in enumerate(linestyles):
    ax.plot(x, np.sin(x) + i, linestyle=ls, linewidth=2,
            label=f"'{ls}'")
ax.legend()
plt.show()

# Combined format string
# Format: '[marker][line][color]'
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2], 'ro-')   # Red circles with solid line
ax.plot([1, 2, 3], [2, 3, 1], 'bs--')  # Blue squares with dashed line
ax.plot([1, 2, 3], [0.5, 2.5, 1.5], 'g^:')  # Green triangles with dotted line
plt.show()

# Detailed customization
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2],
        marker='o',
        markersize=10,
        markerfacecolor='red',
        markeredgecolor='black',
        markeredgewidth=2,
        linestyle='--',
        linewidth=2,
        color='blue',
        alpha=0.7)
plt.show()
```

### Labels, Titles, and Legends

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# Title with customization
ax.set_title('Trigonometric Functions',
             fontsize=16, fontweight='bold',
             pad=20)

# Axis labels
ax.set_xlabel('X Axis', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Axis', fontsize=12, fontweight='bold')

# Legend customization
ax.legend(loc='upper right',           # Location
          frameon=True,                # Frame
          fancybox=True,              # Rounded corners
          shadow=True,                # Shadow
          ncol=2,                     # Number of columns
          fontsize=10,
          title='Functions',
          title_fontsize=12)

# Alternative legend locations
# 'best', 'upper right', 'upper left', 'lower left', 'lower right',
# 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

# Legend outside plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Multiple legends
fig, ax = plt.subplots()
line1, = ax.plot([1, 2, 3], [1, 2, 3], 'r-', label='Red')
line2, = ax.plot([1, 2, 3], [3, 2, 1], 'b-', label='Blue')

# First legend
legend1 = ax.legend(handles=[line1], loc='upper left')
ax.add_artist(legend1)  # Add first legend back

# Second legend
ax.legend(handles=[line2], loc='upper right')
plt.show()
```

### Tick Customization

```python
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# Tick positions
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

# Tick labels
ax.set_xticklabels(['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten'])

# Tick parameters
ax.tick_params(axis='x',
               labelsize=10,
               labelrotation=45,
               labelcolor='blue',
               length=6,
               width=2,
               direction='in')

# Minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', length=3)

# Custom tick formatter
from matplotlib.ticker import FuncFormatter

def currency(x, pos):
    return f'${x:.2f}'

ax.yaxis.set_major_formatter(FuncFormatter(currency))
plt.show()

# Log scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.logspace(0, 3, 100)
y = x ** 2

ax1.plot(x, y)
ax1.set_title('Linear Scale')

ax2.plot(x, y)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Log Scale')
ax2.grid(True, which='both', alpha=0.3)
plt.show()
```

### Spines and Frames

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(-5, 5, 100)
y = x ** 2

# Default
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Default')

# Remove top and right spines
axes[0, 1].plot(x, y)
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)
axes[0, 1].set_title('Clean')

# Move spines to zero
axes[1, 0].plot(x, y)
axes[1, 0].spines['left'].set_position('zero')
axes[1, 0].spines['bottom'].set_position('zero')
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)
axes[1, 0].set_title('Centered')

# No spines (floating)
axes[1, 1].plot(x, y)
for spine in axes[1, 1].spines.values():
    spine.set_visible(False)
axes[1, 1].set_title('No Spines')

plt.tight_layout()
plt.show()
```

### Annotations and Text

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

# Simple text
ax.text(5, 0.5, 'Peak Region', fontsize=12)

# Text with box
bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(8, -0.5, 'Trough', fontsize=12, bbox=bbox_props)

# Annotation with arrow
ax.annotate('Maximum',
            xy=(np.pi/2, 1),      # Point to annotate
            xytext=(2, 0.5),      # Text position
            fontsize=12,
            arrowprops=dict(facecolor='red',
                          shrink=0.05,
                          width=2,
                          headwidth=8))

# Multiple annotation styles
ax.annotate('Fancy Arrow',
            xy=(3*np.pi/2, -1),
            xytext=(7, -0.3),
            arrowprops=dict(arrowstyle='->',
                          connectionstyle='arc3,rad=0.3',
                          color='blue',
                          lw=2))

# Mathematical text (LaTeX)
ax.text(1, -0.8, r'$y = \sin(x)$', fontsize=16)
ax.text(5, -0.8, r'$\int_0^{\pi} \sin(x)dx = 2$', fontsize=14)

plt.show()

# Arrow styles
arrow_styles = ['-', '->', '-[', '|-|', '-|>', '<-', '<->',
                'fancy', 'simple', 'wedge']
```

### Adding Shapes

```python
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse, FancyBboxPatch
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots(figsize=(10, 8))

# Circle
circle = Circle((2, 2), 0.5, color='red', alpha=0.5)
ax.add_patch(circle)

# Rectangle
rect = Rectangle((4, 1), 1, 2, color='blue', alpha=0.5)
ax.add_patch(rect)

# Ellipse
ellipse = Ellipse((7, 2), 1, 2, angle=30, color='green', alpha=0.5)
ax.add_patch(ellipse)

# Polygon
triangle = Polygon([[1, 4], [2, 6], [3, 4]], color='purple', alpha=0.5)
ax.add_patch(triangle)

# Fancy box
fancy = FancyBboxPatch((5, 4), 2, 1.5,
                       boxstyle="round,pad=0.1",
                       edgecolor='orange',
                       facecolor='yellow',
                       linewidth=2,
                       alpha=0.5)
ax.add_patch(fancy)

ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
plt.show()
```

---

## Advanced Plot Types

### 3D Plots

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D line plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')

t = np.linspace(0, 10, 1000)
x = np.sin(t)
y = np.cos(t)
z = t

ax.plot(x, y, z, linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Line Plot')

# 3D scatter
ax = fig.add_subplot(122, projection='3d')
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
colors = np.random.rand(100)

scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50)
ax.set_title('3D Scatter')
plt.colorbar(scatter, ax=ax)
plt.show()

# 3D surface
fig = plt.figure(figsize=(12, 5))

# Surface plot
ax = fig.add_subplot(121, projection='3d')
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax.set_title('Surface Plot')
plt.colorbar(surf, ax=ax, shrink=0.5)

# Wireframe
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5)
ax.set_title('Wireframe Plot')
plt.show()

# Contour3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='viridis')
ax.set_title('3D Contour')
plt.show()
```

### Contour Plots

```python
# 2D contour
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Filled contour
contourf = ax1.contourf(X, Y, Z, levels=20, cmap='RdYlBu')
ax1.set_title('Filled Contour')
plt.colorbar(contourf, ax=ax1)

# Line contour
contour = ax2.contour(X, Y, Z, levels=10, colors='black')
ax2.clabel(contour, inline=True, fontsize=8)  # Label contours
ax2.set_title('Line Contour')

# Combined
ax3.contourf(X, Y, Z, levels=20, cmap='RdYlBu', alpha=0.7)
contour = ax3.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
ax3.clabel(contour, inline=True, fontsize=8)
ax3.set_title('Combined')

plt.tight_layout()
plt.show()
```

### Heatmaps and imshow

```python
# Heatmap
data = np.random.rand(10, 12)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value', rotation=270, labelpad=20)

# Ticks and labels
ax.set_xticks(np.arange(12))
ax.set_yticks(np.arange(10))
ax.set_xticklabels([f'Col {i}' for i in range(12)])
ax.set_yticklabels([f'Row {i}' for i in range(10)])

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Add values in cells
for i in range(10):
    for j in range(12):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                      ha='center', va='center', color='black')

ax.set_title('Heatmap with Values')
plt.tight_layout()
plt.show()
```

### Error Bars

```python
x = np.linspace(0, 10, 20)
y = np.sin(x)
yerr = 0.1 + 0.05 * np.random.rand(len(x))
xerr = 0.1 + 0.05 * np.random.rand(len(x))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Y error bars only
ax1.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5,
             capthick=2, label='Data')
ax1.set_title('Y Error Bars')
ax1.legend()

# X and Y error bars
ax2.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='s-',
             capsize=5, alpha=0.7)
ax2.set_title('X and Y Error Bars')

# Shaded error region
ax3.plot(x, y, 'o-', label='Mean')
ax3.fill_between(x, y - yerr, y + yerr, alpha=0.3, label='±1 std')
ax3.set_title('Shaded Error Region')
ax3.legend()

plt.tight_layout()
plt.show()
```

### Box Plots and Violin Plots

```python
# Generate sample data
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
bp = ax1.boxplot(data,
                 labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                 notch=True,  # Notched box
                 patch_artist=True)  # Fill with color

# Customize colors
colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_title('Box Plot')
ax1.set_ylabel('Values')

# Violin plot
parts = ax2.violinplot(data, showmeans=True, showmedians=True)
ax2.set_title('Violin Plot')
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])

plt.tight_layout()
plt.show()

# Horizontal box plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(data, vert=False, labels=['A', 'B', 'C', 'D'])
ax.set_xlabel('Values')
plt.show()
```

### Stream Plots and Quiver Plots

```python
# Vector field (quiver plot)
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
U = -Y  # x-component
V = X   # y-component

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Quiver plot
ax1.quiver(X, Y, U, V, alpha=0.8)
ax1.set_title('Quiver Plot (Vector Field)')
ax1.set_aspect('equal')

# Stream plot
ax2.streamplot(X, Y, U, V, density=1.5, color=np.sqrt(U**2 + V**2),
               cmap='viridis', linewidth=1)
ax2.set_title('Stream Plot')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
```

### Polar Plots

```python
# Polar line plot
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + np.sin(4*theta)

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'),
                                figsize=(12, 5))

ax1.plot(theta, r)
ax1.set_title('Polar Line Plot')

# Polar scatter with colors
theta2 = np.random.uniform(0, 2*np.pi, 100)
r2 = np.random.uniform(0, 2, 100)
colors = theta2

ax2.scatter(theta2, r2, c=colors, cmap='hsv', alpha=0.75)
ax2.set_title('Polar Scatter')

plt.show()

# Polar bar (rose diagram)
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
radii = np.random.rand(8) * 10
width = 2*np.pi / 8

bars = ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.7)

# Color bars by height
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10))

plt.show()
```

---

## Styling and Themes

### Built-in Styles

```python
# See available styles
print(plt.style.available)

# Use a style
plt.style.use('seaborn-v0_8-darkgrid')
# Or: 'ggplot', 'fivethirtyeight', 'bmh', 'dark_background', etc.

# Example with different styles
styles = ['default', 'seaborn-v0_8-darkgrid', 'ggplot', 'fivethirtyeight']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
x = np.linspace(0, 10, 100)

for ax, style in zip(axes.flat, styles):
    with plt.style.context(style):
        ax.plot(x, np.sin(x), label='sin(x)')
        ax.plot(x, np.cos(x), label='cos(x)')
        ax.set_title(style)
        ax.legend()

plt.tight_layout()
plt.show()
```

### rcParams Configuration

```python
import matplotlib as mpl

# View current settings
print(mpl.rcParams['font.size'])

# Temporary changes
with mpl.rc_context({'font.size': 14, 'lines.linewidth': 2}):
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.show()

# Global changes (persists for session)
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Reset to defaults
mpl.rcParams.update(mpl.rcParamsDefault)

# Common rcParams for publications
pub_params = {
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1,
    'lines.linewidth': 1.5,
}
mpl.rcParams.update(pub_params)
```

### Custom Style Sheets

```python
# Create custom style file: ~/.matplotlib/stylelib/mystyle.mplstyle
"""
# mystyle.mplstyle
figure.figsize: 10, 6
figure.dpi: 100

axes.grid: True
axes.grid.axis: both
grid.alpha: 0.3
grid.linestyle: --

axes.spines.top: False
axes.spines.right: False

font.size: 12
axes.labelsize: 14
axes.titlesize: 16

lines.linewidth: 2
lines.markersize: 8

legend.frameon: False
legend.loc: best
"""

# Use custom style
# plt.style.use('mystyle')

# Or use directly with context
custom_style = {
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

with plt.style.context(custom_style):
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.show()
```

---

## ML/Data Science Visualizations

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False,
                         cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    return ax

# Example usage
y_true = np.random.randint(0, 3, 100)
y_pred = np.random.randint(0, 3, 100)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=['Class A', 'Class B', 'Class C'])
plt.show()
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores, n_classes):
    """
    Plot ROC curves for multi-class classification
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, color in enumerate(colors):
        # Binary indicators for class i
        y_true_binary = (y_true == i).astype(int)
        y_score_class = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_class)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'Class {i} (AUC = {roc_auc:.2f})')

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    return ax

# Example
n_samples, n_classes = 1000, 3
y_true = np.random.randint(0, n_classes, n_samples)
y_scores = np.random.rand(n_samples, n_classes)
y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)  # Normalize

plot_roc_curve(y_true, y_scores, n_classes)
plt.show()
```

### Learning Curves

```python
def plot_learning_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """
    Plot training and validation loss/accuracy curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss',
                 markersize=4)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss',
                 markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curves (if provided)
    if train_accs is not None and val_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-o', label='Training Accuracy',
                     markersize=4)
        axes[1].plot(epochs, val_accs, 'r-s', label='Validation Accuracy',
                     markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    return fig

# Example
epochs = 50
train_losses = 2.0 * np.exp(-np.arange(epochs) / 10) + 0.1 * np.random.rand(epochs)
val_losses = 2.0 * np.exp(-np.arange(epochs) / 10) + 0.2 * np.random.rand(epochs) + 0.1
train_accs = 1 - np.exp(-np.arange(epochs) / 10) * 0.9
val_accs = 1 - np.exp(-np.arange(epochs) / 10) * 0.9 - 0.05

plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
plt.show()
```

### Feature Importance

```python
def plot_feature_importance(feature_names, importances, top_n=20):
    """
    Plot feature importance bar chart
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart
    y_pos = np.arange(len(sorted_names))
    colors = plt.cm.viridis(sorted_importances / sorted_importances.max())

    bars = ax.barh(y_pos, sorted_importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val, i, f' {val:.3f}', va='center')

    plt.tight_layout()
    return fig

# Example
n_features = 50
feature_names = [f'Feature_{i}' for i in range(n_features)]
importances = np.random.exponential(0.1, n_features)

plot_feature_importance(feature_names, importances, top_n=15)
plt.show()
```

### Decision Boundaries

```python
def plot_decision_boundary(X, y, model, resolution=0.02):
    """
    Plot decision boundary for 2D classification
    """
    # Setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu

    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict on grid
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot filled contour
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.contour(xx1, xx2, Z, colors='black', linewidths=0.5, alpha=0.5)

    # Plot data points
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                  alpha=0.8, c=[colors[idx]], marker=markers[idx],
                  s=100, edgecolor='black', label=f'Class {cl}')

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Decision Boundary', fontsize=14)
    ax.legend()

    return fig

# Example (requires a model with predict method)
# from sklearn.svm import SVC
# X = np.random.randn(200, 2)
# y = (X[:, 0] + X[:, 1] > 0).astype(int)
# model = SVC(kernel='rbf').fit(X, y)
# plot_decision_boundary(X, y, model)
```

### Attention Heatmap

```python
def plot_attention_heatmap(attention_matrix, x_labels=None, y_labels=None):
    """
    Plot attention weights as heatmap
    Useful for visualizing transformer attention
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    # Labels
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    ax.set_xlabel('Keys', fontsize=12)
    ax.set_ylabel('Queries', fontsize=12)
    ax.set_title('Attention Heatmap', fontsize=14)

    # Grid
    ax.set_xticks(np.arange(attention_matrix.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(attention_matrix.shape[0]) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig

# Example
seq_len = 10
attention = np.random.rand(seq_len, seq_len)
attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize

tokens = [f'Token_{i}' for i in range(seq_len)]
plot_attention_heatmap(attention, x_labels=tokens, y_labels=tokens)
plt.show()
```

### Image Grid

```python
def plot_image_grid(images, labels=None, nrows=4, ncols=4, figsize=(12, 12)):
    """
    Display a grid of images
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # Handle grayscale and RGB
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap='gray')
            else:
                ax.imshow(images[idx])

            if labels is not None:
                ax.set_title(f'Label: {labels[idx]}')

            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig

# Example
n_images = 16
images = [np.random.rand(28, 28) for _ in range(n_images)]
labels = np.random.randint(0, 10, n_images)

plot_image_grid(images, labels, nrows=4, ncols=4)
plt.show()
```

### Correlation Matrix

```python
def plot_correlation_matrix(data, feature_names=None, method='pearson'):
    """
    Plot correlation matrix heatmap
    """
    # Compute correlation
    if method == 'pearson':
        corr = np.corrcoef(data.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr, _ = spearmanr(data)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    # Labels
    if feature_names is not None:
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)

    # Add correlation values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if abs(corr[i, j]) > 0.5 else 'black',
                          fontsize=8)

    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14)
    plt.tight_layout()
    return fig

# Example
n_samples, n_features = 100, 10
data = np.random.randn(n_samples, n_features)
feature_names = [f'Feature {i}' for i in range(n_features)]

plot_correlation_matrix(data, feature_names)
plt.show()
```

---

## Working with Images

### Displaying Images

```python
# Single image
img = np.random.rand(100, 100, 3)  # RGB

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
ax.axis('off')
plt.show()

# Grayscale
img_gray = np.random.rand(100, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Grayscale (gray cmap)')
ax1.axis('off')

ax2.imshow(img_gray, cmap='viridis')
ax2.set_title('Grayscale (viridis cmap)')
ax2.axis('off')

plt.show()

# Control interpolation
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
small_img = np.random.rand(10, 10)

interpolations = ['nearest', 'bilinear', 'bicubic', 'lanczos']
for ax, interp in zip(axes.flat, interpolations):
    ax.imshow(small_img, cmap='gray', interpolation=interp)
    ax.set_title(f'Interpolation: {interp}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### Image Operations

```python
# Load image (with PIL or similar)
# from PIL import Image
# img = np.array(Image.open('image.jpg'))

# Simulated image
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Channels
axes[0, 1].imshow(img[:, :, 0], cmap='Reds')
axes[0, 1].set_title('Red Channel')
axes[0, 1].axis('off')

axes[0, 2].imshow(img[:, :, 1], cmap='Greens')
axes[0, 2].set_title('Green Channel')
axes[0, 2].axis('off')

axes[1, 0].imshow(img[:, :, 2], cmap='Blues')
axes[1, 0].set_title('Blue Channel')
axes[1, 0].axis('off')

# Histogram
axes[1, 1].hist(img[:, :, 0].ravel(), bins=50, alpha=0.5, color='red', label='R')
axes[1, 1].hist(img[:, :, 1].ravel(), bins=50, alpha=0.5, color='green', label='G')
axes[1, 1].hist(img[:, :, 2].ravel(), bins=50, alpha=0.5, color='blue', label='B')
axes[1, 1].set_title('Histogram')
axes[1, 1].legend()

# Grayscale
gray = np.mean(img, axis=2)
axes[1, 2].imshow(gray, cmap='gray')
axes[1, 2].set_title('Grayscale')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### Image Overlays and Masks

```python
# Base image
img = np.random.rand(100, 100, 3)

# Create mask
mask = np.zeros((100, 100))
mask[30:70, 30:70] = 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')

# Mask overlay
ax2.imshow(img)
ax2.imshow(mask, alpha=0.5, cmap='Reds')
ax2.set_title('With Mask Overlay')
ax2.axis('off')

# Masked image
masked_img = img.copy()
masked_img[mask == 0] = 0
ax3.imshow(masked_img)
ax3.set_title('Masked Image')
ax3.axis('off')

plt.tight_layout()
plt.show()
```

---

## Animations

### Basic Animation

```python
from matplotlib.animation import FuncAnimation

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=True)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True, interval=20)

# Save animation
# ani.save('sine_wave.gif', writer='pillow', fps=30)
# ani.save('sine_wave.mp4', writer='ffmpeg', fps=30)

plt.show()
```

### Animated Scatter

```python
# Animated scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
scat = ax.scatter([], [], s=100, alpha=0.6)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(frame):
    # Generate random walk
    n_points = 50
    x = np.random.randn(n_points).cumsum() * 0.1
    y = np.random.randn(n_points).cumsum() * 0.1
    data = np.c_[x, y]
    scat.set_offsets(data)
    scat.set_array(np.arange(n_points))
    return scat,

ani = FuncAnimation(fig, update, frames=100, init_func=init,
                    blit=True, interval=50)
plt.show()
```

### Animated Heatmap

```python
# Animated heatmap (useful for gradient visualization)
fig, ax = plt.subplots(figsize=(8, 6))

def animate(frame):
    ax.clear()
    data = np.random.rand(10, 10) * frame / 100
    im = ax.imshow(data, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Frame {frame}')
    return [im]

ani = FuncAnimation(fig, animate, frames=100, interval=50)
plt.show()
```

---

## Integration Patterns

### With NumPy

```python
# NumPy arrays are matplotlib's native format
x = np.linspace(0, 10, 1000)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

# Multi-dimensional data
data = np.random.randn(100, 100)
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='viridis')
plt.colorbar(im, ax=ax)
plt.show()
```

### With Pandas

```python
import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Pandas plotting (uses matplotlib)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
df['x'].hist(ax=axes[0, 0], bins=20)
axes[0, 0].set_title('Histogram')

# Scatter with categories
for cat in df['category'].unique():
    subset = df[df['category'] == cat]
    axes[0, 1].scatter(subset['x'], subset['y'], label=cat, alpha=0.6)
axes[0, 1].legend()
axes[0, 1].set_title('Scatter by Category')

# Box plot
df.boxplot(column=['x', 'y'], ax=axes[1, 0])
axes[1, 0].set_title('Box Plot')

# Time series
ts_df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'value': np.random.randn(100).cumsum()
})
ts_df.plot(x='date', y='value', ax=axes[1, 1])
axes[1, 1].set_title('Time Series')

plt.tight_layout()
plt.show()
```

### Jupyter Notebook Integration

```python
# Enable inline plotting
%matplotlib inline

# For interactive plots
%matplotlib notebook  # Old interactive backend
%matplotlib widget    # New interactive backend (requires ipympl)

# High-resolution figures
%config InlineBackend.figure_format = 'retina'

# Or in code
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')  # or 'pdf', 'retina'
```

---

## Performance & Best Practices

### Backends

```python
import matplotlib

# Check current backend
print(matplotlib.get_backend())

# Set backend (do this before importing pyplot)
# matplotlib.use('Agg')  # Non-interactive (for servers)
# matplotlib.use('TkAgg')  # Interactive
# matplotlib.use('Qt5Agg')  # Interactive with Qt

# Common backends:
# - 'Agg': PNG output, no display
# - 'PDF', 'PS', 'SVG': Vector outputs
# - 'TkAgg', 'Qt5Agg', 'GTK3Agg': Interactive
```

### Memory Management

```python
# Close figures to free memory
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
plt.savefig('plot.png')
plt.close(fig)  # Explicitly close

# Or close all figures
plt.close('all')

# For large datasets, downsample
large_x = np.linspace(0, 100, 1000000)
large_y = np.sin(large_x)

# Don't plot all points
step = len(large_x) // 1000
fig, ax = plt.subplots()
ax.plot(large_x[::step], large_y[::step])
plt.show()
```

### Saving Figures

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])

# Vector formats (scalable, publication-quality)
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('plot.svg', format='svg', bbox_inches='tight')
plt.savefig('plot.eps', format='eps', bbox_inches='tight')

# Raster formats
plt.savefig('plot.png', format='png', bbox_inches='tight', dpi=300)
plt.savefig('plot.jpg', format='jpg', bbox_inches='tight', dpi=300, quality=95)

# Transparent background
plt.savefig('plot.png', transparent=True, bbox_inches='tight', dpi=300)

# Specific size
fig.set_size_inches(8, 6)
plt.savefig('plot.png', dpi=300)  # Will be 2400x1800 pixels
```

### Publication-Quality Figures

```python
# Configure for publication
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

# Create plot
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(alpha=0.3)

# Save for publication
plt.savefig('publication_figure.pdf', format='pdf')
plt.savefig('publication_figure.png', dpi=600)  # High DPI for raster
plt.show()
```

---

## Common Patterns & Recipes

### Multi-Panel Figure

```python
# Complex multi-panel figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main plot (spans 2x2)
ax_main = fig.add_subplot(gs[:2, :2])
x = np.linspace(0, 10, 100)
ax_main.plot(x, np.sin(x))
ax_main.set_title('Main Plot', fontsize=14, fontweight='bold')

# Top right
ax_top = fig.add_subplot(gs[0, 2])
ax_top.hist(np.random.randn(1000), bins=30)
ax_top.set_title('Distribution')

# Middle right
ax_mid = fig.add_subplot(gs[1, 2])
ax_mid.scatter(np.random.rand(50), np.random.rand(50))
ax_mid.set_title('Scatter')

# Bottom (spans all columns)
ax_bottom = fig.add_subplot(gs[2, :])
ax_bottom.plot(x, np.cos(x))
ax_bottom.set_title('Bottom Plot')
ax_bottom.set_xlabel('X')

plt.show()
```

### Shared Color Scale

```python
# Multiple subplots with shared colorbar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

vmin, vmax = -1, 1  # Shared scale

for i, ax in enumerate(axes):
    data = np.random.randn(10, 10)
    im = ax.imshow(data, cmap='RdBu', vmin=vmin, vmax=vmax)
    ax.set_title(f'Subplot {i+1}')

# Single colorbar for all subplots
fig.colorbar(im, ax=axes, orientation='horizontal',
             fraction=0.05, pad=0.1, label='Value')

plt.tight_layout()
plt.show()
```

### Date Plotting

```python
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Generate time series data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
values = np.random.randn(365).cumsum()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, values)

# Format x-axis
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

# Rotate dates
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series with Date Formatting')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Logarithmic Scales

```python
x = np.logspace(0, 5, 100)
y = x ** 2

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear-linear
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Linear-Linear')

# Log-linear (semi-log y)
axes[0, 1].semilogy(x, y)
axes[0, 1].set_title('Log-Linear')

# Linear-log (semi-log x)
axes[1, 0].semilogx(x, y)
axes[1, 0].set_title('Linear-Log')

# Log-log
axes[1, 1].loglog(x, y)
axes[1, 1].set_title('Log-Log')

for ax in axes.flat:
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
```

### Filled Areas

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 1

# Fill between two curves
ax.fill_between(x, y1, y2, alpha=0.3, label='Between curves')

# Fill to axis
ax.fill_between(x, 0, y1, where=(y1 > 0), alpha=0.3,
                color='green', label='Positive')
ax.fill_between(x, 0, y1, where=(y1 < 0), alpha=0.3,
                color='red', label='Negative')

ax.plot(x, y1, 'k-', linewidth=2)
ax.plot(x, y2, 'k-', linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)

ax.legend()
ax.set_title('Filled Areas')
plt.show()
```

---

## Summary

Matplotlib is incredibly powerful and flexible. Key takeaways:

1. **Use the OO interface** for complex plots and production code
2. **Customize everything** - matplotlib gives you full control
3. **Plan your layout** with GridSpec for complex figures
4. **Think about your audience** - adjust style for presentations vs publications
5. **Use the right format** - vector (PDF/SVG) for publications, raster (PNG) for web
6. **Manage memory** - close figures, downsample large datasets
7. **Leverage colormaps** thoughtfully - use perceptually uniform for data
8. **Practice common patterns** - ML visualizations, multi-panel figures

**Next Steps:**
- Explore [Seaborn](https://seaborn.pydata.org/) for statistical visualizations
- Try [Plotly](https://plotly.com/) for interactive plots
- Check out [matplotlib gallery](https://matplotlib.org/stable/gallery/index.html) for inspiration
- Read [matplotlib cheatsheets](https://matplotlib.org/cheatsheets/)

**Resources:**
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Nicolas Rougier's Tutorial](https://github.com/rougier/matplotlib-tutorial)
- [Python Graph Gallery](https://www.python-graph-gallery.com/)
