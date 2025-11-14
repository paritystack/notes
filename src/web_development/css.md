# CSS (Cascading Style Sheets)

## Overview

CSS (Cascading Style Sheets) is a stylesheet language used to describe the presentation of HTML documents. It controls the visual appearance, layout, and responsive behavior of web pages, separating content from presentation.

**Key Features:**
- Separation of content and presentation
- Cascading and inheritance system
- Powerful selector system
- Flexible layout mechanisms (Flexbox, Grid)
- Animations and transitions
- Responsive design capabilities
- CSS Variables for dynamic styling
- Modular and maintainable with preprocessors

---

## Selectors

Selectors define which HTML elements to style. Understanding selectors is fundamental to effective CSS.

### Basic Selectors

```css
/* Universal selector - selects all elements */
* {
  margin: 0;
  padding: 0;
}

/* Element/Type selector */
p {
  font-size: 16px;
}

/* Class selector */
.container {
  max-width: 1200px;
}

/* ID selector */
#header {
  background-color: #333;
}

/* Multiple selectors */
h1, h2, h3 {
  font-family: Arial, sans-serif;
}
```

### Attribute Selectors

```css
/* Element with specific attribute */
[disabled] {
  opacity: 0.5;
}

/* Exact attribute value */
[type="text"] {
  border: 1px solid #ccc;
}

/* Attribute contains value */
[class*="btn"] {
  padding: 10px 20px;
}

/* Attribute starts with value */
[href^="https"] {
  color: green;
}

/* Attribute ends with value */
[href$=".pdf"] {
  color: red;
}

/* Attribute contains word */
[title~="important"] {
  font-weight: bold;
}

/* Attribute value or starts with value- */
[lang|="en"] {
  direction: ltr;
}
```

### Combinators

```css
/* Descendant combinator (space) - all descendants */
div p {
  color: blue;
}

/* Child combinator (>) - direct children only */
ul > li {
  list-style: none;
}

/* Adjacent sibling combinator (+) - immediately following sibling */
h2 + p {
  margin-top: 0;
}

/* General sibling combinator (~) - all following siblings */
h2 ~ p {
  color: gray;
}
```

### Pseudo-classes

Pseudo-classes select elements based on their state or position.

```css
/* Link states */
a:link { color: blue; }
a:visited { color: purple; }
a:hover { color: red; }
a:active { color: orange; }

/* Form states */
input:focus {
  outline: 2px solid blue;
}

input:disabled {
  background-color: #f0f0f0;
}

input:checked {
  accent-color: green;
}

input:required {
  border-color: red;
}

input:valid {
  border-color: green;
}

input:invalid {
  border-color: red;
}

/* Structural pseudo-classes */
/* First/last child */
li:first-child {
  font-weight: bold;
}

li:last-child {
  border-bottom: none;
}

/* Only child */
p:only-child {
  margin: 0;
}

/* nth-child patterns */
tr:nth-child(odd) {
  background-color: #f9f9f9;
}

tr:nth-child(even) {
  background-color: #fff;
}

tr:nth-child(3n) {
  /* Every 3rd element */
  background-color: yellow;
}

tr:nth-child(3n+1) {
  /* 1st, 4th, 7th, etc. */
  background-color: lightblue;
}

/* nth-of-type - same as nth-child but considers element type */
p:nth-of-type(2) {
  color: red;
}

/* first-of-type / last-of-type */
article p:first-of-type {
  font-size: 1.2em;
}

/* Other structural */
:root {
  /* Root element (html) */
  --primary-color: #007bff;
}

:empty {
  /* Elements with no children */
  display: none;
}

/* Negation pseudo-class */
div:not(.excluded) {
  display: block;
}

input:not([type="submit"]) {
  border: 1px solid #ccc;
}

/* Target pseudo-class */
:target {
  /* Element targeted by URL fragment */
  background-color: yellow;
}
```

### Pseudo-elements

Pseudo-elements style specific parts of elements.

```css
/* ::before and ::after - insert content */
.quote::before {
  content: """;
  font-size: 2em;
  color: #999;
}

.quote::after {
  content: """;
}

/* ::first-letter */
p::first-letter {
  font-size: 2em;
  font-weight: bold;
  float: left;
  line-height: 1;
}

/* ::first-line */
p::first-line {
  font-weight: bold;
  color: #333;
}

/* ::selection - highlighted text */
::selection {
  background-color: yellow;
  color: black;
}

/* ::placeholder */
input::placeholder {
  color: #999;
  font-style: italic;
}

/* ::marker - list item markers */
li::marker {
  color: red;
  font-weight: bold;
}
```

### Selector Specificity

Specificity determines which styles are applied when multiple rules match an element.

**Specificity Calculation:**
- Inline styles: 1000
- IDs: 100
- Classes, attributes, pseudo-classes: 10
- Elements, pseudo-elements: 1

```css
/* Specificity: 1 */
p { color: black; }

/* Specificity: 10 */
.text { color: blue; }

/* Specificity: 100 */
#main { color: green; }

/* Specificity: 111 */
#main p.text { color: red; }

/* Specificity: 1000 */
<p style="color: purple;">

/* !important overrides specificity (use sparingly!) */
p { color: orange !important; }
```

**Specificity Best Practices:**
1. Keep specificity low
2. Avoid IDs for styling
3. Use classes primarily
4. Avoid `!important` except for utilities
5. Order matters when specificity is equal

---

## The Box Model

Every element in CSS is a rectangular box consisting of content, padding, border, and margin.

### Box Model Components

```css
.box {
  /* Content area */
  width: 300px;
  height: 200px;

  /* Padding - space inside border */
  padding: 20px;
  /* or */
  padding-top: 10px;
  padding-right: 20px;
  padding-bottom: 10px;
  padding-left: 20px;
  /* or shorthand */
  padding: 10px 20px;          /* vertical horizontal */
  padding: 10px 20px 15px;     /* top horizontal bottom */
  padding: 10px 20px 15px 5px; /* top right bottom left (clockwise) */

  /* Border */
  border: 2px solid #333;
  /* or detailed */
  border-width: 2px;
  border-style: solid;  /* solid, dashed, dotted, double, groove, ridge, inset, outset, none */
  border-color: #333;
  /* individual sides */
  border-top: 1px solid red;
  border-right: 2px dashed blue;

  /* Margin - space outside border */
  margin: 20px;
  /* same shorthand patterns as padding */
  margin: 10px auto;  /* vertical=10px, horizontal=auto (centers block element) */

  /* Margin collapse - adjacent vertical margins collapse to larger value */
}

/* Box-sizing */
.box-default {
  box-sizing: content-box;  /* Default: width/height apply to content only */
  width: 300px;
  padding: 20px;
  border: 5px solid black;
  /* Actual width: 300 + 40 (padding) + 10 (border) = 350px */
}

.box-border {
  box-sizing: border-box;  /* Width/height include padding and border */
  width: 300px;
  padding: 20px;
  border: 5px solid black;
  /* Actual width: 300px (includes padding and border) */
  /* Content width: 300 - 40 - 10 = 250px */
}

/* Global box-sizing (common practice) */
*, *::before, *::after {
  box-sizing: border-box;
}
```

### Display Property

```css
/* Block - full width, new line */
div {
  display: block;
  width: 100%;  /* Takes full width by default */
}

/* Inline - flows with text, width/height ignored */
span {
  display: inline;
  width: 100px;    /* Ignored */
  height: 50px;    /* Ignored */
  margin: 10px 0;  /* Vertical margins ignored */
}

/* Inline-block - flows with text but respects width/height */
.button {
  display: inline-block;
  width: 100px;
  height: 40px;
  padding: 10px;
}

/* None - removes from document flow */
.hidden {
  display: none;  /* Element not rendered, no space taken */
}

/* Visibility (alternative to display: none) */
.invisible {
  visibility: hidden;  /* Element invisible but space preserved */
}

/* Flex - flexible box layout */
.container {
  display: flex;
}

/* Grid - grid layout */
.grid-container {
  display: grid;
}

/* Table display values */
.table { display: table; }
.table-row { display: table-row; }
.table-cell { display: table-cell; }
```

---

## Positioning

CSS positioning controls how elements are placed in the document flow.

### Position Values

```css
/* Static (default) - normal document flow */
.static {
  position: static;
  /* top, right, bottom, left have no effect */
}

/* Relative - positioned relative to normal position */
.relative {
  position: relative;
  top: 20px;    /* Moves down 20px from normal position */
  left: 10px;   /* Moves right 10px from normal position */
  /* Space in normal flow is preserved */
}

/* Absolute - positioned relative to nearest positioned ancestor */
.absolute {
  position: absolute;
  top: 0;
  right: 0;
  /* Removed from normal flow, no space reserved */
  /* If no positioned ancestor, positioned relative to viewport */
}

/* Fixed - positioned relative to viewport */
.fixed {
  position: fixed;
  bottom: 20px;
  right: 20px;
  /* Stays in place when scrolling */
}

/* Sticky - hybrid of relative and fixed */
.sticky {
  position: sticky;
  top: 0;
  /* Acts as relative until scroll threshold, then becomes fixed */
}

/* Z-index - stacking order */
.modal {
  position: absolute;
  z-index: 1000;  /* Higher values appear on top */
}

.overlay {
  position: fixed;
  z-index: 999;
}

/* Common pattern: Centered absolute positioning */
.centered {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* Positioning context */
.parent {
  position: relative;  /* Creates positioning context for children */
}

.child {
  position: absolute;
  top: 0;
  left: 0;  /* Positioned relative to .parent */
}
```

---

## Flexbox

Flexbox is a one-dimensional layout system for arranging items in rows or columns.

### Flex Container Properties

```css
.container {
  display: flex;  /* or inline-flex */

  /* Flex direction - main axis direction */
  flex-direction: row;              /* Default: left to right */
  flex-direction: row-reverse;      /* Right to left */
  flex-direction: column;           /* Top to bottom */
  flex-direction: column-reverse;   /* Bottom to top */

  /* Flex wrap - whether items wrap to new lines */
  flex-wrap: nowrap;   /* Default: single line */
  flex-wrap: wrap;     /* Multi-line, top to bottom */
  flex-wrap: wrap-reverse;  /* Multi-line, bottom to top */

  /* Shorthand for direction and wrap */
  flex-flow: row wrap;

  /* Justify content - alignment along main axis */
  justify-content: flex-start;    /* Default: start of container */
  justify-content: flex-end;      /* End of container */
  justify-content: center;        /* Center of container */
  justify-content: space-between; /* Even spacing, no space at edges */
  justify-content: space-around;  /* Even spacing, half space at edges */
  justify-content: space-evenly;  /* Even spacing including edges */

  /* Align items - alignment along cross axis */
  align-items: stretch;      /* Default: stretch to fill */
  align-items: flex-start;   /* Start of cross axis */
  align-items: flex-end;     /* End of cross axis */
  align-items: center;       /* Center of cross axis */
  align-items: baseline;     /* Align baselines */

  /* Align content - alignment of multiple lines (when wrapped) */
  align-content: stretch;
  align-content: flex-start;
  align-content: flex-end;
  align-content: center;
  align-content: space-between;
  align-content: space-around;

  /* Gap between items (modern) */
  gap: 20px;            /* Both row and column gap */
  row-gap: 10px;
  column-gap: 20px;
}
```

### Flex Item Properties

```css
.item {
  /* Flex grow - how much item grows relative to siblings */
  flex-grow: 0;    /* Default: don't grow */
  flex-grow: 1;    /* Grow to fill space equally */
  flex-grow: 2;    /* Grow twice as much as items with flex-grow: 1 */

  /* Flex shrink - how much item shrinks when needed */
  flex-shrink: 1;  /* Default: shrink if necessary */
  flex-shrink: 0;  /* Don't shrink */

  /* Flex basis - initial size before growing/shrinking */
  flex-basis: auto;    /* Default: based on content */
  flex-basis: 200px;   /* Specific size */
  flex-basis: 0;       /* Ignore content size */

  /* Shorthand for grow, shrink, basis */
  flex: 0 1 auto;      /* Default */
  flex: 1;             /* flex: 1 1 0 - equal sizing */
  flex: 2;             /* flex: 2 1 0 - twice the size */
  flex: none;          /* flex: 0 0 auto - fixed size */
  flex: auto;          /* flex: 1 1 auto - based on content */

  /* Align self - override align-items for individual item */
  align-self: auto;        /* Default: inherit from container */
  align-self: flex-start;
  align-self: flex-end;
  align-self: center;
  align-self: stretch;
  align-self: baseline;

  /* Order - visual order (doesn't affect DOM order) */
  order: 0;    /* Default */
  order: 1;    /* Appears after order: 0 items */
  order: -1;   /* Appears before order: 0 items */
}
```

### Common Flexbox Patterns

```css
/* Horizontal centering */
.horizontal-center {
  display: flex;
  justify-content: center;
}

/* Vertical centering */
.vertical-center {
  display: flex;
  align-items: center;
}

/* Perfect centering */
.perfect-center {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Equal width columns */
.equal-columns .column {
  flex: 1;
}

/* Sidebar layout */
.sidebar-layout {
  display: flex;
}

.sidebar {
  flex: 0 0 250px;  /* Fixed 250px width */
}

.main-content {
  flex: 1;  /* Takes remaining space */
}

/* Card layout with wrapping */
.card-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.card {
  flex: 1 1 300px;  /* Grow, shrink, min 300px */
}

/* Space between items */
.space-between {
  display: flex;
  justify-content: space-between;
}

/* Align last item to end */
.push-last {
  display: flex;
}

.push-last .last {
  margin-left: auto;
}
```

---

## CSS Grid

CSS Grid is a two-dimensional layout system for creating complex layouts with rows and columns.

### Grid Container Properties

```css
.container {
  display: grid;  /* or inline-grid */

  /* Define columns */
  grid-template-columns: 200px 200px 200px;  /* 3 fixed columns */
  grid-template-columns: 1fr 1fr 1fr;        /* 3 equal flexible columns */
  grid-template-columns: 1fr 2fr 1fr;        /* Middle column twice as wide */
  grid-template-columns: 200px 1fr 200px;    /* Fixed sidebars, flexible center */
  grid-template-columns: repeat(3, 1fr);     /* Shorthand for equal columns */
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));  /* Responsive columns */

  /* Define rows */
  grid-template-rows: 100px auto 100px;      /* Header, content, footer */
  grid-template-rows: repeat(3, 200px);      /* 3 equal rows */

  /* Named grid lines */
  grid-template-columns: [start] 1fr [middle] 1fr [end];

  /* Grid template areas - visual layout */
  grid-template-areas:
    "header header header"
    "sidebar main main"
    "footer footer footer";

  /* Gap between grid cells */
  gap: 20px;               /* Both row and column gap */
  row-gap: 10px;
  column-gap: 20px;

  /* Justify items - horizontal alignment within cells */
  justify-items: stretch;  /* Default */
  justify-items: start;
  justify-items: end;
  justify-items: center;

  /* Align items - vertical alignment within cells */
  align-items: stretch;    /* Default */
  align-items: start;
  align-items: end;
  align-items: center;

  /* Justify content - horizontal alignment of grid within container */
  justify-content: start;
  justify-content: end;
  justify-content: center;
  justify-content: space-between;
  justify-content: space-around;
  justify-content: space-evenly;

  /* Align content - vertical alignment of grid within container */
  align-content: start;
  align-content: end;
  align-content: center;
  align-content: space-between;
  align-content: space-around;
  align-content: space-evenly;

  /* Auto rows/columns - size of implicit tracks */
  grid-auto-rows: 100px;
  grid-auto-columns: 200px;

  /* Auto flow - how auto-placed items flow */
  grid-auto-flow: row;      /* Default: fill rows */
  grid-auto-flow: column;   /* Fill columns */
  grid-auto-flow: dense;    /* Fill gaps (may reorder) */
}
```

### Grid Item Properties

```css
.item {
  /* Grid column placement */
  grid-column-start: 1;
  grid-column-end: 3;      /* Spans from column 1 to 3 */
  grid-column: 1 / 3;      /* Shorthand */
  grid-column: 1 / span 2; /* Start at 1, span 2 columns */
  grid-column: 1 / -1;     /* Span to last column */

  /* Grid row placement */
  grid-row-start: 1;
  grid-row-end: 3;
  grid-row: 1 / 3;         /* Shorthand */
  grid-row: 2 / span 2;    /* Start at row 2, span 2 rows */

  /* Grid area - shorthand for row-start / column-start / row-end / column-end */
  grid-area: 1 / 1 / 3 / 3;

  /* Named area */
  grid-area: header;       /* Uses grid-template-areas from container */

  /* Justify self - horizontal alignment for this item */
  justify-self: stretch;
  justify-self: start;
  justify-self: end;
  justify-self: center;

  /* Align self - vertical alignment for this item */
  align-self: stretch;
  align-self: start;
  align-self: end;
  align-self: center;

  /* Z-index works with grid items */
  z-index: 1;
}
```

### Common Grid Patterns

```css
/* Simple 3-column layout */
.three-column {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

/* Responsive grid - auto-fit columns */
.responsive-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

/* Holy Grail layout */
.holy-grail {
  display: grid;
  grid-template-areas:
    "header header header"
    "nav main aside"
    "footer footer footer";
  grid-template-rows: auto 1fr auto;
  grid-template-columns: 200px 1fr 200px;
  gap: 10px;
  min-height: 100vh;
}

.header { grid-area: header; }
.nav { grid-area: nav; }
.main { grid-area: main; }
.aside { grid-area: aside; }
.footer { grid-area: footer; }

/* Card grid with different sizes */
.masonry-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  grid-auto-rows: 100px;
  gap: 10px;
}

.card-large {
  grid-row: span 2;
  grid-column: span 2;
}

/* Centered grid */
.centered-grid {
  display: grid;
  place-items: center;  /* Shorthand for justify-items: center; align-items: center; */
}

/* Full-page layout */
.page-layout {
  display: grid;
  grid-template-rows: 60px 1fr 40px;
  min-height: 100vh;
}
```

---

## Typography

### Font Properties

```css
.text {
  /* Font family */
  font-family: Arial, Helvetica, sans-serif;
  font-family: 'Times New Roman', serif;
  font-family: 'Courier New', monospace;
  font-family: Georgia, 'Times New Roman', serif;  /* Fallback fonts */

  /* Font size */
  font-size: 16px;          /* Absolute */
  font-size: 1.2em;         /* Relative to parent */
  font-size: 1.2rem;        /* Relative to root (html) */
  font-size: 100%;          /* Relative to parent */
  font-size: larger;        /* Keyword */

  /* Font weight */
  font-weight: normal;      /* 400 */
  font-weight: bold;        /* 700 */
  font-weight: lighter;     /* Relative to parent */
  font-weight: bolder;      /* Relative to parent */
  font-weight: 100;         /* Thin */
  font-weight: 300;         /* Light */
  font-weight: 400;         /* Normal */
  font-weight: 500;         /* Medium */
  font-weight: 700;         /* Bold */
  font-weight: 900;         /* Black */

  /* Font style */
  font-style: normal;
  font-style: italic;
  font-style: oblique;

  /* Font variant */
  font-variant: normal;
  font-variant: small-caps;

  /* Font shorthand: style variant weight size/line-height family */
  font: italic small-caps bold 16px/1.5 Arial, sans-serif;

  /* Line height */
  line-height: 1.5;         /* Recommended for body text */
  line-height: 24px;        /* Absolute */
  line-height: 150%;        /* Percentage */

  /* Letter spacing */
  letter-spacing: normal;
  letter-spacing: 0.05em;
  letter-spacing: 2px;

  /* Word spacing */
  word-spacing: normal;
  word-spacing: 5px;
}
```

### Text Properties

```css
.text {
  /* Text alignment */
  text-align: left;
  text-align: right;
  text-align: center;
  text-align: justify;

  /* Text decoration */
  text-decoration: none;
  text-decoration: underline;
  text-decoration: overline;
  text-decoration: line-through;
  text-decoration: underline dotted red;  /* line style color */

  /* Text transform */
  text-transform: none;
  text-transform: uppercase;
  text-transform: lowercase;
  text-transform: capitalize;  /* First letter of each word */

  /* Text indent */
  text-indent: 0;
  text-indent: 2em;     /* Indent first line */
  text-indent: -999px;  /* Hide text (accessibility hack) */

  /* White space */
  white-space: normal;    /* Collapse whitespace, wrap lines */
  white-space: nowrap;    /* No wrapping */
  white-space: pre;       /* Preserve whitespace, no wrapping */
  white-space: pre-wrap;  /* Preserve whitespace, wrap lines */
  white-space: pre-line;  /* Preserve line breaks, wrap lines */

  /* Word break */
  word-break: normal;
  word-break: break-all;  /* Break anywhere */
  word-break: keep-all;   /* Don't break CJK text */

  /* Overflow wrap */
  overflow-wrap: normal;
  overflow-wrap: break-word;  /* Break long words */

  /* Text overflow */
  overflow: hidden;
  text-overflow: clip;
  text-overflow: ellipsis;  /* Show ... when text overflows */
  white-space: nowrap;       /* Required for ellipsis */

  /* Vertical alignment */
  vertical-align: baseline;  /* Default */
  vertical-align: top;
  vertical-align: middle;
  vertical-align: bottom;
  vertical-align: sub;
  vertical-align: super;
  vertical-align: 5px;       /* Relative to baseline */

  /* Text shadow */
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);  /* x-offset y-offset blur color */
  text-shadow: 1px 1px 2px black, 0 0 25px blue;  /* Multiple shadows */
}
```

### Web Fonts

```css
/* Google Fonts import */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

/* Custom font face */
@font-face {
  font-family: 'CustomFont';
  src: url('fonts/custom-font.woff2') format('woff2'),
       url('fonts/custom-font.woff') format('woff');
  font-weight: normal;
  font-style: normal;
  font-display: swap;  /* Improves performance */
}

.custom-text {
  font-family: 'CustomFont', sans-serif;
}

/* Variable fonts */
@font-face {
  font-family: 'Variable Font';
  src: url('font.woff2') format('woff2-variations');
  font-weight: 100 900;  /* Range of weights available */
}

.variable-text {
  font-family: 'Variable Font';
  font-weight: 450;  /* Any weight in range */
  font-variation-settings: 'wght' 450, 'wdth' 100;
}
```

---

## Colors and Backgrounds

### Color Values

```css
.colors {
  /* Named colors */
  color: red;
  color: cornflowerblue;
  color: transparent;

  /* Hexadecimal */
  color: #ff0000;      /* Red */
  color: #f00;         /* Short form */
  color: #ff0000ff;    /* With alpha (RGBA) */

  /* RGB */
  color: rgb(255, 0, 0);
  color: rgba(255, 0, 0, 0.5);  /* With alpha (50% opacity) */

  /* HSL (Hue, Saturation, Lightness) */
  color: hsl(0, 100%, 50%);       /* Red */
  color: hsla(0, 100%, 50%, 0.5); /* With alpha */

  /* Modern syntax (space-separated, optional alpha) */
  color: rgb(255 0 0 / 50%);
  color: hsl(0 100% 50% / 0.5);

  /* currentColor - inherits text color */
  border-color: currentColor;
}
```

### Background Properties

```css
.background {
  /* Background color */
  background-color: #f0f0f0;
  background-color: rgba(0, 0, 0, 0.1);

  /* Background image */
  background-image: url('image.jpg');
  background-image: url('data:image/svg+xml,...');  /* Data URI */

  /* Multiple backgrounds */
  background-image: url('overlay.png'), url('background.jpg');

  /* Background repeat */
  background-repeat: repeat;      /* Default */
  background-repeat: no-repeat;
  background-repeat: repeat-x;    /* Horizontal only */
  background-repeat: repeat-y;    /* Vertical only */
  background-repeat: space;       /* Repeat with spacing */
  background-repeat: round;       /* Repeat and scale */

  /* Background position */
  background-position: top left;
  background-position: center;
  background-position: 50% 50%;
  background-position: right 20px bottom 10px;

  /* Background size */
  background-size: auto;          /* Default */
  background-size: cover;         /* Scale to cover entire element */
  background-size: contain;       /* Scale to fit within element */
  background-size: 100px 50px;    /* Specific dimensions */
  background-size: 50%;           /* Percentage */

  /* Background attachment */
  background-attachment: scroll;  /* Default: scrolls with page */
  background-attachment: fixed;   /* Fixed relative to viewport */
  background-attachment: local;   /* Scrolls with element content */

  /* Background origin */
  background-origin: padding-box; /* Default */
  background-origin: border-box;
  background-origin: content-box;

  /* Background clip */
  background-clip: border-box;    /* Default */
  background-clip: padding-box;
  background-clip: content-box;
  background-clip: text;          /* Clip to text (requires -webkit-) */

  /* Background shorthand */
  background: #f0f0f0 url('bg.jpg') no-repeat center / cover fixed;
  /* color image repeat position / size attachment */
}
```

### Gradients

```css
/* Linear gradients */
.linear-gradient {
  background: linear-gradient(to right, red, blue);
  background: linear-gradient(45deg, red, blue);
  background: linear-gradient(to bottom right, red, yellow, blue);
  background: linear-gradient(red 0%, yellow 50%, blue 100%);

  /* Multiple color stops */
  background: linear-gradient(
    to right,
    red 0%,
    orange 20%,
    yellow 40%,
    green 60%,
    blue 80%,
    purple 100%
  );
}

/* Radial gradients */
.radial-gradient {
  background: radial-gradient(circle, red, blue);
  background: radial-gradient(ellipse at center, red, blue);
  background: radial-gradient(circle at top left, red, blue);
  background: radial-gradient(circle closest-side, red, blue);
  background: radial-gradient(circle farthest-corner at 30% 40%, red, blue);
}

/* Conic gradients */
.conic-gradient {
  background: conic-gradient(red, yellow, green, blue, red);
  background: conic-gradient(from 45deg, red, blue);
  background: conic-gradient(at 30% 40%, red, blue);
}

/* Repeating gradients */
.repeating-gradient {
  background: repeating-linear-gradient(
    45deg,
    red 0px,
    red 10px,
    blue 10px,
    blue 20px
  );

  background: repeating-radial-gradient(
    circle,
    red 0px,
    red 10px,
    blue 10px,
    blue 20px
  );
}
```

---

## Borders, Shadows, and Effects

### Borders

```css
.borders {
  /* Border properties */
  border: 2px solid #333;
  border-width: 1px;
  border-style: solid;  /* solid, dashed, dotted, double, groove, ridge, inset, outset */
  border-color: #333;

  /* Individual sides */
  border-top: 1px solid red;
  border-right: 2px dashed blue;
  border-bottom: 3px dotted green;
  border-left: 4px double purple;

  /* Border radius - rounded corners */
  border-radius: 5px;
  border-radius: 10px 20px;           /* top-left/bottom-right top-right/bottom-left */
  border-radius: 10px 20px 30px 40px; /* top-left top-right bottom-right bottom-left */
  border-radius: 50%;                 /* Circle (on square element) */

  /* Individual corners */
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  border-bottom-right-radius: 10px;
  border-bottom-left-radius: 10px;

  /* Elliptical corners */
  border-radius: 50px / 25px;  /* horizontal / vertical */

  /* Border image */
  border-image-source: url('border.png');
  border-image-slice: 30;
  border-image-repeat: stretch;  /* stretch, repeat, round, space */
  border-image: url('border.png') 30 stretch;  /* Shorthand */
}
```

### Shadows

```css
/* Box shadow */
.box-shadow {
  /* x-offset y-offset blur spread color */
  box-shadow: 2px 2px 10px 0px rgba(0, 0, 0, 0.3);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

  /* Inset shadow */
  box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);

  /* Multiple shadows */
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.12),
    0 1px 2px rgba(0, 0, 0, 0.24);

  /* Elevated effect */
  box-shadow:
    0 2.8px 2.2px rgba(0, 0, 0, 0.034),
    0 6.7px 5.3px rgba(0, 0, 0, 0.048),
    0 12.5px 10px rgba(0, 0, 0, 0.06),
    0 22.3px 17.9px rgba(0, 0, 0, 0.072);
}

/* Drop shadow (for non-rectangular shapes) */
.drop-shadow {
  filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.5));
}

/* Text shadow */
.text-shadow {
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  text-shadow:
    1px 1px 2px black,
    0 0 25px blue,
    0 0 5px darkblue;
}
```

### Filters

```css
.filters {
  /* Blur */
  filter: blur(5px);

  /* Brightness */
  filter: brightness(1.2);  /* 120% */

  /* Contrast */
  filter: contrast(1.5);

  /* Grayscale */
  filter: grayscale(100%);

  /* Hue rotation */
  filter: hue-rotate(90deg);

  /* Invert */
  filter: invert(100%);

  /* Opacity */
  filter: opacity(50%);

  /* Saturate */
  filter: saturate(200%);

  /* Sepia */
  filter: sepia(100%);

  /* Drop shadow */
  filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.5));

  /* Multiple filters */
  filter: brightness(1.1) contrast(1.2) saturate(1.3);
}

/* Backdrop filter - filters background behind element */
.backdrop-filter {
  backdrop-filter: blur(10px);
  background-color: rgba(255, 255, 255, 0.5);
}
```

### Opacity

```css
.opacity {
  opacity: 1;     /* Fully opaque (default) */
  opacity: 0.5;   /* 50% transparent */
  opacity: 0;     /* Fully transparent */

  /* Opacity affects entire element including children */
  /* Use rgba() for transparency without affecting children */
}
```

---

## Transitions and Animations

### Transitions

Transitions enable smooth changes between property values.

```css
.transition {
  /* Individual properties */
  transition-property: background-color;
  transition-duration: 0.3s;
  transition-timing-function: ease;
  transition-delay: 0s;

  /* Shorthand: property duration timing-function delay */
  transition: background-color 0.3s ease 0s;
  transition: all 0.3s ease;  /* Transition all properties */

  /* Multiple properties */
  transition:
    background-color 0.3s ease,
    transform 0.2s ease-in-out,
    box-shadow 0.3s ease;
}

/* Timing functions */
.timing-functions {
  transition-timing-function: linear;       /* Constant speed */
  transition-timing-function: ease;         /* Default: slow-fast-slow */
  transition-timing-function: ease-in;      /* Slow start */
  transition-timing-function: ease-out;     /* Slow end */
  transition-timing-function: ease-in-out;  /* Slow start and end */
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);  /* Custom curve */
  transition-timing-function: steps(4);     /* Stepped animation */
  transition-timing-function: step-start;
  transition-timing-function: step-end;
}

/* Common transition patterns */
.button {
  background-color: blue;
  transform: scale(1);
  transition: all 0.3s ease;
}

.button:hover {
  background-color: darkblue;
  transform: scale(1.05);
}

.card {
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: box-shadow 0.3s ease;
}

.card:hover {
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
```

### Animations

Animations provide more control than transitions with keyframes.

```css
/* Define animation with keyframes */
@keyframes slideIn {
  from {
    transform: translateX(-100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* Alternative keyframe syntax with percentages */
@keyframes bounce {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-20px);
  }
  100% {
    transform: translateY(0);
  }
}

/* Complex animation */
@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
}

/* Apply animation */
.animated {
  /* Individual properties */
  animation-name: slideIn;
  animation-duration: 1s;
  animation-timing-function: ease;
  animation-delay: 0s;
  animation-iteration-count: 1;      /* or infinite */
  animation-direction: normal;       /* normal, reverse, alternate, alternate-reverse */
  animation-fill-mode: forwards;     /* none, forwards, backwards, both */
  animation-play-state: running;     /* running, paused */

  /* Shorthand: name duration timing-function delay iteration-count direction fill-mode */
  animation: slideIn 1s ease 0s 1 normal forwards;
  animation: bounce 2s ease-in-out infinite;

  /* Multiple animations */
  animation:
    slideIn 1s ease forwards,
    pulse 2s ease-in-out infinite;
}

/* Control animation with JavaScript or pseudo-classes */
.element:hover {
  animation-play-state: paused;
}

/* Common animation patterns */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes fadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
  20%, 40%, 60%, 80% { transform: translateX(10px); }
}

@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
```

---

## Transforms

Transforms modify the coordinate space of elements.

```css
.transforms {
  /* 2D Transforms */

  /* Translate - move element */
  transform: translateX(50px);
  transform: translateY(20px);
  transform: translate(50px, 20px);  /* x, y */

  /* Scale - resize element */
  transform: scaleX(1.5);
  transform: scaleY(0.5);
  transform: scale(1.2);          /* uniform scaling */
  transform: scale(1.5, 0.8);     /* x, y */

  /* Rotate - rotate element */
  transform: rotate(45deg);
  transform: rotate(-90deg);

  /* Skew - skew element */
  transform: skewX(20deg);
  transform: skewY(10deg);
  transform: skew(20deg, 10deg);

  /* Multiple transforms */
  transform: translate(50px, 20px) rotate(45deg) scale(1.2);

  /* Transform origin - point around which transforms occur */
  transform-origin: center;           /* Default */
  transform-origin: top left;
  transform-origin: 50% 50%;
  transform-origin: 0 0;

  /* 3D Transforms */

  /* Translate 3D */
  transform: translateZ(50px);
  transform: translate3d(50px, 20px, 10px);  /* x, y, z */

  /* Scale 3D */
  transform: scaleZ(2);
  transform: scale3d(1.5, 1.5, 2);

  /* Rotate 3D */
  transform: rotateX(45deg);
  transform: rotateY(45deg);
  transform: rotateZ(45deg);  /* Same as rotate() */
  transform: rotate3d(1, 1, 0, 45deg);  /* x, y, z, angle */

  /* Perspective - 3D depth */
  perspective: 1000px;        /* On parent */
  transform: perspective(1000px) rotateY(45deg);  /* On element */

  /* Perspective origin */
  perspective-origin: center;
  perspective-origin: 50% 50%;

  /* Transform style - preserve 3D */
  transform-style: flat;        /* Default */
  transform-style: preserve-3d; /* Children in 3D space */

  /* Backface visibility */
  backface-visibility: visible; /* Default */
  backface-visibility: hidden;  /* Hide back face when rotated */
}

/* Common transform patterns */
.card-flip {
  transform-style: preserve-3d;
  transition: transform 0.6s;
}

.card-flip:hover {
  transform: rotateY(180deg);
}

.zoom-on-hover {
  transition: transform 0.3s;
}

.zoom-on-hover:hover {
  transform: scale(1.1);
}
```

---

## Responsive Design

### Media Queries

```css
/* Basic media query syntax */
@media media-type and (condition) {
  /* CSS rules */
}

/* Common breakpoints */
/* Mobile first approach */
/* Base styles for mobile */
.container {
  width: 100%;
  padding: 15px;
}

/* Tablet (768px and up) */
@media (min-width: 768px) {
  .container {
    width: 750px;
    margin: 0 auto;
  }
}

/* Desktop (1024px and up) */
@media (min-width: 1024px) {
  .container {
    width: 970px;
  }
}

/* Large desktop (1200px and up) */
@media (min-width: 1200px) {
  .container {
    width: 1170px;
  }
}

/* Desktop first approach */
/* Base styles for desktop */
.sidebar {
  width: 25%;
  float: left;
}

/* Tablet and below */
@media (max-width: 1023px) {
  .sidebar {
    width: 100%;
    float: none;
  }
}

/* Mobile */
@media (max-width: 767px) {
  .sidebar {
    margin-bottom: 20px;
  }
}

/* Range queries */
@media (min-width: 768px) and (max-width: 1023px) {
  /* Tablet only */
}

/* Orientation */
@media (orientation: portrait) {
  /* Portrait orientation */
}

@media (orientation: landscape) {
  /* Landscape orientation */
}

/* Device pixel ratio (retina displays) */
@media (min-resolution: 2dppx) {
  /* Retina displays */
  .logo {
    background-image: url('logo@2x.png');
  }
}

/* Prefer color scheme */
@media (prefers-color-scheme: dark) {
  body {
    background-color: #222;
    color: #fff;
  }
}

@media (prefers-color-scheme: light) {
  body {
    background-color: #fff;
    color: #222;
  }
}

/* Reduced motion (accessibility) */
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
  }
}

/* Print styles */
@media print {
  .no-print {
    display: none;
  }

  body {
    font-size: 12pt;
    color: black;
    background: white;
  }
}

/* Hover capability */
@media (hover: hover) {
  /* Device supports hover */
  .button:hover {
    background-color: blue;
  }
}

@media (hover: none) {
  /* Touch device without hover */
  .button:active {
    background-color: blue;
  }
}
```

### Container Queries (Modern)

```css
/* Container queries allow responsive design based on container size */
.container {
  container-type: inline-size;  /* Creates query container */
  container-name: card;         /* Optional name */
}

@container (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 1fr 2fr;
  }
}

@container card (min-width: 600px) {
  /* Query named container */
  .card {
    grid-template-columns: 1fr 1fr 1fr;
  }
}
```

### Responsive Units

```css
.responsive-units {
  /* Viewport units */
  width: 100vw;    /* 100% of viewport width */
  height: 100vh;   /* 100% of viewport height */
  width: 50vmin;   /* 50% of smaller viewport dimension */
  width: 50vmax;   /* 50% of larger viewport dimension */

  /* Relative units */
  font-size: 1em;    /* Relative to parent font-size */
  font-size: 1rem;   /* Relative to root (html) font-size */
  width: 50%;        /* Relative to parent width */

  /* Fluid typography */
  font-size: calc(16px + 0.5vw);  /* Scales with viewport */
  font-size: clamp(16px, 4vw, 24px);  /* Min, preferred, max */
}

/* Responsive font sizing */
html {
  font-size: 16px;  /* Base size */
}

@media (min-width: 768px) {
  html {
    font-size: 18px;
  }
}

@media (min-width: 1200px) {
  html {
    font-size: 20px;
  }
}

h1 {
  font-size: 2rem;  /* Scales with base font-size */
}
```

---

## CSS Variables (Custom Properties)

CSS Variables enable dynamic, reusable values throughout stylesheets.

```css
/* Define variables (usually in :root) */
:root {
  /* Colors */
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --success-color: #28a745;
  --danger-color: #dc3545;
  --text-color: #333;
  --bg-color: #fff;

  /* Spacing */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* Typography */
  --font-primary: 'Arial', sans-serif;
  --font-secondary: 'Georgia', serif;
  --font-size-base: 16px;
  --line-height-base: 1.5;

  /* Breakpoints */
  --breakpoint-sm: 576px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 992px;
  --breakpoint-xl: 1200px;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);

  /* Border radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 16px;
  --radius-full: 9999px;

  /* Transitions */
  --transition-fast: 150ms;
  --transition-base: 300ms;
  --transition-slow: 500ms;
}

/* Use variables with var() */
.button {
  background-color: var(--primary-color);
  color: var(--bg-color);
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--radius-md);
  font-family: var(--font-primary);
  transition: all var(--transition-base) ease;
}

/* Fallback values */
.element {
  color: var(--undefined-variable, #333);  /* Falls back to #333 */
}

/* Scoped variables */
.card {
  --card-bg: #fff;
  --card-padding: 20px;

  background-color: var(--card-bg);
  padding: var(--card-padding);
}

.card.dark {
  --card-bg: #222;  /* Override for dark variant */
}

/* Variables in calc() */
.responsive-spacing {
  margin: calc(var(--spacing-md) * 2);
  padding: calc(var(--spacing-sm) + 5px);
}

/* Theme switching with variables */
:root {
  --text: #333;
  --background: #fff;
}

[data-theme="dark"] {
  --text: #fff;
  --background: #222;
}

body {
  color: var(--text);
  background-color: var(--background);
  transition: background-color var(--transition-base), color var(--transition-base);
}

/* Dynamic variables with JavaScript */
/* JavaScript: document.documentElement.style.setProperty('--primary-color', '#ff0000'); */
```

---

## Modern CSS Features

### Logical Properties

Logical properties adapt to different writing modes and text directions.

```css
.logical-properties {
  /* Instead of left/right, use start/end */
  margin-inline-start: 20px;   /* Left in LTR, right in RTL */
  margin-inline-end: 20px;
  padding-inline: 20px;        /* Both start and end */

  /* Instead of top/bottom, use block-start/block-end */
  margin-block-start: 10px;
  margin-block-end: 10px;
  padding-block: 10px;

  /* Border */
  border-inline-start: 2px solid red;
  border-block-end: 1px solid blue;

  /* Width/height */
  inline-size: 300px;   /* Width in horizontal writing mode */
  block-size: 200px;    /* Height in horizontal writing mode */
}
```

### Clamp, Min, Max Functions

```css
.math-functions {
  /* clamp(min, preferred, max) - responsive sizing */
  font-size: clamp(16px, 4vw, 24px);
  width: clamp(300px, 50%, 800px);
  padding: clamp(1rem, 2vw, 3rem);

  /* min() - uses smallest value */
  width: min(90%, 1200px);
  font-size: min(5vw, 32px);

  /* max() - uses largest value */
  width: max(300px, 50%);
  font-size: max(16px, 1.5vw);
}
```

### Aspect Ratio

```css
.aspect-ratio {
  aspect-ratio: 16 / 9;   /* 16:9 aspect ratio */
  aspect-ratio: 1;        /* Square */
  aspect-ratio: 4 / 3;    /* 4:3 aspect ratio */
  width: 100%;            /* Width determines height via aspect ratio */
}
```

### Gap (for Flexbox and Grid)

```css
.gap-usage {
  display: flex;
  gap: 20px;           /* Space between flex items */
  row-gap: 10px;
  column-gap: 20px;
}

.grid-gap {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(3, 1fr);
}
```

### Object Fit and Object Position

```css
.image-container {
  width: 300px;
  height: 200px;
}

.image-container img {
  width: 100%;
  height: 100%;

  /* How image fits in container */
  object-fit: fill;       /* Default: stretch to fill */
  object-fit: contain;    /* Fit within container, maintain aspect ratio */
  object-fit: cover;      /* Fill container, maintain aspect ratio, crop if needed */
  object-fit: scale-down; /* Use contain or none, whichever is smaller */
  object-fit: none;       /* Original size */

  /* Position of image within container */
  object-position: center;
  object-position: top left;
  object-position: 50% 75%;
}
```

### Scroll Behavior

```css
html {
  scroll-behavior: smooth;  /* Smooth scrolling for anchor links */
}

.scroll-container {
  /* Scroll snap */
  scroll-snap-type: y mandatory;  /* Snap on y-axis */
  scroll-snap-type: x proximity;  /* Snap on x-axis when close */
  overflow-y: scroll;
}

.scroll-item {
  scroll-snap-align: start;  /* Snap to start of container */
  scroll-snap-align: center;
  scroll-snap-align: end;

  scroll-snap-stop: always;  /* Always stop at this element */
}

/* Scroll margin/padding */
.section {
  scroll-margin-top: 80px;  /* Offset for fixed header */
  scroll-padding-top: 80px;
}
```

---

## Common Patterns and Operations

### Centering Elements

```css
/* Horizontal centering */
.horizontal-center {
  /* Block element with width */
  margin: 0 auto;
  width: 80%;
}

/* Vertical and horizontal centering */

/* Method 1: Flexbox (recommended) */
.flex-center {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Method 2: Grid */
.grid-center {
  display: grid;
  place-items: center;
}

/* Method 3: Absolute positioning */
.absolute-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* Method 4: Table display */
.table-center {
  display: table;
  width: 100%;
}

.table-cell-center {
  display: table-cell;
  vertical-align: middle;
  text-align: center;
}

/* Text centering */
.text-center {
  text-align: center;
}

.vertical-text-center {
  line-height: 100px;  /* Same as height */
  height: 100px;
}
```

### Clearfix (for floats)

```css
/* Modern clearfix */
.clearfix::after {
  content: "";
  display: table;
  clear: both;
}

/* Usage */
.container {
  /* Contains floated children */
}

.container::after {
  content: "";
  display: table;
  clear: both;
}
```

### Truncate Text

```css
/* Single line truncation */
.truncate {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
}

/* Multi-line truncation (webkit only) */
.truncate-multiline {
  display: -webkit-box;
  -webkit-line-clamp: 3;  /* Number of lines */
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

### Overlay

```css
.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  z-index: 1000;
}
```

### Triangle with CSS

```css
.triangle-up {
  width: 0;
  height: 0;
  border-left: 10px solid transparent;
  border-right: 10px solid transparent;
  border-bottom: 10px solid red;
}

.triangle-down {
  width: 0;
  height: 0;
  border-left: 10px solid transparent;
  border-right: 10px solid transparent;
  border-top: 10px solid red;
}

.triangle-left {
  width: 0;
  height: 0;
  border-top: 10px solid transparent;
  border-bottom: 10px solid transparent;
  border-right: 10px solid red;
}

.triangle-right {
  width: 0;
  height: 0;
  border-top: 10px solid transparent;
  border-bottom: 10px solid transparent;
  border-left: 10px solid red;
}
```

### Sticky Footer

```css
/* Flexbox method (recommended) */
body {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

main {
  flex: 1;
}

/* Grid method */
body {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}
```

### Card Component Pattern

```css
.card {
  background-color: white;
  border-radius: var(--radius-md, 8px);
  box-shadow: var(--shadow-md, 0 4px 6px rgba(0, 0, 0, 0.1));
  padding: var(--spacing-lg, 24px);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg, 0 10px 15px rgba(0, 0, 0, 0.1));
}

.card-header {
  margin-bottom: var(--spacing-md, 16px);
  padding-bottom: var(--spacing-md, 16px);
  border-bottom: 1px solid #e0e0e0;
}

.card-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.card-body {
  margin-bottom: var(--spacing-md, 16px);
}

.card-footer {
  margin-top: var(--spacing-md, 16px);
  padding-top: var(--spacing-md, 16px);
  border-top: 1px solid #e0e0e0;
}
```

### Loading Spinner

```css
.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-left-color: #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Skeleton Loading

```css
.skeleton {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton-text {
  height: 16px;
  margin-bottom: 8px;
}

.skeleton-title {
  height: 24px;
  width: 60%;
  margin-bottom: 16px;
}
```

---

## Preprocessors

### Sass/SCSS

```scss
// Variables
$primary-color: #007bff;
$secondary-color: #6c757d;
$spacing-unit: 8px;

// Nesting
.nav {
  background-color: $primary-color;

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    display: inline-block;

    &:hover {  // & refers to parent selector
      background-color: darken($primary-color, 10%);
    }
  }

  a {
    color: white;
    text-decoration: none;
    padding: $spacing-unit * 2;

    &.active {
      font-weight: bold;
    }
  }
}

// Mixins
@mixin flex-center {
  display: flex;
  justify-content: center;
  align-items: center;
}

@mixin responsive($breakpoint) {
  @if $breakpoint == mobile {
    @media (max-width: 767px) { @content; }
  } @else if $breakpoint == tablet {
    @media (min-width: 768px) and (max-width: 1023px) { @content; }
  } @else if $breakpoint == desktop {
    @media (min-width: 1024px) { @content; }
  }
}

.container {
  @include flex-center;

  @include responsive(mobile) {
    flex-direction: column;
  }
}

// Functions
@function px-to-rem($px, $base: 16px) {
  @return ($px / $base) * 1rem;
}

.text {
  font-size: px-to-rem(18px);
}

// Extend/Inheritance
%button-base {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.button-primary {
  @extend %button-base;
  background-color: $primary-color;
  color: white;
}

.button-secondary {
  @extend %button-base;
  background-color: $secondary-color;
  color: white;
}

// Partials and imports
@import 'variables';
@import 'mixins';
@import 'base';
@import 'components/button';
@import 'components/card';

// Loops
@for $i from 1 through 5 {
  .margin-#{$i} {
    margin: #{$i * $spacing-unit};
  }
}

// Maps
$colors: (
  primary: #007bff,
  secondary: #6c757d,
  success: #28a745,
  danger: #dc3545
);

@each $name, $color in $colors {
  .btn-#{$name} {
    background-color: $color;
  }
}
```

### PostCSS

```javascript
// postcss.config.js
module.exports = {
  plugins: [
    require('autoprefixer'),      // Add vendor prefixes
    require('postcss-preset-env'), // Use modern CSS features
    require('cssnano'),            // Minify CSS
    require('postcss-nested'),     // Sass-like nesting
  ]
}
```

```css
/* PostCSS with future CSS syntax */
:root {
  --primary-color: #007bff;
}

.button {
  /* Nesting (with postcss-nested) */
  background-color: var(--primary-color);

  &:hover {
    background-color: color-mod(var(--primary-color) shade(10%));
  }

  /* Autoprefixer adds vendor prefixes automatically */
  display: flex;
  user-select: none;
}
```

---

## Best Practices

### Organization and Structure

```css
/* 1. Use a consistent organization pattern */

/* Variables/Custom Properties */
:root {
  --primary-color: #007bff;
}

/* Reset/Normalize */
*, *::before, *::after {
  box-sizing: border-box;
}

/* Base/Typography */
body {
  font-family: Arial, sans-serif;
  line-height: 1.6;
}

/* Layout */
.container { }
.grid { }
.flex { }

/* Components */
.button { }
.card { }
.nav { }

/* Utilities */
.text-center { }
.mt-4 { }
.hidden { }

/* Media Queries (mobile-first) */
@media (min-width: 768px) { }
```

### Naming Conventions

```css
/* BEM (Block Element Modifier) */
.block { }
.block__element { }
.block--modifier { }

.card { }
.card__header { }
.card__title { }
.card__body { }
.card--featured { }
.card--large { }

/* OOCSS (Object-Oriented CSS) */
/* Separate structure from skin */
.button { /* Structure */ }
.button-primary { /* Skin */ }
.button-large { /* Size */ }

/* Utility-first (like Tailwind) */
.flex { display: flex; }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
.p-4 { padding: 1rem; }
.mt-8 { margin-top: 2rem; }
```

### Performance Optimization

```css
/* 1. Minimize repaints and reflows */
/* Avoid changing layout properties in animations */
.efficient-animation {
  /* Good: only transform and opacity */
  transition: transform 0.3s, opacity 0.3s;
}

.inefficient-animation {
  /* Bad: causes layout recalculation */
  transition: width 0.3s, height 0.3s, top 0.3s;
}

/* 2. Use efficient selectors */
/* Good: simple selectors */
.button { }
.nav-item { }

/* Bad: overly specific, slow */
div.container > ul.list > li.item > a.link { }

/* 3. Avoid universal selector in complex selectors */
/* Bad */
.container * { }

/* 4. Use CSS containment for independent regions */
.widget {
  contain: layout style paint;
}

/* 5. Use will-change sparingly for upcoming animations */
.will-animate {
  will-change: transform;
}

.will-animate.animating {
  transform: scale(1.2);
}

/* Remove will-change after animation */
.will-animate.done {
  will-change: auto;
}

/* 6. Minimize expensive properties */
/* Expensive: box-shadow, filter, border-radius on large elements */
/* Use sparingly or in animations */

/* 7. Use content-visibility for off-screen content */
.off-screen-section {
  content-visibility: auto;
  contain-intrinsic-size: 0 500px;  /* Estimated height */
}
```

### Accessibility

```css
/* 1. Maintain sufficient color contrast */
.text {
  color: #333;  /* At least 4.5:1 contrast ratio with background */
}

/* 2. Don't rely solely on color */
.error {
  color: red;
  border-left: 4px solid red;  /* Visual indicator beyond color */
}

.error::before {
  content: "⚠ ";  /* Icon for additional context */
}

/* 3. Ensure focus visibility */
a:focus,
button:focus,
input:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

/* Custom focus styles */
.button:focus-visible {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

/* 4. Use :focus-visible to hide focus on mouse click */
.button:focus:not(:focus-visible) {
  outline: none;
}

/* 5. Ensure interactive elements are large enough */
.button {
  min-height: 44px;  /* Touch target size */
  min-width: 44px;
  padding: 12px 24px;
}

/* 6. Respect user preferences */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

@media (prefers-color-scheme: dark) {
  /* Dark mode styles */
}

@media (prefers-contrast: high) {
  /* High contrast styles */
}

/* 7. Hide elements properly */
.visually-hidden {
  /* Accessible to screen readers, visually hidden */
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* 8. Skip links for keyboard navigation */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: white;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

### Maintainability

```css
/* 1. Use CSS variables for reusable values */
:root {
  --spacing-unit: 8px;
  --primary-color: #007bff;
}

.button {
  padding: calc(var(--spacing-unit) * 2);
  background-color: var(--primary-color);
}

/* 2. Comment complex or non-obvious code */
.complex-layout {
  /* Using negative margin to offset parent padding */
  margin: calc(var(--spacing-unit) * -2);
}

/* 3. Group related properties */
.element {
  /* Positioning */
  position: relative;
  top: 0;
  left: 0;

  /* Box model */
  display: block;
  width: 100%;
  padding: 20px;
  margin: 10px 0;

  /* Typography */
  font-size: 16px;
  line-height: 1.5;
  color: #333;

  /* Visual */
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 4px;

  /* Misc */
  cursor: pointer;
  transition: all 0.3s ease;
}

/* 4. Avoid !important (use specificity properly) */
/* Bad */
.text {
  color: red !important;
}

/* Good: increase specificity instead */
.component .text {
  color: red;
}

/* !important is acceptable for utilities */
.hidden {
  display: none !important;
}

/* 5. Keep selectors shallow */
/* Bad: too specific, hard to override */
.header .nav .list .item .link { }

/* Good: use classes */
.nav-link { }
```

---

## Browser Compatibility

### Vendor Prefixes

```css
/* Modern approach: use autoprefixer */
/* Write standard CSS, autoprefixer adds prefixes */

.element {
  display: flex;
  user-select: none;
  transform: scale(1.5);
}

/* Autoprefixer output: */
.element {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-transform: scale(1.5);
  -ms-transform: scale(1.5);
  transform: scale(1.5);
}
```

### Feature Queries

```css
/* @supports - progressive enhancement */
.element {
  /* Fallback for older browsers */
  display: block;
}

@supports (display: grid) {
  .element {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
  }
}

/* Complex queries */
@supports (display: flex) and (gap: 20px) {
  .container {
    display: flex;
    gap: 20px;
  }
}

/* Not supported */
@supports not (display: grid) {
  .fallback-layout {
    display: flex;
  }
}

/* Selector support */
@supports selector(:has(*)) {
  .parent:has(.child) {
    background-color: yellow;
  }
}
```

### Fallbacks

```css
.element {
  /* Fallback for older browsers */
  background-color: #007bff;

  /* Modern syntax with fallback */
  background-color: rgba(0, 123, 255, 0.8);

  /* Multiple backgrounds with fallback */
  background: url('fallback.jpg');
  background: linear-gradient(to right, red, blue), url('image.jpg');
}

/* CSS Grid with Flexbox fallback */
.container {
  display: flex;  /* Fallback */
  flex-wrap: wrap;
}

@supports (display: grid) {
  .container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  }
}

/* Custom properties with fallback */
.text {
  color: #333;  /* Fallback */
  color: var(--text-color, #333);
}
```

---

## Debugging CSS

```css
/* 1. Border debug - visualize all elements */
* {
  outline: 1px solid red;
}

/* 2. Background debug - see element boundaries */
* {
  background-color: rgba(255, 0, 0, 0.1);
}

/* 3. Debug specific issues */
.debug-z-index {
  position: relative;
  z-index: 999999;
  background-color: yellow;
}

.debug-overflow {
  overflow: visible !important;
}

/* 4. Named grid lines for debugging */
.grid-debug {
  display: grid;
  grid-template-columns: [start] 1fr [middle] 1fr [end];
  grid-template-rows: [top] auto [center] auto [bottom];
}

/* 5. Use browser DevTools effectively */
/* - Inspect element
/* - Check computed styles
/* - Toggle properties on/off
/* - Edit styles live
/* - Check layout (box model, flex, grid)
```

---

## Common CSS Pitfalls

```css
/* 1. Margin collapse */
.parent {
  margin-bottom: 20px;
}

.child {
  margin-top: 30px;  /* Only 30px gap, not 50px! */
}

/* Fix: add padding or border to parent, or use flexbox/grid */
.parent {
  padding-top: 1px;  /* Prevents collapse */
}

/* 2. Percentage heights require parent height */
.parent {
  /* height: auto; (default) - child percentage height won't work */
  height: 500px;  /* Now child percentage works */
}

.child {
  height: 50%;  /* Now this works */
}

/* 3. Floats need clearing */
.container {
  /* Floated children don't contribute to parent height */
}

.container::after {
  content: "";
  display: table;
  clear: both;
}

/* 4. Z-index only works on positioned elements */
.element {
  z-index: 999;  /* Doesn't work without position */
}

.element {
  position: relative;  /* Now z-index works */
  z-index: 999;
}

/* 5. Inline elements ignore width/height */
span {
  width: 100px;  /* Ignored */
  height: 50px;  /* Ignored */
}

/* Fix: use inline-block or block */
span {
  display: inline-block;
  width: 100px;  /* Now works */
  height: 50px;
}

/* 6. Transform creates new stacking context */
.parent {
  position: relative;
  z-index: 1;
}

.child {
  position: absolute;
  transform: scale(1.1);  /* Creates new stacking context */
  z-index: 999;  /* Only relative to parent, not global */
}
```

---

## Resources and Tools

### Online Tools
- **Can I Use** (caniuse.com) - Browser compatibility
- **CSS Tricks** - Tutorials and references
- **MDN Web Docs** - Comprehensive documentation
- **CodePen** - Experiment and share CSS
- **CSS Grid Generator** - Visual grid layout builder
- **Flexbox Froggy** - Learn Flexbox interactively
- **Grid Garden** - Learn CSS Grid interactively
- **Coolors** - Color scheme generator
- **Google Fonts** - Free web fonts

### CSS Frameworks
- **Tailwind CSS** - Utility-first framework
- **Bootstrap** - Component library
- **Bulma** - Modern CSS framework
- **Foundation** - Responsive front-end framework

### Build Tools
- **PostCSS** - Transform CSS with JavaScript
- **Sass** - CSS preprocessor
- **Less** - CSS preprocessor
- **Autoprefixer** - Add vendor prefixes automatically
- **PurgeCSS** - Remove unused CSS

---

## Summary

CSS is a powerful language for styling web pages with:
- **Flexible selectors** for targeting elements
- **Box model** for understanding element sizing
- **Modern layouts** with Flexbox and Grid
- **Responsive design** via media queries
- **Animations and transitions** for interactivity
- **CSS variables** for maintainable code
- **Preprocessors** for advanced features
- **Best practices** for performance and accessibility

Master CSS by:
1. Understanding the cascade and specificity
2. Learning Flexbox and Grid thoroughly
3. Practicing responsive design patterns
4. Using CSS variables for maintainability
5. Following accessibility best practices
6. Optimizing for performance
7. Staying current with modern CSS features

CSS continues to evolve with new features like Container Queries, :has() selector, and more powerful layout capabilities. Regular practice and staying updated with modern techniques will make you proficient in creating beautiful, responsive, and performant web interfaces.
