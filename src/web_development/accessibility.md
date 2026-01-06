# Web Accessibility (A11y)

> **Domain:** Frontend Development, UX, Legal Compliance
> **Key Concepts:** WCAG, Semantic HTML, ARIA, Keyboard Navigation, Screen Readers

**Web Accessibility** (often abbreviated as **a11y**) is the practice of designing and developing websites so that people with disabilities can use them. Disabilities include visual, auditory, physical, speech, cognitive, and neurological disabilities.

---

## 1. The Business & Legal Case

*   **Inclusivity:** 15% of the world's population lives with some form of disability.
*   **SEO:** Search engines are essentially blind users. Semantic, accessible sites rank better.
*   **Legal:** Laws like the **ADA (Americans with Disabilities Act)** and **EAA (European Accessibility Act)** mandate compliance.

---

## 2. The Standard: WCAG

The **Web Content Accessibility Guidelines (WCAG)** are organized around four principles (**POUR**):

1.  **Perceivable:** Information must be presentable to users in ways they can perceive.
    *   *Example:* Alt text for images, captions for video.
2.  **Operable:** User interface components and navigation must be operable.
    *   *Example:* Keyboard accessibility (No mouse required).
3.  **Understandable:** Information and the operation of user interface must be understandable.
    *   *Example:* Consistent navigation, clear error messages.
4.  **Robust:** Content must be robust enough that it can be interpreted reliably by a wide variety of user agents (browsers, assistive tech).
    *   *Example:* Valid HTML parsing.

---

## 3. Semantic HTML (The First Line of Defense)

The best way to be accessible is to use the correct HTML tag for the job.

*   **Buttons vs. Divs:**
    *   *Bad:* `<div onclick="...">Click me</div>` (Not focusable, no screen reader announcement).
    *   *Good:* `<button>Click me</button>` (Focusable by Tab, Enter/Space to activate).
*   **Headings:** Use `<h1>` through `<h6>` strictly for structure, not for font sizing. Screen reader users jump between headings to scan content.
*   **Forms:** Always link `<label>` to `<input>`.
    ```html
    <label for="email">Email</label>
    <input id="email" type="email" />
    ```

---

## 4. ARIA (Accessible Rich Internet Applications)

If you *must* create a complex custom widget (like a custom dropdown or modal), HTML isn't enough. You need **WAI-ARIA**.

*   **Role:** Defines what an element is. `role="dialog"`, `role="tablist"`.
*   **State:** Defines current condition. `aria-expanded="true"`, `aria-checked="false"`.
*   **Property:** Defines nature. `aria-required="true"`, `aria-label="Close"`.

**The First Rule of ARIA:** Don't use ARIA. Use native HTML elements if they exist.

---

## 5. Common Patterns & Fixes

### 5.1. Focus Management
When a user opens a Modal, focus must move *inside* the modal. When they close it, focus must return to the *trigger button*.
*   **Focus Trapping:** Prevent the user from tabbing *out* of the modal while it is open.

### 5.2. Color Contrast
Text must differ sufficiently from the background.
*   **WCAG AA Ratio:** 4.5:1 for normal text.
*   **Tools:** Chrome DevTools -> CSS Overview.

### 5.3. Keyboard Navigation
*   **Tab Order:** Should follow the visual flow (Left to Right, Top to Bottom).
*   **Focus Visible:** Never do `outline: none` in CSS unless you replace it with a custom focus style. Users need to know where they are.

---

## 6. Testing Tools

1.  **Lighthouse:** Built into Chrome. Basic automated check.
2.  **axe DevTools:** Industry standard browser extension.
3.  **Screen Readers:**
    *   **NVDA:** Free, Windows. Highly recommended for testing.
    *   **VoiceOver:** Built-in on macOS.
