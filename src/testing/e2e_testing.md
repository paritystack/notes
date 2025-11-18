# End-to-End Testing

End-to-end (E2E) testing validates complete user workflows by simulating real user interactions with the application. These tests verify that all integrated components work together correctly from the user's perspective.

## Table of Contents

- [E2E Testing Fundamentals](#e2e-testing-fundamentals)
- [Playwright Deep Dive](#playwright-deep-dive)
- [Cypress Comparison](#cypress-comparison)
- [Test Organization Patterns](#test-organization-patterns)
- [Page Object Model](#page-object-model)
- [Test Data Management](#test-data-management)
- [Handling Authentication](#handling-authentication)
- [Dealing with Flaky Tests](#dealing-with-flaky-tests)
- [Visual Regression Testing](#visual-regression-testing)
- [Accessibility Testing](#accessibility-testing)
- [Cross-Browser Testing](#cross-browser-testing)
- [Mobile E2E Testing](#mobile-e2e-testing)
- [CI/CD Integration](#cicd-integration)
- [Best Practices and Anti-Patterns](#best-practices-and-anti-patterns)
- [Performance Considerations](#performance-considerations)
- [Debugging E2E Tests](#debugging-e2e-tests)

## E2E Testing Fundamentals

### What is E2E Testing?

E2E tests simulate real user scenarios by:
- Interacting with the UI like a real user
- Validating complete workflows from start to finish
- Testing the entire application stack (frontend, backend, database)
- Verifying integration between all system components

### When to Write E2E Tests

**Write E2E tests for:**
- Critical user journeys (signup, login, checkout)
- Revenue-generating workflows
- Complex multi-step processes
- Integration between major components
- Scenarios difficult to test at lower levels

**Avoid E2E tests for:**
- Edge cases (use unit tests instead)
- Simple logic (use integration tests)
- Every possible path (too slow and costly)

### Testing Pyramid

```
        E2E Tests (10%)
       /              \
      Integration (20%)
     /                  \
    Unit Tests (70%)
```

E2E tests should be:
- **Few**: Expensive to write and maintain
- **High-value**: Test critical user journeys
- **Stable**: Not flaky or brittle

### Key Concepts

**User Flows**: Complete sequences of actions
```
Login → Browse Products → Add to Cart → Checkout → Payment → Confirmation
```

**Test Isolation**: Each test should:
- Start with a clean state
- Not depend on other tests
- Clean up after itself

**Assertions**: Verify expected outcomes
- Page loaded correctly
- Elements visible/hidden
- Data saved to database
- Navigation occurred

## Playwright Deep Dive

Playwright is a modern, cross-browser automation framework built by Microsoft. It supports Chromium, Firefox, and WebKit with a single API.

### Installation and Setup

```bash
# Initialize Playwright project
npm init playwright@latest

# Or add to existing project
npm install -D @playwright/test

# Install browsers
npx playwright install
```

**Project Structure:**
```
my-project/
├── tests/
│   ├── auth.spec.ts
│   ├── checkout.spec.ts
│   └── search.spec.ts
├── playwright.config.ts
└── package.json
```

**Basic Configuration:**

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',

  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],

  webServer: {
    command: 'npm run start',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

### Writing Tests

**Basic Test Structure:**

```typescript
import { test, expect } from '@playwright/test';

test.describe('User Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Setup before each test
    await page.goto('/');
  });

  test('should login successfully', async ({ page }) => {
    // Arrange
    await page.goto('/login');

    // Act
    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // Assert
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toContainText('Welcome');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    await page.fill('[name="email"]', 'wrong@example.com');
    await page.fill('[name="password"]', 'wrongpass');
    await page.click('button[type="submit"]');

    const error = page.locator('.error-message');
    await expect(error).toBeVisible();
    await expect(error).toContainText('Invalid credentials');
  });
});
```

### Selectors and Locators

Playwright provides powerful selector strategies:

**CSS Selectors:**
```typescript
await page.locator('button').click();
await page.locator('.submit-button').click();
await page.locator('#login-form input[name="email"]').fill('test@example.com');
```

**Text Selectors:**
```typescript
await page.locator('text=Sign in').click();
await page.locator('button:has-text("Submit")').click();
```

**Role-based Selectors (Recommended):**
```typescript
// Most robust and accessible
await page.getByRole('button', { name: 'Sign in' }).click();
await page.getByRole('textbox', { name: 'Email' }).fill('test@example.com');
await page.getByRole('link', { name: 'About' }).click();
```

**Data-testid Selectors:**
```typescript
// HTML: <button data-testid="submit-btn">Submit</button>
await page.getByTestId('submit-btn').click();
```

**Label Selectors:**
```typescript
await page.getByLabel('Email address').fill('test@example.com');
await page.getByLabel('Password').fill('secret');
```

**Placeholder Selectors:**
```typescript
await page.getByPlaceholder('Enter your email').fill('test@example.com');
```

**Chaining Selectors:**
```typescript
// Find button inside a specific form
await page.locator('form#login').getByRole('button', { name: 'Submit' }).click();

// Find element with multiple conditions
await page.locator('button').filter({ hasText: 'Submit' }).filter({ has: page.locator('.icon') }).click();
```

**Best Practices for Selectors:**
1. Prefer role-based selectors (accessible and stable)
2. Use data-testid for complex components
3. Avoid CSS classes (change frequently)
4. Avoid XPath (hard to read and maintain)

### Auto-Waiting

Playwright automatically waits for elements to be actionable before performing actions.

**Actionable Checks:**
- Element is attached to DOM
- Element is visible
- Element is stable (not animating)
- Element receives events
- Element is enabled

```typescript
// Playwright waits automatically
await page.click('button'); // Waits for button to be clickable

// No need for manual waits
await page.fill('input', 'text'); // Waits for input to be ready
```

**Custom Waits:**

```typescript
// Wait for element to be visible
await page.locator('.modal').waitFor({ state: 'visible' });

// Wait for element to be hidden
await page.locator('.loader').waitFor({ state: 'hidden' });

// Wait for element to exist in DOM
await page.locator('.dynamic-content').waitFor({ state: 'attached' });

// Wait for specific condition
await page.waitForFunction(() => {
  return document.querySelectorAll('.list-item').length > 5;
});

// Wait for URL change
await page.waitForURL('**/dashboard');

// Wait for load state
await page.waitForLoadState('networkidle');
await page.waitForLoadState('domcontentloaded');
```

### Network Interception

Playwright can intercept, modify, and mock network requests.

**Monitoring Requests:**

```typescript
// Listen to all requests
page.on('request', request => {
  console.log('>>', request.method(), request.url());
});

// Listen to all responses
page.on('response', response => {
  console.log('<<', response.status(), response.url());
});

// Wait for specific request
const response = await page.waitForResponse(
  response => response.url().includes('/api/users') && response.status() === 200
);
```

**Mocking API Responses:**

```typescript
// Mock API endpoint
await page.route('**/api/users', route => {
  route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify([
      { id: 1, name: 'Test User' },
      { id: 2, name: 'Another User' }
    ])
  });
});

// Navigate to page (will use mocked data)
await page.goto('/users');
```

**Modifying Requests:**

```typescript
// Add authentication header
await page.route('**/api/**', route => {
  const headers = {
    ...route.request().headers(),
    'Authorization': 'Bearer fake-token'
  };
  route.continue({ headers });
});
```

**Blocking Resources:**

```typescript
// Block images and stylesheets for faster tests
await page.route('**/*.{png,jpg,jpeg,css}', route => route.abort());
```

**Advanced Network Interception:**

```typescript
// Simulate slow network
await page.route('**/api/**', async route => {
  await new Promise(resolve => setTimeout(resolve, 1000)); // 1s delay
  route.continue();
});

// Simulate network failure
await page.route('**/api/flaky-endpoint', route => {
  route.abort('failed');
});
```

### Screenshots and Videos

**Screenshots:**

```typescript
// Screenshot entire page
await page.screenshot({ path: 'screenshot.png' });

// Full page screenshot (scrolls automatically)
await page.screenshot({ path: 'fullpage.png', fullPage: true });

// Screenshot specific element
const element = page.locator('.header');
await element.screenshot({ path: 'header.png' });

// Screenshot to buffer
const buffer = await page.screenshot();

// Automatic screenshot on failure (in config)
use: {
  screenshot: 'only-on-failure',
}
```

**Videos:**

```typescript
// playwright.config.ts
use: {
  video: 'on', // 'off' | 'on' | 'retain-on-failure' | 'on-first-retry'
}

// Access video path in test
test('example', async ({ page }, testInfo) => {
  await page.goto('/');
  // ... test actions

  // Video path available after test
  const videoPath = await page.video()?.path();
  console.log(videoPath);
});
```

**Traces:**

Playwright traces provide a complete recording of test execution.

```typescript
// playwright.config.ts
use: {
  trace: 'on-first-retry', // 'off' | 'on' | 'retain-on-failure' | 'on-first-retry'
}

// View traces
// npx playwright show-trace trace.zip
```

Traces include:
- DOM snapshots
- Network activity
- Console logs
- Actions and timings
- Screenshots

## Cypress Comparison

### Architecture Differences

**Playwright:**
- Runs outside the browser
- Multi-browser support (Chromium, Firefox, WebKit)
- Language support: JavaScript, TypeScript, Python, .NET, Java
- True multi-tab support
- Better for browser automation

**Cypress:**
- Runs inside the browser
- Limited browser support (Chrome, Firefox, Edge)
- JavaScript/TypeScript only
- Single tab limitation
- Better developer experience

### Strengths and Weaknesses

**Playwright Strengths:**
- True cross-browser testing
- Multiple tabs and contexts
- Better for complex automation
- Faster execution
- Better mobile emulation
- Network interception without service workers

**Playwright Weaknesses:**
- Steeper learning curve
- Less mature ecosystem
- Fewer plugins

**Cypress Strengths:**
- Excellent developer experience
- Time-travel debugging
- Real-time reloading
- Better documentation
- Larger community
- Easier to learn

**Cypress Weaknesses:**
- Browser limitations
- No multi-tab support
- Slower for large test suites
- iframe limitations
- No multi-browser parallel execution

### When to Choose Cypress vs Playwright

**Choose Cypress when:**
- Team is familiar with JavaScript
- Developer experience is priority
- Testing single-page applications
- Need time-travel debugging
- Community plugins are important

**Choose Playwright when:**
- Need true cross-browser testing
- Testing complex multi-tab workflows
- Need mobile browser testing
- Performance is critical
- Prefer modern async/await syntax

### Code Comparison

**Login Test:**

```javascript
// Cypress
describe('Login', () => {
  it('logs in successfully', () => {
    cy.visit('/login');
    cy.get('[name="email"]').type('user@example.com');
    cy.get('[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();

    cy.url().should('include', '/dashboard');
    cy.get('h1').should('contain', 'Welcome');
  });
});

// Playwright
test('logs in successfully', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[name="email"]', 'user@example.com');
  await page.fill('[name="password"]', 'password123');
  await page.click('button[type="submit"]');

  await expect(page).toHaveURL(/.*dashboard/);
  await expect(page.locator('h1')).toContainText('Welcome');
});
```

## Test Organization Patterns

### File Organization

**By Feature:**
```
tests/
├── auth/
│   ├── login.spec.ts
│   ├── signup.spec.ts
│   ├── password-reset.spec.ts
│   └── logout.spec.ts
├── products/
│   ├── search.spec.ts
│   ├── filter.spec.ts
│   └── details.spec.ts
└── checkout/
    ├── cart.spec.ts
    ├── payment.spec.ts
    └── confirmation.spec.ts
```

**By User Journey:**
```
tests/
├── user-journeys/
│   ├── new-user-signup.spec.ts
│   ├── returning-user-purchase.spec.ts
│   └── admin-workflow.spec.ts
├── critical-paths/
│   ├── checkout.spec.ts
│   └── payment.spec.ts
└── edge-cases/
    └── error-handling.spec.ts
```

### Shared Setup

**Fixtures:**

```typescript
// fixtures/auth.ts
import { test as base } from '@playwright/test';

type AuthFixtures = {
  authenticatedPage: Page;
};

export const test = base.extend<AuthFixtures>({
  authenticatedPage: async ({ page }, use) => {
    // Login before test
    await page.goto('/login');
    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await page.waitForURL('**/dashboard');

    // Use authenticated page in test
    await use(page);

    // Cleanup (logout) after test
    await page.click('[data-testid="logout"]');
  },
});

// Use in tests
import { test } from './fixtures/auth';

test('access protected resource', async ({ authenticatedPage }) => {
  await authenticatedPage.goto('/profile');
  await expect(authenticatedPage.locator('h1')).toContainText('My Profile');
});
```

**Global Setup:**

```typescript
// global-setup.ts
import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Perform one-time setup
  await page.goto('http://localhost:3000/setup');
  await page.click('button[data-testid="init-db"]');

  await browser.close();
}

export default globalSetup;
```

```typescript
// playwright.config.ts
export default defineConfig({
  globalSetup: require.resolve('./global-setup'),
});
```

### Helper Functions

```typescript
// helpers/auth.ts
import { Page } from '@playwright/test';

export async function login(page: Page, email: string, password: string) {
  await page.goto('/login');
  await page.fill('[name="email"]', email);
  await page.fill('[name="password"]', password);
  await page.click('button[type="submit"]');
  await page.waitForURL('**/dashboard');
}

export async function logout(page: Page) {
  await page.click('[data-testid="user-menu"]');
  await page.click('[data-testid="logout"]');
  await page.waitForURL('**/login');
}

// Use in tests
import { login, logout } from './helpers/auth';

test('user workflow', async ({ page }) => {
  await login(page, 'user@example.com', 'password123');
  // ... test actions
  await logout(page);
});
```

## Page Object Model

Page Object Model (POM) encapsulates page structure and interactions, improving test maintainability.

### Basic Page Object

```typescript
// pages/LoginPage.ts
import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.getByLabel('Email');
    this.passwordInput = page.getByLabel('Password');
    this.submitButton = page.getByRole('button', { name: 'Sign in' });
    this.errorMessage = page.locator('.error-message');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async getErrorText() {
    return await this.errorMessage.textContent();
  }
}
```

**Using Page Objects:**

```typescript
import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/LoginPage';

test('login with valid credentials', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password123');

  await expect(page).toHaveURL(/.*dashboard/);
});

test('login with invalid credentials', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('wrong@example.com', 'wrongpass');

  await expect(loginPage.errorMessage).toBeVisible();
  const errorText = await loginPage.getErrorText();
  expect(errorText).toContain('Invalid credentials');
});
```

### Advanced Page Object

```typescript
// pages/DashboardPage.ts
import { Page, Locator } from '@playwright/test';

export class DashboardPage {
  readonly page: Page;
  readonly header: Locator;
  readonly userMenu: Locator;
  readonly notifications: Locator;

  constructor(page: Page) {
    this.page = page;
    this.header = page.locator('header');
    this.userMenu = page.getByTestId('user-menu');
    this.notifications = page.getByTestId('notifications');
  }

  async goto() {
    await this.page.goto('/dashboard');
  }

  async openUserMenu() {
    await this.userMenu.click();
  }

  async logout() {
    await this.openUserMenu();
    await this.page.getByRole('menuitem', { name: 'Logout' }).click();
  }

  async getNotificationCount(): Promise<number> {
    const badge = this.notifications.locator('.badge');
    const text = await badge.textContent();
    return parseInt(text || '0');
  }

  async clickNotification(index: number) {
    await this.notifications.click();
    await this.page.locator('.notification-item').nth(index).click();
  }
}
```

### Component Objects

For reusable components:

```typescript
// components/SearchComponent.ts
import { Locator, Page } from '@playwright/test';

export class SearchComponent {
  readonly container: Locator;
  readonly input: Locator;
  readonly submitButton: Locator;
  readonly results: Locator;

  constructor(page: Page, containerSelector: string = '.search-component') {
    this.container = page.locator(containerSelector);
    this.input = this.container.getByPlaceholder('Search...');
    this.submitButton = this.container.getByRole('button', { name: 'Search' });
    this.results = this.container.locator('.search-results');
  }

  async search(query: string) {
    await this.input.fill(query);
    await this.submitButton.click();
    await this.results.waitFor({ state: 'visible' });
  }

  async getResultCount(): Promise<number> {
    const items = this.results.locator('.result-item');
    return await items.count();
  }

  async clickResult(index: number) {
    await this.results.locator('.result-item').nth(index).click();
  }
}

// Use in page objects
import { SearchComponent } from '../components/SearchComponent';

export class ProductsPage {
  readonly page: Page;
  readonly search: SearchComponent;

  constructor(page: Page) {
    this.page = page;
    this.search = new SearchComponent(page, '.products-search');
  }

  async goto() {
    await this.page.goto('/products');
  }
}
```

## Test Data Management

### Fixtures and Seed Data

**JSON Fixtures:**

```typescript
// fixtures/users.json
{
  "validUser": {
    "email": "user@example.com",
    "password": "password123",
    "name": "Test User"
  },
  "adminUser": {
    "email": "admin@example.com",
    "password": "admin123",
    "role": "admin"
  }
}

// Use in tests
import users from './fixtures/users.json';

test('login as valid user', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[name="email"]', users.validUser.email);
  await page.fill('[name="password"]', users.validUser.password);
  await page.click('button[type="submit"]');
});
```

### Dynamic Test Data

**Faker for Random Data:**

```typescript
import { faker } from '@faker-js/faker';

test('user registration', async ({ page }) => {
  const user = {
    firstName: faker.person.firstName(),
    lastName: faker.person.lastName(),
    email: faker.internet.email(),
    password: faker.internet.password(),
  };

  await page.goto('/signup');
  await page.fill('[name="firstName"]', user.firstName);
  await page.fill('[name="lastName"]', user.lastName);
  await page.fill('[name="email"]', user.email);
  await page.fill('[name="password"]', user.password);
  await page.click('button[type="submit"]');

  await expect(page.locator('.success-message')).toBeVisible();
});
```

### Database Seeding

**Prisma Example:**

```typescript
// helpers/db.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function seedTestData() {
  await prisma.user.createMany({
    data: [
      { email: 'user1@example.com', name: 'User 1' },
      { email: 'user2@example.com', name: 'User 2' },
    ],
  });
}

export async function cleanDatabase() {
  await prisma.user.deleteMany();
  await prisma.order.deleteMany();
}

// Use in tests
import { seedTestData, cleanDatabase } from './helpers/db';

test.beforeEach(async () => {
  await cleanDatabase();
  await seedTestData();
});

test.afterEach(async () => {
  await cleanDatabase();
});
```

### API-based Data Setup

```typescript
// helpers/api.ts
export async function createUser(userData: any) {
  const response = await fetch('http://localhost:3000/api/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(userData),
  });
  return response.json();
}

export async function deleteUser(userId: string) {
  await fetch(`http://localhost:3000/api/users/${userId}`, {
    method: 'DELETE',
  });
}

// Use in tests
test('user profile', async ({ page }) => {
  // Setup: Create user via API
  const user = await createUser({
    email: 'test@example.com',
    name: 'Test User',
  });

  // Test: Navigate and verify
  await page.goto(`/users/${user.id}`);
  await expect(page.locator('h1')).toContainText(user.name);

  // Cleanup: Delete user via API
  await deleteUser(user.id);
});
```

## Handling Authentication

### Storage State

Save and reuse authentication state:

```typescript
// auth.setup.ts
import { test as setup } from '@playwright/test';

const authFile = 'playwright/.auth/user.json';

setup('authenticate', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[name="email"]', 'user@example.com');
  await page.fill('[name="password"]', 'password123');
  await page.click('button[type="submit"]');

  await page.waitForURL('**/dashboard');

  // Save signed-in state
  await page.context().storageState({ path: authFile });
});

// playwright.config.ts
export default defineConfig({
  projects: [
    // Setup project
    { name: 'setup', testMatch: /.*\.setup\.ts/ },

    // Authenticated tests
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        storageState: authFile,
      },
      dependencies: ['setup'],
    },
  ],
});

// Tests automatically use authenticated state
test('access protected page', async ({ page }) => {
  await page.goto('/profile'); // Already logged in
  await expect(page.locator('h1')).toContainText('My Profile');
});
```

### Multiple User Roles

```typescript
// Setup for different roles
const adminAuthFile = 'playwright/.auth/admin.json';
const userAuthFile = 'playwright/.auth/user.json';

setup('authenticate as admin', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[name="email"]', 'admin@example.com');
  await page.fill('[name="password"]', 'admin123');
  await page.click('button[type="submit"]');
  await page.context().storageState({ path: adminAuthFile });
});

setup('authenticate as user', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[name="email"]', 'user@example.com');
  await page.fill('[name="password"]', 'user123');
  await page.click('button[type="submit"]');
  await page.context().storageState({ path: userAuthFile });
});

// Configure projects
projects: [
  {
    name: 'admin-tests',
    use: { storageState: adminAuthFile },
    testMatch: /admin.*.spec.ts/,
  },
  {
    name: 'user-tests',
    use: { storageState: userAuthFile },
    testMatch: /user.*.spec.ts/,
  },
]
```

### Token-Based Authentication

```typescript
// For API token auth
test.use({
  extraHTTPHeaders: {
    'Authorization': 'Bearer your-token-here',
  },
});

// Or set cookies directly
test.beforeEach(async ({ context }) => {
  await context.addCookies([
    {
      name: 'auth_token',
      value: 'your-token-value',
      domain: 'localhost',
      path: '/',
      httpOnly: true,
      secure: false,
      sameSite: 'Lax',
    },
  ]);
});
```

## Dealing with Flaky Tests

### Common Causes

1. **Race Conditions**
   - Async operations not properly awaited
   - Elements not fully loaded before interaction

2. **Timing Issues**
   - Hard-coded waits
   - Network delays
   - Animation/transition timing

3. **Test Dependencies**
   - Tests depending on execution order
   - Shared state between tests

4. **External Dependencies**
   - Third-party APIs
   - Variable network conditions
   - Database state inconsistencies

5. **Non-deterministic Behavior**
   - Random data causing different outcomes
   - Time-based logic
   - Randomized UI elements

### Mitigation Strategies

**1. Use Proper Waits:**

```typescript
// Bad: Hard-coded wait
await page.waitForTimeout(5000);

// Good: Wait for specific condition
await page.waitForSelector('[data-testid="result"]');
await page.waitForLoadState('networkidle');

// Better: Wait for specific API response
await page.waitForResponse(response =>
  response.url().includes('/api/data') && response.status() === 200
);
```

**2. Ensure Element Stability:**

```typescript
// Wait for animations to complete
await page.locator('.modal').waitFor({ state: 'visible' });
await page.locator('.modal').evaluate(el => {
  return Promise.all(el.getAnimations().map(animation => animation.finished));
});

// Wait for element to stop moving
const element = page.locator('.draggable');
await element.waitFor({ state: 'visible' });

// Playwright auto-waits for stability
await element.click(); // Waits for element to stop animating
```

**3. Isolate Tests:**

```typescript
// Bad: Tests share state
test('create user', async ({ page }) => {
  // Creates user in database
});

test('verify user exists', async ({ page }) => {
  // Depends on previous test
});

// Good: Each test independent
test.beforeEach(async () => {
  await cleanDatabase();
  await seedTestData();
});

test('create user', async ({ page }) => {
  // Creates its own test data
});

test('verify user', async ({ page }) => {
  // Creates its own test data
});
```

**4. Mock Unstable Dependencies:**

```typescript
// Mock third-party API
await page.route('**/api/external/**', route => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ data: 'mocked response' }),
  });
});

// Mock time-based functionality
await page.addInitScript(() => {
  const now = new Date('2024-01-01T12:00:00Z').getTime();
  Date.now = () => now;
});
```

**5. Add Retry Logic:**

```typescript
// Retry assertion with custom timeout
await expect(async () => {
  const count = await page.locator('.item').count();
  expect(count).toBeGreaterThan(0);
}).toPass({
  timeout: 10000,
  intervals: [1000, 2000, 3000],
});

// Retry flaky action
async function retryClick(locator: Locator, maxAttempts = 3) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      await locator.click({ timeout: 5000 });
      return;
    } catch (error) {
      if (i === maxAttempts - 1) throw error;
      await page.waitForTimeout(1000);
    }
  }
}
```

### Retry Strategies

**Test-Level Retries:**

```typescript
// playwright.config.ts
export default defineConfig({
  retries: process.env.CI ? 2 : 0, // Retry twice in CI
});

// Per-test retries
test('flaky test', async ({ page }) => {
  test.info().annotations.push({ type: 'issue', description: 'Flaky test' });
  // test code
});
```

**Custom Retry Logic:**

```typescript
async function waitForCondition(
  condition: () => Promise<boolean>,
  options: { timeout?: number; interval?: number } = {}
) {
  const { timeout = 30000, interval = 1000 } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error('Condition not met within timeout');
}

// Usage
await waitForCondition(async () => {
  const count = await page.locator('.item').count();
  return count > 0;
}, { timeout: 10000, interval: 500 });
```

**Exponential Backoff:**

```typescript
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      const delay = baseDelay * Math.pow(2, i);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error('Max retries exceeded');
}

// Usage
await retryWithBackoff(async () => {
  await page.click('[data-testid="submit"]');
  await expect(page.locator('.success')).toBeVisible({ timeout: 5000 });
});
```

## Visual Regression Testing

Visual regression testing captures screenshots and compares them against baselines to detect unintended UI changes.

### Playwright Visual Comparisons

```typescript
// Basic screenshot comparison
test('visual regression', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot('homepage.png');
});

// Element screenshot comparison
test('button visual', async ({ page }) => {
  await page.goto('/components');
  const button = page.getByRole('button', { name: 'Submit' });
  await expect(button).toHaveScreenshot('submit-button.png');
});

// With threshold for minor differences
test('with threshold', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot('homepage.png', {
    maxDiffPixels: 100, // Allow up to 100 different pixels
  });
});

// Full page screenshot
test('full page', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot('fullpage.png', {
    fullPage: true,
  });
});

// Mask dynamic elements
test('with masks', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveScreenshot('dashboard.png', {
    mask: [page.locator('.timestamp'), page.locator('.user-avatar')],
  });
});
```

### Percy Integration

Percy provides visual testing as a service with better diff tools.

```bash
npm install --save-dev @percy/cli @percy/playwright
```

```typescript
import percySnapshot from '@percy/playwright';

test('percy snapshot', async ({ page }) => {
  await page.goto('/');
  await percySnapshot(page, 'Homepage');
});

test('responsive snapshots', async ({ page }) => {
  await page.goto('/');
  await percySnapshot(page, 'Homepage', {
    widths: [375, 768, 1280],
  });
});
```

### Managing Visual Tests

**Update Baselines:**

```bash
# Update all screenshots
npx playwright test --update-snapshots

# Update specific test
npx playwright test visual.spec.ts --update-snapshots
```

**Organize Screenshots:**

```typescript
// Custom snapshot path
test('homepage', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot({
    name: 'homepage/desktop.png',
  });
});

// Platform-specific snapshots
test('cross-platform', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot(); // Automatically organized by platform
});
```

## Accessibility Testing

### Axe-Core Integration

```bash
npm install --save-dev @axe-core/playwright
```

```typescript
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('should not have accessibility violations', async ({ page }) => {
  await page.goto('/');

  const accessibilityScanResults = await new AxeBuilder({ page })
    .analyze();

  expect(accessibilityScanResults.violations).toEqual([]);
});

// Test specific regions
test('header accessibility', async ({ page }) => {
  await page.goto('/');

  const results = await new AxeBuilder({ page })
    .include('header')
    .analyze();

  expect(results.violations).toEqual([]);
});

// Disable specific rules
test('with exceptions', async ({ page }) => {
  await page.goto('/');

  const results = await new AxeBuilder({ page })
    .disableRules(['color-contrast']) // Disable specific rule
    .analyze();

  expect(results.violations).toEqual([]);
});

// Test specific WCAG level
test('WCAG AA compliance', async ({ page }) => {
  await page.goto('/');

  const results = await new AxeBuilder({ page })
    .withTags(['wcag2aa'])
    .analyze();

  expect(results.violations).toEqual([]);
});
```

### Keyboard Navigation

```typescript
test('keyboard navigation', async ({ page }) => {
  await page.goto('/');

  // Tab through interactive elements
  await page.keyboard.press('Tab');
  await expect(page.locator(':focus')).toHaveAttribute('href', '/about');

  await page.keyboard.press('Tab');
  await expect(page.locator(':focus')).toHaveAttribute('href', '/contact');

  // Press Enter to activate
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/.*contact/);
});

test('escape key closes modal', async ({ page }) => {
  await page.goto('/');
  await page.click('[data-testid="open-modal"]');

  await expect(page.locator('.modal')).toBeVisible();

  await page.keyboard.press('Escape');

  await expect(page.locator('.modal')).not.toBeVisible();
});
```

### Screen Reader Testing

```typescript
test('aria labels', async ({ page }) => {
  await page.goto('/');

  // Check for proper ARIA labels
  const searchButton = page.getByRole('button', { name: 'Search' });
  await expect(searchButton).toHaveAttribute('aria-label', 'Search');

  // Check for descriptive text
  const menuButton = page.getByRole('button', { name: /menu/i });
  await expect(menuButton).toHaveAttribute('aria-expanded', 'false');

  await menuButton.click();
  await expect(menuButton).toHaveAttribute('aria-expanded', 'true');
});

test('live regions', async ({ page }) => {
  await page.goto('/');

  const liveRegion = page.locator('[aria-live="polite"]');
  await expect(liveRegion).toBeEmpty();

  await page.click('[data-testid="trigger-notification"]');

  await expect(liveRegion).toContainText('Action completed');
});
```

## Cross-Browser Testing

### Configuration

```typescript
// playwright.config.ts
export default defineConfig({
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'edge',
      use: { ...devices['Desktop Edge'], channel: 'msedge' },
    },
  ],
});
```

### Run Specific Browsers

```bash
# Run all browsers
npx playwright test

# Run specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox

# Run multiple specific browsers
npx playwright test --project=chromium --project=firefox
```

### Browser-Specific Tests

```typescript
test('chromium-only feature', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'Chromium-only feature');

  await page.goto('/');
  // Test chromium-specific feature
});

test.describe('cross-browser tests', () => {
  test('works in all browsers', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
  });
});

// Browser-specific configuration
test.use({
  ...test.use(),
  ...(browserName === 'webkit' && { locale: 'en-US' }),
});
```

### Handle Browser Differences

```typescript
test('handle browser differences', async ({ page, browserName }) => {
  await page.goto('/');

  if (browserName === 'webkit') {
    // Safari-specific workaround
    await page.waitForTimeout(1000);
  }

  await page.click('[data-testid="button"]');
  await expect(page.locator('.result')).toBeVisible();
});
```

## Mobile E2E Testing

### Mobile Device Emulation

```typescript
// playwright.config.ts
import { devices } from '@playwright/test';

export default defineConfig({
  projects: [
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 13'] },
    },
    {
      name: 'Tablet',
      use: { ...devices['iPad Pro'] },
    },
  ],
});
```

### Custom Mobile Configuration

```typescript
test.use({
  viewport: { width: 375, height: 667 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
  userAgent: 'Mozilla/5.0...',
});

test('mobile layout', async ({ page }) => {
  await page.goto('/');

  // Mobile menu should be visible
  await expect(page.locator('.mobile-menu-button')).toBeVisible();

  // Desktop menu should be hidden
  await expect(page.locator('.desktop-menu')).not.toBeVisible();
});
```

### Touch Gestures

```typescript
test('swipe gesture', async ({ page }) => {
  await page.goto('/gallery');

  const carousel = page.locator('.carousel');

  // Swipe left
  await carousel.hover();
  await page.mouse.down();
  await page.mouse.move(100, 0);
  await page.mouse.up();

  await expect(page.locator('.carousel-item-2')).toBeVisible();
});

test('tap gesture', async ({ page }) => {
  await page.goto('/');

  // Tap element
  await page.locator('[data-testid="button"]').tap();

  await expect(page.locator('.result')).toBeVisible();
});

test('pinch zoom', async ({ page }) => {
  await page.goto('/image');

  const image = page.locator('img');

  // Simulate pinch zoom
  await page.touchscreen.tap(100, 100);
  // Pinch gesture implementation varies by framework
});
```

### Orientation Testing

```typescript
test('landscape orientation', async ({ page }) => {
  await page.setViewportSize({ width: 667, height: 375 });

  await page.goto('/');

  await expect(page.locator('.landscape-layout')).toBeVisible();
});

test('portrait orientation', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });

  await page.goto('/');

  await expect(page.locator('.portrait-layout')).toBeVisible();
});
```

### Responsive Testing

```typescript
const viewports = [
  { name: 'mobile', width: 375, height: 667 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1920, height: 1080 },
];

for (const viewport of viewports) {
  test(`responsive design on ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto('/');

    // Verify layout adapts correctly
    const screenshot = await page.screenshot();
    expect(screenshot).toMatchSnapshot(`${viewport.name}-layout.png`);
  });
}
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/playwright.yml
name: Playwright Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    timeout-minutes: 60
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 18

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright Browsers
        run: npx playwright install --with-deps

      - name: Run Playwright tests
        run: npx playwright test

      - name: Upload Playwright Report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: test-results/
```

### Running in Docker

**Dockerfile:**

```dockerfile
FROM mcr.microsoft.com/playwright:v1.40.0-jammy

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Run tests
CMD ["npx", "playwright", "test"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  playwright:
    build: .
    volumes:
      - ./test-results:/app/test-results
      - ./playwright-report:/app/playwright-report
    environment:
      - CI=true
      - BASE_URL=http://web:3000
    depends_on:
      - web

  web:
    image: nginx:alpine
    ports:
      - "3000:80"
    volumes:
      - ./public:/usr/share/nginx/html
```

**Run tests:**

```bash
# Build and run
docker-compose up --abort-on-container-exit

# Run specific tests
docker-compose run playwright npx playwright test auth.spec.ts
```

### Parallelization

**Configuration:**

```typescript
// playwright.config.ts
export default defineConfig({
  workers: process.env.CI ? 4 : undefined,
  fullyParallel: true,

  // Shard tests across multiple machines
  shard: process.env.CI ? {
    current: parseInt(process.env.SHARD_INDEX || '1'),
    total: parseInt(process.env.SHARD_TOTAL || '1'),
  } : undefined,
});
```

**GitHub Actions Sharding:**

```yaml
jobs:
  test:
    strategy:
      matrix:
        shardIndex: [1, 2, 3, 4]
        shardTotal: [4]
    steps:
      - name: Run Playwright tests
        run: npx playwright test
        env:
          SHARD_INDEX: ${{ matrix.shardIndex }}
          SHARD_TOTAL: ${{ matrix.shardTotal }}
```

### Test Reporting

**HTML Reporter:**

```typescript
// playwright.config.ts
export default defineConfig({
  reporter: [
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ['json', { outputFile: 'test-results.json' }],
    ['junit', { outputFile: 'test-results.xml' }],
  ],
});
```

**Custom Reporter:**

```typescript
// custom-reporter.ts
import { Reporter, TestCase, TestResult } from '@playwright/test/reporter';

class CustomReporter implements Reporter {
  onTestEnd(test: TestCase, result: TestResult) {
    console.log(`Test: ${test.title}`);
    console.log(`Status: ${result.status}`);
    console.log(`Duration: ${result.duration}ms`);
  }

  onEnd() {
    console.log('All tests completed');
  }
}

export default CustomReporter;
```

**Use custom reporter:**

```typescript
// playwright.config.ts
export default defineConfig({
  reporter: [
    ['./custom-reporter.ts'],
    ['html'],
  ],
});
```

## Best Practices and Anti-Patterns

### Best Practices

**1. Use Semantic Selectors:**

```typescript
// Good
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByLabel('Email').fill('test@example.com');
await page.getByText('Welcome back').click();

// Avoid
await page.click('.btn-primary-submit-form-action');
await page.fill('#input_23847');
```

**2. Keep Tests Independent:**

```typescript
// Good
test('create user', async ({ page }) => {
  await createUser();
  await verifyUserCreated();
  await deleteUser();
});

// Bad
test('create user', async ({ page }) => {
  await createUser();
});

test('verify user exists', async ({ page }) => {
  await verifyUserCreated(); // Depends on previous test
});
```

**3. Use Page Object Model:**

```typescript
// Good
const loginPage = new LoginPage(page);
await loginPage.login('user@example.com', 'password');

// Avoid
await page.fill('#email', 'user@example.com');
await page.fill('#password', 'password');
await page.click('.login-button');
```

**4. Avoid Hard-Coded Waits:**

```typescript
// Good
await page.waitForSelector('[data-testid="result"]');
await page.waitForLoadState('networkidle');

// Bad
await page.waitForTimeout(3000);
```

**5. Test User Journeys, Not Implementation:**

```typescript
// Good - Tests user behavior
test('user can purchase product', async ({ page }) => {
  await page.goto('/products');
  await page.click('[data-testid="product-1"]');
  await page.click('[data-testid="add-to-cart"]');
  await page.click('[data-testid="checkout"]');
  await fillPaymentDetails(page);
  await page.click('[data-testid="complete-order"]');

  await expect(page.locator('.success-message')).toBeVisible();
});

// Bad - Tests implementation details
test('cart state updates correctly', async ({ page }) => {
  await page.evaluate(() => {
    window.store.dispatch({ type: 'ADD_TO_CART', payload: { id: 1 } });
  });
  // Testing internals, not user behavior
});
```

### Anti-Patterns

**1. Over-Reliance on XPath:**

```typescript
// Bad
await page.click('//div[@class="container"]//button[contains(@class, "submit")]');

// Good
await page.getByRole('button', { name: 'Submit' }).click();
```

**2. Testing Too Many Scenarios in One Test:**

```typescript
// Bad
test('user journey', async ({ page }) => {
  // Test signup
  // Test login
  // Test profile update
  // Test password change
  // Test logout
  // 100+ lines of test code
});

// Good - Split into focused tests
test('user can signup', async ({ page }) => { /* ... */ });
test('user can login', async ({ page }) => { /* ... */ });
test('user can update profile', async ({ page }) => { /* ... */ });
```

**3. Not Cleaning Up Test Data:**

```typescript
// Bad
test('create user', async ({ page }) => {
  await createUser();
  // No cleanup - pollutes database
});

// Good
test('create user', async ({ page }) => {
  const user = await createUser();

  // Test code

  await deleteUser(user.id);
});
```

**4. Coupling Tests to CSS Classes:**

```typescript
// Bad - Breaks when styles change
await page.click('.btn-blue-rounded-lg-submit');

// Good - Uses semantic selectors
await page.getByRole('button', { name: 'Submit' }).click();
```

**5. Not Using Auto-Waiting:**

```typescript
// Bad
await page.waitForTimeout(1000);
await page.click('button');

// Good - Playwright waits automatically
await page.click('button');
```

## Performance Considerations

### Test Execution Speed

**1. Parallelize Tests:**

```typescript
// playwright.config.ts
export default defineConfig({
  fullyParallel: true,
  workers: 4, // Run 4 tests concurrently
});
```

**2. Reuse Browser Contexts:**

```typescript
// Slow - New browser per test
test('test 1', async ({ page }) => { /* ... */ });
test('test 2', async ({ page }) => { /* ... */ });

// Faster - Reuse browser, new context per test (default)
test.describe.configure({ mode: 'parallel' });
```

**3. Share Authentication State:**

```typescript
// Slow - Login in every test
test('test 1', async ({ page }) => {
  await login(page);
  // test code
});

// Fast - Login once, reuse state
test.use({ storageState: 'auth.json' });
test('test 1', async ({ page }) => {
  // Already logged in
});
```

**4. Mock Slow External Services:**

```typescript
// Slow - Real API calls
test('test', async ({ page }) => {
  await page.goto('/'); // Calls real APIs
});

// Fast - Mocked responses
test('test', async ({ page }) => {
  await page.route('**/api/slow-endpoint', route => {
    route.fulfill({ body: JSON.stringify({ data: 'mocked' }) });
  });
  await page.goto('/');
});
```

### Resource Optimization

**Block Unnecessary Resources:**

```typescript
test.beforeEach(async ({ page }) => {
  // Block images, fonts, analytics
  await page.route('**/*.{png,jpg,jpeg,gif,svg,woff,woff2}', route => route.abort());
  await page.route('**/analytics.js', route => route.abort());
  await page.route('**/tracking/**', route => route.abort());
});
```

**Use Headed Mode Sparingly:**

```typescript
// Slow - Opens visible browser
npx playwright test --headed

// Fast - Headless by default
npx playwright test
```

### Monitoring Performance

```typescript
test('measure page load time', async ({ page }) => {
  const startTime = Date.now();

  await page.goto('/');
  await page.waitForLoadState('networkidle');

  const loadTime = Date.now() - startTime;
  console.log(`Page load time: ${loadTime}ms`);

  expect(loadTime).toBeLessThan(3000); // Assert performance
});

test('track specific metrics', async ({ page }) => {
  await page.goto('/');

  const metrics = await page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    return {
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
      loadComplete: navigation.loadEventEnd - navigation.fetchStart,
      firstPaint: performance.getEntriesByType('paint')[0]?.startTime,
    };
  });

  console.log(metrics);
  expect(metrics.loadComplete).toBeLessThan(5000);
});
```

## Debugging E2E Tests

### Playwright Inspector

```bash
# Run with debugger
PWDEBUG=1 npx playwright test

# Debug specific test
npx playwright test --debug test-name.spec.ts

# Debug from specific line
# Add: await page.pause();
test('debug test', async ({ page }) => {
  await page.goto('/');
  await page.pause(); // Debugger opens here
  await page.click('button');
});
```

### UI Mode

```bash
# Run tests in UI mode
npx playwright test --ui

# Features:
# - Watch mode
# - Time travel
# - Pick locator
# - View traces
# - Edit tests
```

### VS Code Debugger

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Playwright Test",
      "program": "${workspaceFolder}/node_modules/@playwright/test/cli.js",
      "args": ["test", "--headed", "${file}"],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

### Console Logs

```typescript
// Capture browser console
page.on('console', msg => {
  console.log(`Browser log: ${msg.type()} ${msg.text()}`);
});

// Capture page errors
page.on('pageerror', error => {
  console.log(`Page error: ${error.message}`);
});

// Capture network failures
page.on('requestfailed', request => {
  console.log(`Failed request: ${request.url()}`);
});
```

### Screenshots and Videos

```typescript
// Take screenshot on failure
test('example', async ({ page }, testInfo) => {
  try {
    await page.goto('/');
    // test code
  } catch (error) {
    await page.screenshot({
      path: `failures/${testInfo.title}.png`,
      fullPage: true
    });
    throw error;
  }
});

// Automatic screenshots (in config)
use: {
  screenshot: 'only-on-failure',
  video: 'retain-on-failure',
}
```

### Trace Viewer

```bash
# Record trace
npx playwright test --trace on

# View trace
npx playwright show-trace trace.zip
```

Trace includes:
- Full timeline
- DOM snapshots
- Network activity
- Console logs
- Source code
- Screenshots

### Verbose Logging

```bash
# Debug mode
DEBUG=pw:api npx playwright test

# More verbose
DEBUG=pw:* npx playwright test

# Specific module
DEBUG=pw:browser npx playwright test
```

### Common Debugging Patterns

```typescript
// 1. Slow down execution
test.use({
  launchOptions: {
    slowMo: 1000 // 1 second delay between actions
  }
});

// 2. Keep browser open on failure
test.use({
  launchOptions: {
    headless: false,
    devtools: true
  }
});

// 3. Log element state
const button = page.locator('button');
console.log('Visible:', await button.isVisible());
console.log('Enabled:', await button.isEnabled());
console.log('Text:', await button.textContent());

// 4. Wait and inspect
await page.pause(); // Opens inspector
await page.waitForTimeout(5000); // Manual wait to inspect

// 5. Take intermediate screenshots
await page.screenshot({ path: 'step1.png' });
// ... actions ...
await page.screenshot({ path: 'step2.png' });
```

## Quick Reference

### Tool Comparison

| Tool | Language | Browsers | Best For |
|------|----------|----------|----------|
| Playwright | JS/TS/Python/.NET/Java | Chrome, Firefox, Safari | Modern cross-browser testing |
| Cypress | JavaScript | Chrome, Firefox, Edge | Developer experience, SPAs |
| Selenium | Multi-language | All major browsers | Legacy support, multi-language |
| Puppeteer | JavaScript | Chrome only | Chrome automation, scraping |

### Common Commands

```bash
# Playwright
npx playwright test                    # Run all tests
npx playwright test --headed          # Run with visible browser
npx playwright test --debug           # Debug mode
npx playwright test --ui              # UI mode
npx playwright show-report            # Show HTML report
npx playwright codegen                # Record tests

# Cypress
npx cypress open                      # Open Test Runner
npx cypress run                       # Run headless
npx cypress run --headed              # Run headed
npx cypress run --browser chrome      # Specific browser
```

### Best Practices Summary

1. Use semantic selectors (roles, labels)
2. Keep tests independent and isolated
3. Avoid hard-coded waits
4. Use Page Object Model for complex apps
5. Test user journeys, not implementation
6. Mock external dependencies
7. Clean up test data
8. Run tests in parallel
9. Use proper authentication state management
10. Monitor and fix flaky tests

## Further Resources

- [Playwright Documentation](https://playwright.dev/)
- [Cypress Documentation](https://docs.cypress.io/)
- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [Testing Best Practices](https://testingjavascript.com/)
- [Web Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Martin Fowler - Testing](https://martinfowler.com/testing/)
