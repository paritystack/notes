# End-to-End Testing

End-to-end (E2E) testing validates complete user workflows by simulating real user interactions with the application. These tests verify that all integrated components work together correctly from the user's perspective.

## Overview

E2E tests validate:
- Complete user workflows and journeys
- Cross-browser compatibility
- UI interactions and behavior
- Integration of frontend, backend, and databases
- Third-party service integrations
- Critical business processes

## Popular E2E Testing Tools

### Playwright
Modern, cross-browser automation framework by Microsoft.

```javascript
// tests/e2e/login.spec.js
const { test, expect } = require('@playwright/test');

test('user can login successfully', async ({ page }) => {
  // Navigate to login page
  await page.goto('https://example.com/login');

  // Fill in credentials
  await page.fill('input[name="email"]', 'user@example.com');
  await page.fill('input[name="password"]', 'password123');

  // Click login button
  await page.click('button[type="submit"]');

  // Verify redirect to dashboard
  await expect(page).toHaveURL('https://example.com/dashboard');
  await expect(page.locator('h1')).toContainText('Welcome');
});

test('shows error for invalid credentials', async ({ page }) => {
  await page.goto('https://example.com/login');
  await page.fill('input[name="email"]', 'wrong@example.com');
  await page.fill('input[name="password"]', 'wrongpass');
  await page.click('button[type="submit"]');

  await expect(page.locator('.error-message')).toBeVisible();
  await expect(page.locator('.error-message')).toContainText('Invalid credentials');
});
```

### Cypress
Developer-friendly E2E testing framework with time-travel debugging.

```javascript
// cypress/e2e/checkout.cy.js
describe('E-commerce Checkout Flow', () => {
  beforeEach(() => {
    cy.visit('/products');
  });

  it('completes full checkout process', () => {
    // Add product to cart
    cy.get('[data-testid="product-1"]').click();
    cy.get('[data-testid="add-to-cart"]').click();
    cy.get('[data-testid="cart-count"]').should('contain', '1');

    // Go to cart
    cy.get('[data-testid="cart-icon"]').click();
    cy.url().should('include', '/cart');

    // Proceed to checkout
    cy.get('[data-testid="checkout-button"]').click();

    // Fill shipping information
    cy.get('input[name="firstName"]').type('John');
    cy.get('input[name="lastName"]').type('Doe');
    cy.get('input[name="address"]').type('123 Main St');
    cy.get('input[name="city"]').type('New York');
    cy.get('select[name="state"]').select('NY');
    cy.get('input[name="zip"]').type('10001');

    // Fill payment information
    cy.get('input[name="cardNumber"]').type('4242424242424242');
    cy.get('input[name="expiry"]').type('12/25');
    cy.get('input[name="cvv"]').type('123');

    // Submit order
    cy.get('[data-testid="place-order"]').click();

    // Verify success
    cy.url().should('include', '/confirmation');
    cy.get('[data-testid="order-success"]').should('be.visible');
    cy.get('[data-testid="order-number"]').should('exist');
  });

  it('validates required fields', () => {
    cy.get('[data-testid="product-1"]').click();
    cy.get('[data-testid="add-to-cart"]').click();
    cy.get('[data-testid="cart-icon"]').click();
    cy.get('[data-testid="checkout-button"]').click();

    // Try to submit without filling fields
    cy.get('[data-testid="place-order"]').click();

    // Verify validation errors
    cy.get('.field-error').should('have.length.greaterThan', 0);
  });
});
```

### Selenium
Cross-language web automation framework.

```python
# tests/e2e/test_search.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_search_functionality():
    driver = webdriver.Chrome()
    try:
        # Navigate to homepage
        driver.get('https://example.com')

        # Find search box and enter query
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys('test query')
        search_box.submit()

        # Wait for results
        wait = WebDriverWait(driver, 10)
        results = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'search-results'))
        )

        # Verify results exist
        assert results.is_displayed()
        items = driver.find_elements(By.CLASS_NAME, 'result-item')
        assert len(items) > 0
    finally:
        driver.quit()
```

### Puppeteer
Node library for controlling headless Chrome.

```javascript
// tests/e2e/navigation.test.js
const puppeteer = require('puppeteer');

describe('Navigation Tests', () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    page = await browser.newPage();
  });

  test('navigation menu works correctly', async () => {
    await page.goto('https://example.com');

    // Click on Products menu
    await page.click('nav a[href="/products"]');
    await page.waitForNavigation();

    expect(page.url()).toContain('/products');

    // Verify page content
    const heading = await page.$eval('h1', el => el.textContent);
    expect(heading).toBe('Our Products');
  });

  test('mobile menu works', async () => {
    await page.setViewport({ width: 375, height: 667 });
    await page.goto('https://example.com');

    // Open mobile menu
    await page.click('[data-testid="mobile-menu-button"]');

    // Wait for menu to be visible
    await page.waitForSelector('.mobile-menu', { visible: true });

    // Click menu item
    await page.click('.mobile-menu a[href="/about"]');
    await page.waitForNavigation();

    expect(page.url()).toContain('/about');
  });
});
```

## Page Object Model

Organize tests using the Page Object Model pattern for better maintainability.

```javascript
// pages/LoginPage.js
class LoginPage {
  constructor(page) {
    this.page = page;
    this.emailInput = 'input[name="email"]';
    this.passwordInput = 'input[name="password"]';
    this.submitButton = 'button[type="submit"]';
    this.errorMessage = '.error-message';
  }

  async navigate() {
    await this.page.goto('https://example.com/login');
  }

  async login(email, password) {
    await this.page.fill(this.emailInput, email);
    await this.page.fill(this.passwordInput, password);
    await this.page.click(this.submitButton);
  }

  async getErrorMessage() {
    return await this.page.textContent(this.errorMessage);
  }
}

// tests/auth.spec.js
const { test, expect } = require('@playwright/test');
const { LoginPage } = require('../pages/LoginPage');

test('login with valid credentials', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.navigate();
  await loginPage.login('user@example.com', 'password123');

  await expect(page).toHaveURL(/.*dashboard/);
});
```

## Common Test Patterns

### Waiting for Elements

```javascript
// Cypress
cy.get('[data-testid="submit"]', { timeout: 10000 }).should('be.visible');

// Playwright
await page.waitForSelector('[data-testid="submit"]', { state: 'visible' });

// Selenium
wait = WebDriverWait(driver, 10)
element = wait.until(EC.visibility_of_element_located((By.ID, "submit")))
```

### Handling Dynamic Content

```javascript
// Wait for network requests to complete
await page.waitForLoadState('networkidle');

// Wait for specific API call
await page.waitForResponse(response =>
  response.url().includes('/api/users') && response.status() === 200
);

// Retry logic for flaky elements
await page.waitForFunction(() => {
  const element = document.querySelector('[data-testid="dynamic"]');
  return element && element.textContent.includes('Loaded');
});
```

### Authentication State

```javascript
// Playwright - Save and reuse authentication
const { test: setup } = require('@playwright/test');

setup('authenticate', async ({ page }) => {
  await page.goto('https://example.com/login');
  await page.fill('input[name="email"]', 'user@example.com');
  await page.fill('input[name="password"]', 'password123');
  await page.click('button[type="submit"]');

  await page.waitForURL('**/dashboard');

  // Save authentication state
  await page.context().storageState({ path: 'auth.json' });
});

// Use saved state in tests
test.use({ storageState: 'auth.json' });

test('access protected page', async ({ page }) => {
  // Already authenticated
  await page.goto('https://example.com/profile');
  await expect(page.locator('h1')).toContainText('My Profile');
});
```

### File Uploads

```javascript
// Playwright
await page.setInputFiles('input[type="file"]', 'path/to/file.pdf');

// Cypress
cy.get('input[type="file"]').selectFile('cypress/fixtures/file.pdf');

// Multiple files
cy.get('input[type="file"]').selectFile([
  'cypress/fixtures/file1.pdf',
  'cypress/fixtures/file2.pdf'
]);
```

### Screenshots and Videos

```javascript
// Playwright configuration
// playwright.config.js
module.exports = {
  use: {
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
};

// Take custom screenshot
await page.screenshot({ path: 'screenshot.png' });
await page.screenshot({ path: 'fullpage.png', fullPage: true });

// Cypress automatically captures on failure
// Manual screenshot
cy.screenshot('custom-name');
```

## Testing Best Practices

### 1. Use Data Attributes

```html
<!-- Good: Dedicated test attributes -->
<button data-testid="submit-button">Submit</button>
<input data-testid="email-input" type="email" />

<!-- Avoid: CSS classes or IDs that may change -->
<button class="btn btn-primary">Submit</button>
```

### 2. Independent Tests

```javascript
// Good: Each test is independent
beforeEach(async () => {
  await resetDatabase();
  await seedTestData();
});

test('create user', async () => {
  // Test creates its own data
});

test('delete user', async () => {
  // Test creates its own data
});

// Bad: Tests depend on each other
test('create user', () => { /* ... */ });
test('delete user created in previous test', () => { /* ... */ }); // BAD
```

### 3. Avoid Hard-Coded Waits

```javascript
// Bad
await page.waitForTimeout(5000); // Arbitrary wait

// Good
await page.waitForSelector('[data-testid="result"]');
await page.waitForLoadState('networkidle');
await page.waitForResponse(response => response.url().includes('/api/data'));
```

### 4. Test Critical User Journeys

Focus on high-value workflows:
- User registration and login
- Purchase/checkout flow
- Core feature usage
- Critical business processes

### 5. Parallelize Tests

```javascript
// Playwright - run tests in parallel
// playwright.config.js
module.exports = {
  workers: 4, // Run 4 tests concurrently
  fullyParallel: true,
};

// Cypress - parallel execution
// cypress.config.js
module.exports = {
  e2e: {
    experimentalRunAllSpecs: true,
  },
};
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
```

### Docker for Consistent Environments

```dockerfile
# Dockerfile.e2e
FROM mcr.microsoft.com/playwright:v1.40.0

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

CMD ["npm", "run", "test:e2e"]
```

```bash
# Run E2E tests in Docker
docker build -f Dockerfile.e2e -t e2e-tests .
docker run --rm e2e-tests
```

## Debugging E2E Tests

### Playwright Debug Mode

```bash
# Run in headed mode with slowmo
PWDEBUG=1 npx playwright test

# Debug specific test
npx playwright test --debug test-name

# Use Playwright Inspector
npx playwright test --ui
```

### Cypress Debug

```javascript
// Add breakpoints
cy.get('[data-testid="button"]').click();
cy.debug(); // Pause test here
cy.get('[data-testid="result"]').should('be.visible');

// Use .pause()
cy.get('[data-testid="button"]').click();
cy.pause(); // Pause with debugger
```

### Console Logs

```javascript
// Capture browser console logs
page.on('console', msg => console.log('Browser:', msg.text()));

// Capture network failures
page.on('requestfailed', request => {
  console.log('Failed:', request.url());
});
```

## Common Pitfalls

### 1. Flaky Tests
**Problem**: Tests pass/fail inconsistently
**Solution**:
- Use proper waits instead of hard-coded timeouts
- Wait for network requests to complete
- Ensure test data independence

### 2. Slow Test Execution
**Problem**: Tests take too long
**Solution**:
- Run tests in parallel
- Use authentication state sharing
- Mock external services when appropriate
- Optimize test data setup

### 3. Brittle Selectors
**Problem**: Tests break when UI changes
**Solution**:
- Use data-testid attributes
- Avoid CSS class selectors
- Use Page Object Model pattern

### 4. Environment Differences
**Problem**: Tests work locally but fail in CI
**Solution**:
- Use Docker for consistent environments
- Configure headless mode properly
- Check viewport sizes
- Verify environment variables

## Configuration Examples

### Playwright Configuration

```javascript
// playwright.config.js
module.exports = {
  testDir: './tests/e2e',
  timeout: 30000,
  retries: 2,
  workers: 4,

  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
    },
    {
      name: 'webkit',
      use: { browserName: 'webkit' },
    },
    {
      name: 'mobile',
      use: {
        browserName: 'chromium',
        viewport: { width: 375, height: 667 },
        deviceScaleFactor: 2,
        isMobile: true,
      },
    },
  ],

  webServer: {
    command: 'npm run start',
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
};
```

### Cypress Configuration

```javascript
// cypress.config.js
const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    screenshotOnRunFailure: true,

    setupNodeEvents(on, config) {
      // implement node event listeners here
    },

    env: {
      apiUrl: 'http://localhost:4000/api',
    },
  },

  retries: {
    runMode: 2,
    openMode: 0,
  },
});
```

## Quick Reference

| Tool | Language | Browser Support | Headless | Best For |
|------|----------|----------------|----------|----------|
| Playwright | JS/TS/Python | Chrome, Firefox, Safari | Yes | Modern apps, cross-browser |
| Cypress | JavaScript | Chrome, Firefox, Edge | Yes | Developer experience, debugging |
| Selenium | Multi-language | All major browsers | Yes | Legacy support, multi-language |
| Puppeteer | JavaScript | Chrome only | Yes | Chrome-specific, performance |

## Test Organization

```
tests/
├── e2e/
│   ├── auth/
│   │   ├── login.spec.js
│   │   ├── signup.spec.js
│   │   └── logout.spec.js
│   ├── checkout/
│   │   ├── cart.spec.js
│   │   └── payment.spec.js
│   └── admin/
│       ├── users.spec.js
│       └── settings.spec.js
├── pages/
│   ├── LoginPage.js
│   ├── DashboardPage.js
│   └── CheckoutPage.js
├── fixtures/
│   ├── users.json
│   └── products.json
└── support/
    ├── commands.js
    └── helpers.js
```

## Further Resources

- [Playwright Documentation](https://playwright.dev/)
- [Cypress Documentation](https://docs.cypress.io/)
- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [Puppeteer Documentation](https://pptr.dev/)
- [Testing Library](https://testing-library.com/)
- [Page Object Model Pattern](https://martinfowler.com/bliki/PageObject.html)

E2E tests provide confidence that your application works as users expect, catching integration issues that unit and integration tests might miss.
