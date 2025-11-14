# GitHub Actions

GitHub's native CI/CD and automation platform for building, testing, and deploying code directly from GitHub repositories.

## Core Concepts

### Workflows
YAML files defining automated processes:
```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
```

### Components

#### Events
Triggers that start workflows:
- **push**: Code pushed to repository
- **pull_request**: PR opened, synchronized, or reopened
- **schedule**: Cron-based scheduling
- **workflow_dispatch**: Manual trigger
- **release**: Release published
- **issues**: Issue opened or modified
- **workflow_call**: Called by another workflow

#### Jobs
Set of steps that execute on same runner:
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Run tests
        run: npm test

  deploy:
    runs-on: ubuntu-latest
    needs: [build, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy
        run: ./deploy.sh
```

#### Steps
Individual tasks in a job:
```yaml
steps:
  - name: Checkout repository
    uses: actions/checkout@v3

  - name: Setup Node.js
    uses: actions/setup-node@v3
    with:
      node-version: '18'

  - name: Install dependencies
    run: npm ci

  - name: Run build
    run: npm run build
```

#### Actions
Reusable units of code:
```yaml
# Using marketplace action
- uses: actions/checkout@v3
  with:
    fetch-depth: 0

# Using local action
- uses: ./.github/actions/custom-action
  with:
    parameter: value

# Using action from another repo
- uses: owner/repo@v1
  with:
    token: ${{ secrets.GITHUB_TOKEN }}
```

#### Runners
Servers that execute workflows:
```yaml
jobs:
  linux:
    runs-on: ubuntu-latest

  macos:
    runs-on: macos-latest

  windows:
    runs-on: windows-latest

  self-hosted:
    runs-on: [self-hosted, linux, x64]
```

## Common Triggers

### Push Events
```yaml
on:
  push:
    branches:
      - main
      - develop
      - 'feature/**'
    tags:
      - 'v*'
    paths:
      - 'src/**'
      - '**.js'
    paths-ignore:
      - 'docs/**'
      - '**.md'
```

### Pull Request Events
```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - main
    paths:
      - 'src/**'
```

### Schedule
```yaml
on:
  schedule:
    # Every day at 2 AM UTC
    - cron: '0 2 * * *'
    # Every Monday at 9 AM UTC
    - cron: '0 9 * * 1'
```

### Manual Trigger
```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - development
          - staging
          - production
      debug:
        description: 'Enable debug mode'
        required: false
        type: boolean
        default: false
```

### Multiple Events
```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:
```

## Common Patterns

### CI Pipeline
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run build
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-output
          path: dist/
```

### CD Pipeline
```yaml
name: CD

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options:
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: ${{ github.event.inputs.environment || 'staging' }}
      url: https://${{ steps.deploy.outputs.url }}
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Deploy to AWS
        id: deploy
        run: |
          aws s3 sync ./dist s3://my-bucket
          echo "url=my-app.com" >> $GITHUB_OUTPUT

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployment ${{ job.status }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Matrix Builds
```yaml
name: Matrix Build

on: [push]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [16, 18, 20]
        include:
          - os: ubuntu-latest
            node-version: 20
            experimental: true
        exclude:
          - os: macos-latest
            node-version: 16
      fail-fast: false
      max-parallel: 4

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

### Caching Dependencies
```yaml
name: Build with Cache

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Cache Node modules
        uses: actions/cache@v3
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - run: npm ci
      - run: npm run build
```

### Docker Build and Push
```yaml
name: Docker

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: myapp/image
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=myapp/image:buildcache
          cache-to: type=registry,ref=myapp/image:buildcache,mode=max
```

### Release Automation
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: metcalfc/changelog-generator@v4
        with:
          myToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.changelog }}
          files: |
            dist/**
            LICENSE
            README.md
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Monorepo Pattern
```yaml
name: Monorepo CI

on: [push, pull_request]

jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      frontend: ${{ steps.filter.outputs.frontend }}
      backend: ${{ steps.filter.outputs.backend }}
    steps:
      - uses: actions/checkout@v3
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            frontend:
              - 'packages/frontend/**'
            backend:
              - 'packages/backend/**'

  frontend:
    needs: changes
    if: needs.changes.outputs.frontend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm ci
        working-directory: packages/frontend
      - run: npm test
        working-directory: packages/frontend

  backend:
    needs: changes
    if: needs.changes.outputs.backend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm ci
        working-directory: packages/backend
      - run: npm test
        working-directory: packages/backend
```

## Secrets and Variables

### Repository Secrets
```yaml
steps:
  - name: Use secret
    run: echo "Secret value: ${{ secrets.MY_SECRET }}"
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

### Environment Secrets
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy
        run: ./deploy.sh
        env:
          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
```

### Variables
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Use variables
        run: |
          echo "Environment: ${{ vars.ENVIRONMENT }}"
          echo "API URL: ${{ vars.API_URL }}"
```

## Reusable Workflows

### Callable Workflow
```yaml
# .github/workflows/reusable-deploy.yml
name: Reusable Deploy

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      region:
        required: false
        type: string
        default: 'us-east-1'
    secrets:
      aws-access-key:
        required: true
      aws-secret-key:
        required: true
    outputs:
      deployment-url:
        description: "URL of deployment"
        value: ${{ jobs.deploy.outputs.url }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    outputs:
      url: ${{ steps.deploy.outputs.url }}
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        id: deploy
        run: |
          echo "Deploying to ${{ inputs.environment }}"
          echo "url=https://app.example.com" >> $GITHUB_OUTPUT
```

### Calling Workflow
```yaml
# .github/workflows/main.yml
name: Main

on: [push]

jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging
    secrets:
      aws-access-key: ${{ secrets.AWS_ACCESS_KEY_ID }}
      aws-secret-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

  deploy-prod:
    needs: deploy-staging
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
      region: us-west-2
    secrets:
      aws-access-key: ${{ secrets.AWS_ACCESS_KEY_ID }}
      aws-secret-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

## Composite Actions

### Custom Action
```yaml
# .github/actions/setup-app/action.yml
name: 'Setup Application'
description: 'Setup Node.js and install dependencies'

inputs:
  node-version:
    description: 'Node.js version'
    required: false
    default: '18'
  cache:
    description: 'Enable caching'
    required: false
    default: 'true'

runs:
  using: 'composite'
  steps:
    - uses: actions/setup-node@v3
      with:
        node-version: ${{ inputs.node-version }}
        cache: ${{ inputs.cache == 'true' && 'npm' || '' }}

    - name: Install dependencies
      shell: bash
      run: npm ci

    - name: Verify installation
      shell: bash
      run: npm --version
```

### Using Custom Action
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: ./.github/actions/setup-app
        with:
          node-version: '20'
      - run: npm run build
```

## Best Practices

### Security
```yaml
# Use specific versions, not latest
- uses: actions/checkout@v3

# Limit permissions
permissions:
  contents: read
  issues: write

# Use environments for protection rules
jobs:
  deploy:
    environment:
      name: production
      url: https://prod.example.com

# Never log secrets
- run: echo "Token: ***"
  env:
    TOKEN: ${{ secrets.GITHUB_TOKEN }}

# Use OIDC for cloud authentication
- uses: aws-actions/configure-aws-credentials@v2
  with:
    role-to-assume: arn:aws:iam::123456789012:role/MyRole
    aws-region: us-east-1
```

### Performance
```yaml
# Use caching
- uses: actions/cache@v3

# Limit checkout depth
- uses: actions/checkout@v3
  with:
    fetch-depth: 1

# Use concurrency to cancel old runs
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

# Skip CI when not needed
on:
  push:
    paths-ignore:
      - 'docs/**'
      - '**.md'
```

### Maintainability
```yaml
# Use meaningful names
name: Backend CI Pipeline

# Add descriptions to inputs
inputs:
  environment:
    description: 'Target deployment environment'
    required: true

# Use step outputs for data flow
- id: build
  run: echo "version=1.0.0" >> $GITHUB_OUTPUT
- run: echo "Version: ${{ steps.build.outputs.version }}"

# Group related steps
- name: Setup dependencies
  run: |
    npm ci
    npm run setup

# Use continue-on-error for optional steps
- name: Upload test results
  if: always()
  continue-on-error: true
  uses: actions/upload-artifact@v3
```

### Debugging
```yaml
# Enable debug logging
# Set repository secret: ACTIONS_STEP_DEBUG=true

# Use step debugging
- name: Debug
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "Actor: ${{ github.actor }}"

# View context
- name: Dump GitHub context
  env:
    GITHUB_CONTEXT: ${{ toJson(github) }}
  run: echo "$GITHUB_CONTEXT"
```

## Context Variables

### GitHub Context
```yaml
${{ github.repository }}       # owner/repo
${{ github.ref }}               # refs/heads/main
${{ github.sha }}               # commit SHA
${{ github.actor }}             # username who triggered
${{ github.event_name }}        # push, pull_request, etc.
${{ github.run_id }}            # unique run ID
${{ github.run_number }}        # run number
```

### Job Context
```yaml
${{ job.status }}               # success, failure, cancelled
${{ job.container.id }}         # container ID if used
```

### Runner Context
```yaml
${{ runner.os }}                # Linux, Windows, macOS
${{ runner.arch }}              # X64, ARM, ARM64
${{ runner.temp }}              # temp directory path
${{ runner.tool_cache }}        # tool cache path
```

### Environment Variables
```yaml
env:
  NODE_ENV: production
  API_URL: ${{ vars.API_URL }}
  SECRET_KEY: ${{ secrets.SECRET_KEY }}
```

## Advanced Patterns

### Conditional Execution
```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
      - name: Deploy to staging
        if: contains(github.event.head_commit.message, '[staging]')
        run: ./deploy-staging.sh

      - name: Deploy to production
        if: startsWith(github.ref, 'refs/tags/v')
        run: ./deploy-prod.sh
```

### Dynamic Matrix
```yaml
jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - id: set-matrix
        run: |
          echo "matrix={\"node\":[16,18,20]}" >> $GITHUB_OUTPUT

  build:
    needs: setup
    strategy:
      matrix: ${{ fromJson(needs.setup.outputs.matrix) }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
```

### Artifact Management
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build
      - uses: actions/upload-artifact@v3
        with:
          name: build-${{ github.sha }}
          path: dist/
          retention-days: 5

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
        with:
          name: build-${{ github.sha }}
          path: dist/
      - run: ./deploy.sh
```

## Troubleshooting

### Common Issues

#### Workflow not triggering
- Check event filters (branches, paths)
- Verify YAML syntax
- Check if workflow file is in `.github/workflows/`

#### Permission errors
```yaml
permissions:
  contents: write
  packages: write
  pull-requests: write
```

#### Timeout issues
```yaml
jobs:
  build:
    timeout-minutes: 60
    steps:
      - name: Long task
        timeout-minutes: 30
        run: ./long-task.sh
```

#### Rate limiting
```yaml
- name: Wait before API call
  run: sleep 10
```

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Actions Marketplace](https://github.com/marketplace?type=actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
