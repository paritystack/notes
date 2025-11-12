# CI/CD Fundamentals

## Overview

**CI (Continuous Integration)**: Automatically test code on every commit
**CD (Continuous Deployment)**: Automatically deploy to production

## Pipeline Stages

```
Code Commit
    ↓
Build (compile, package)
    ↓
Test (unit, integration, e2e)
    ↓
Deploy to Staging
    ↓
Manual/Automated Approval
    ↓
Deploy to Production
```

## Tools

### GitHub Actions

```yaml
name: CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'

    - name: Install dependencies
      run: npm install

    - name: Run tests
      run: npm test

    - name: Run linter
      run: npm run lint

    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: npm run deploy
```

### GitLab CI

```yaml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/

test:
  stage: test
  script:
    - npm install
    - npm test

deploy:
  stage: deploy
  script:
    - npm run deploy
  only:
    - main
```

## Best Practices

1. **Automated Testing**: Every commit
2. **Fast Feedback**: Minutes, not hours
3. **Deploy Often**: Small, frequent changes
4. **Monitoring**: Alert on failures
5. **Rollback Ready**: Revert quickly if needed

## Pipeline as Code

Define pipeline in version control:

```yaml
# .github/workflows/deploy.yml
name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: docker build -t myapp .
      - run: docker push myapp:latest
      - run: kubectl apply -f deployment.yaml
```

## Deployment Strategies

### Blue-Green
```
Blue (current):  Production v1
Green (new):     Production v2

Switch traffic instantly to v2
If issue: Switch back to v1
```

### Canary
```
Release to 5% of users first
Monitor metrics
If healthy: 10% → 25% → 50% → 100%
If issues: Rollback at any stage
```

### Rolling
```
Stop pod, deploy new version
Repeat for each pod
Zero downtime
```

## Monitoring in CI/CD

```yaml
# Check metrics after deploy
- name: Health check
  run: |
    curl -f https://api.example.com/health || exit 1

- name: Performance check
  run: |
    response_time=$(curl -w '%{time_total}' https://api.example.com)
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
      echo "Slow response: $response_time seconds"
      exit 1
    fi
```

## Common Issues

### Flaky Tests
Tests that pass/fail randomly
**Solution**: Fix test, increase timeout, isolate dependencies

### Deployment Failures
**Solution**: Pre-deployment checks, canary deployments, rollback procedures

### Security Vulnerabilities
**Solution**: Dependency scanning, static code analysis, container scanning

## ELI10

CI/CD is like an assembly line:
1. **CI**: Test each part as made
2. **CD**: Automatically package and ship
3. **Monitoring**: Check if delivery was successful

Catch problems BEFORE customers see them!

## Further Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
- [Jenkins Guide](https://www.jenkins.io/doc/)
- [The Twelve-Factor App](https://12factor.net/)
