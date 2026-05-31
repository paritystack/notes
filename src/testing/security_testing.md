# Security Testing

## Overview

Security testing finds vulnerabilities before attackers do. The modern approach
is **shift-left**: catch issues early and automatically in development and
[CI](ci_testing.md), rather than in a late, manual pen-test. The main techniques
complement each other:

| Technique | Looks at | Catches | Runs |
|-----------|----------|---------|------|
| **SAST** | source code | injection, hardcoded secrets, unsafe APIs | no app needed |
| **DAST** | running app | XSS, auth flaws, misconfig | needs deployed app |
| **SCA** | dependencies | known CVEs in libraries | on the lockfile |
| **Secrets scanning** | code & history | leaked keys/tokens | on every commit |

## SAST — Static Application Security Testing

Analyzes source code without running it, flagging dangerous patterns.

```bash
# Python — Bandit
pip install bandit
bandit -r myapp/                 # scan recursively

# Multi-language — Semgrep (rule-based)
semgrep --config=auto .          # community rulesets

# GitHub — CodeQL runs in Actions via the code-scanning workflow
```

Typical findings: SQL string concatenation, `eval`/`exec` on user input,
`subprocess(..., shell=True)`, weak crypto, hardcoded credentials.

## DAST — Dynamic Application Security Testing

Attacks a *running* application from the outside, like a black-box pen-tester.

```bash
# OWASP ZAP — baseline scan against a running app
zap-baseline.py -t https://staging.example.com -r report.html
```

Catches runtime issues SAST can't see: reflected/stored XSS, broken
authentication, CSRF, security-header misconfiguration, injection via live
endpoints.

## SCA — Software Composition Analysis

Most code is third-party. SCA checks dependencies against vulnerability
databases (CVE/GHSA).

```bash
npm audit                        # Node dependencies
npm audit fix
pip-audit                        # Python dependencies
osv-scanner -r .                 # multi-ecosystem (Google OSV)
```

- **Dependabot / Renovate** open automated PRs to bump vulnerable packages.
- **Snyk** adds licensing checks and fix advice.
- Watch for **transitive** dependencies and lockfile drift.

## Secrets Scanning

Prevent API keys, tokens, and passwords from being committed.

```bash
gitleaks detect --source .       # scans working tree + git history
trufflehog git file://.          # finds & verifies live secrets
```

Run as a **pre-commit hook** and in CI. Scan **history**, not just the latest
commit — a removed secret is still in the git log and must be rotated.

## Fuzzing (brief)

Feed malformed/random input to find crashes and memory-safety bugs. Tools:
`atheris` (Python), `libFuzzer`/AFL++ (C/C++), Go's built-in `go test -fuzz`.
Conceptually related to [property-based testing](property_based_testing.md) but
aimed at robustness/security rather than correctness properties.

## OWASP Top 10

A baseline checklist of the most critical web risks — broken access control,
injection, cryptographic failures, security misconfiguration, vulnerable
components, SSRF, etc. Use it to prioritize what to test for.

## Wiring into CI

Add security gates to the pipeline (see [CI/CD Test Automation](ci_testing.md)):
secrets scan + SAST + SCA on every PR (fail on high severity), DAST against a
staging deploy nightly. Keep PR-blocking scans fast; run heavier scans on a
schedule.

## Best Practices

1. **Automate in CI** and fail builds on high/critical findings; don't rely on
   manual reviews alone.
2. **Triage false positives** with ignore/baseline files so alerts stay credible.
3. **Rotate any leaked secret immediately** — scrubbing git history isn't enough.
4. **Keep dependencies patched** via automated update PRs.
5. **Layer the techniques** — SAST + DAST + SCA + secrets cover different gaps.
6. **Pin and verify** third-party actions/images to reduce supply-chain risk.

## ELI10

Securing a house: SAST is an inspector reading the blueprints for design flaws,
DAST is someone walking around at night trying every window, SCA checks whether
the locks you bought have a known master-key flaw, and secrets scanning makes
sure you didn't leave a key under the mat.

## Further Resources

- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Bandit](https://bandit.readthedocs.io/) · [Semgrep](https://semgrep.dev/) · [OWASP ZAP](https://www.zaproxy.org/)
- [gitleaks](https://github.com/gitleaks/gitleaks) · [OSV-Scanner](https://google.github.io/osv-scanner/)
