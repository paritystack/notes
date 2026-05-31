# Helm

## Overview

Helm is the de-facto package manager for [Kubernetes](kubernetes.md) — "apt/npm for k8s". It
solves a real pain: a single application is dozens of YAML manifests (Deployment, Service,
ConfigMap, Ingress, …), nearly identical across dev/staging/prod except for a handful of
values. Helm packages them into a templated, versioned, parameterizable unit (a **chart**)
that you install, upgrade, and roll back as one release. It's a core building block of
[GitOps](gitops.md) and [CI/CD](cicd.md) deployment to Kubernetes.

```
Without Helm: copy-paste 15 YAML files per environment, hand-edit image tags & replicas.
With Helm:     one chart + a values file per environment → `helm upgrade` deploys it.
```

## Core concepts

```
Chart      a package: templates + default values + metadata. The unit you distribute.
Values     configuration (values.yaml) injected into templates → per-env customization.
Release    a chart installed into a cluster with a name & a REVISION history (for rollback).
Repository a collection of charts (Artifact Hub, Bitnami, or an OCI registry).
```

## Chart structure

```
mychart/
├── Chart.yaml          name, version, appVersion, dependencies
├── values.yaml         DEFAULT configuration values
├── templates/          Go-templated manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── _helpers.tpl    reusable template snippets (named templates)
│   └── NOTES.txt       post-install message
└── charts/             vendored subchart dependencies
```

## Templating

Helm renders Go templates with the merged values to produce plain Kubernetes YAML:

```yaml
# templates/deployment.yaml
spec:
  replicas: {{ .Values.replicaCount }}
  containers:
    - image: "{{ .Values.image.repo }}:{{ .Values.image.tag }}"
      resources:
        {{- toYaml .Values.resources | nindent 8 }}
```

```
Common helpers:
  {{ .Values.x }}                  inject a value
  {{ .Release.Name }}              release/built-in objects
  {{- if .Values.ingress.enabled }} ... {{- end }}   conditionals
  {{- range .Values.envs }} ... {{- end }}           loops
  {{ include "mychart.labels" . }} reusable named template from _helpers.tpl
  | nindent / | quote / | default  pipelines for formatting
Whitespace ({{-  -}}) control matters — bad indentation = invalid YAML.
```

## Values & environment overrides

```
Precedence (later wins):
  chart values.yaml  <  -f custom-values.yaml  <  --set key=value (CLI)

helm install api ./mychart -f values-prod.yaml --set image.tag=1.4.2
⇒ same chart, different values per environment — the whole point.
```

## Lifecycle commands

```
helm install <name> <chart>          create a release
helm upgrade <name> <chart>          apply changes (new revision)
helm upgrade --install               idempotent install-or-upgrade (CI-friendly)
helm rollback <name> <revision>      revert to a previous revision  ← key superpower
helm uninstall <name>                remove the release
helm list                            releases & their revisions
helm template <chart>                render to YAML locally WITHOUT installing (debug/GitOps)
helm diff upgrade (plugin)           preview what an upgrade will change
```

## Dependencies & subcharts

```
Chart.yaml `dependencies:` pull in other charts (e.g. a Bitnami postgres subchart).
helm dependency update fetches them into charts/.
Parent values can override subchart values → compose apps from reusable pieces.
```

## Helm hooks

```
Run jobs at lifecycle points via annotations:
  pre-install / post-install / pre-upgrade / post-upgrade / pre-delete
Uses: DB migrations before an upgrade, smoke tests after install, cleanup on delete.
```

## Helm vs Kustomize

The two dominant config tools — often complementary, sometimes either/or:

```
Helm       templating + packaging + releases + rollback + a dependency ecosystem.
           Best for: distributable apps, lots of variation, third-party software.
           Cost: Go-template complexity; logic hidden in templates.
Kustomize  template-free OVERLAYS that patch a plain base of YAML (built into kubectl).
           Best for: your own apps, simple per-env differences, "no magic" YAML.
Many teams use both: Helm to install third-party charts, Kustomize for in-house manifests.
GitOps agents (Argo CD/Flux) support both natively — see gitops.md.
```

## Security & supply chain

```
- Pin chart versions; review third-party charts before installing (they run in YOUR cluster).
- Prefer charts from trusted publishers; charts can be signed/provenance-checked.
- Store charts in an OCI registry (helm push to OCI) for versioned, access-controlled distribution.
- Don't bake secrets into values.yaml in Git → use external secrets. See
  ../security/secrets_management.md and supply_chain_security.md.
```

## Where this connects

- **[Kubernetes](kubernetes.md)** — what Helm deploys.
- **[GitOps](gitops.md)** — Argo CD/Flux render Helm charts as the desired state.
- **[CI/CD](cicd.md)** — `helm upgrade --install` in pipelines.
- **[Secrets management](../security/secrets_management.md)** — keep secrets out of charts.

## Pitfalls

- **YAML indentation in templates** — `nindent`/whitespace bugs produce invalid manifests;
  `helm template` to verify before applying.
- **Secrets in values.yaml** — committed to Git in plaintext; use external/sealed secrets.
- **Unpinned chart versions** — surprise upgrades; pin `version` in Chart.yaml/dependencies.
- **Over-templating** — charts with deep conditional logic become unmaintainable; Kustomize
  overlays may be simpler for in-house apps.
- **Forgetting rollback history limits** — `--history-max` trims old revisions; keep enough
  to roll back.
