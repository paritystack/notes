# Kubernetes

## Overview

Kubernetes (K8s) orchestrates containerized applications at scale, handling deployment, scaling, and networking.

## Core Concepts

### Pods
Smallest deployable unit (usually one container):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8000
```

### Deployments
Manages replicas of pods:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0
        ports:
        - containerPort: 8000
```

### Services
Expose pods to network:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## kubectl Commands

```bash
# Create/update
kubectl apply -f deployment.yaml

# View resources
kubectl get pods
kubectl get deployments
kubectl get services

# Describe
kubectl describe pod my-pod

# Logs
kubectl logs my-pod

# Execute
kubectl exec -it my-pod -- bash

# Delete
kubectl delete pod my-pod
kubectl delete deployment myapp

# Scale
kubectl scale deployment myapp --replicas=5

# Port forwarding
kubectl port-forward myapp-pod 8000:8000
```

## Architecture

```
┌─────────────────────────┐
│   Control Plane         │
│  - API Server           │
│  - etcd (store)         │
│  - Scheduler            │
│  - Controller Manager   │
└─────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│  Worker Nodes                       │
│  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │ Pod  │  │ Pod  │  │ Pod  │     │
│  └──────┘  └──────┘  └──────┘     │
└─────────────────────────────────────┘
```

## ConfigMap & Secrets

```yaml
# ConfigMap (non-sensitive)
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  DATABASE_HOST: "db.example.com"

---
# Secret (sensitive)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  password: cGFzc3dvcmQxMjM=  # Base64 encoded
```

## Namespaces

Logical cluster partitions:

```bash
kubectl create namespace development
kubectl apply -f deployment.yaml -n development
kubectl get pods -n development
```

## Scaling & Updates

```bash
# Manual scaling
kubectl scale deployment myapp --replicas=10

# Rolling update
kubectl set image deployment/myapp myapp=myapp:2.0
kubectl rollout status deployment/myapp
kubectl rollout undo deployment/myapp  # Revert
```

## Resource Limits

```yaml
spec:
  containers:
  - name: myapp
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

## ELI10

Kubernetes is like a smart warehouse manager:
- Receives orders (deployments)
- Assigns workers (pods)
- Keeps right number working
- Fixes broken ones automatically
- Spreads load across workers

Imagine managing 1000 containers automatically!

## Further Resources

- [Kubernetes Docs](https://kubernetes.io/docs/)
- [Interactive Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [kubectl Cheatsheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
