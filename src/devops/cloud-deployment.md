# Cloud Deployment

Multi-cloud deployment strategies, patterns, and best practices for AWS, GCP, and Azure.

## Cloud Platforms Overview

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Compute | EC2, ECS, EKS, Lambda | Compute Engine, GKE, Cloud Run | VMs, AKS, Container Instances |
| Storage | S3, EBS, EFS | Cloud Storage, Persistent Disk | Blob Storage, Managed Disks |
| Database | RDS, DynamoDB, Aurora | Cloud SQL, Firestore, Spanner | SQL Database, Cosmos DB |
| Networking | VPC, CloudFront, Route53 | VPC, Cloud CDN, Cloud DNS | VNet, CDN, DNS |
| Serverless | Lambda, API Gateway | Cloud Functions, Cloud Run | Functions, API Management |
| Container Orchestration | EKS, ECS, Fargate | GKE, Cloud Run | AKS, Container Apps |

## AWS Deployment

### EC2 Instances

#### Launch Configuration
```bash
# Create instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.micro \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --user-data file://userdata.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'

# Using Auto Scaling
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name web-asg \
  --launch-template LaunchTemplateName=web-template \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3 \
  --vpc-zone-identifier "subnet-1,subnet-2,subnet-3" \
  --target-group-arns arn:aws:elasticloadbalancing:region:account:targetgroup/name \
  --health-check-type ELB \
  --health-check-grace-period 300
```

#### User Data Script
```bash
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Pull and run application
docker pull myapp:latest
docker run -d -p 80:3000 myapp:latest

# Configure CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm
```

### Elastic Container Service (ECS)

#### Task Definition
```json
{
  "family": "web-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NODE_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:db-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/web-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "app"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Service Definition
```json
{
  "serviceName": "web-service",
  "taskDefinition": "web-app:1",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-1", "subnet-2"],
      "securityGroups": ["sg-12345"],
      "assignPublicIp": "ENABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/name",
      "containerName": "app",
      "containerPort": 3000
    }
  ],
  "deploymentConfiguration": {
    "maximumPercent": 200,
    "minimumHealthyPercent": 100,
    "deploymentCircuitBreaker": {
      "enable": true,
      "rollback": true
    }
  },
  "serviceRegistries": [
    {
      "registryArn": "arn:aws:servicediscovery:region:account:service/srv-12345"
    }
  ]
}
```

### Lambda Deployment

#### Function Code
```javascript
// index.js
exports.handler = async (event) => {
  const { httpMethod, path, body } = event;

  if (httpMethod === 'GET' && path === '/health') {
    return {
      statusCode: 200,
      body: JSON.stringify({ status: 'healthy' })
    };
  }

  try {
    const data = JSON.parse(body);
    // Process data
    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ message: 'Success' })
    };
  } catch (error) {
    console.error('Error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Internal Server Error' })
    };
  }
};
```

#### SAM Template
```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    Runtime: nodejs18.x
    Environment:
      Variables:
        TABLE_NAME: !Ref DynamoDBTable

Resources:
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: index.handler
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref DynamoDBTable
      VpcConfig:
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
        SecurityGroupIds:
          - !Ref LambdaSecurityGroup

  DynamoDBTable:
    Type: AWS::DynamoDB::Table
    Properties:
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH

Outputs:
  ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub 'https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/'
```

#### Deployment
```bash
# Package and deploy with SAM
sam build
sam deploy --guided

# Or with Serverless Framework
serverless deploy --stage production --region us-east-1

# Direct Lambda update
zip function.zip index.js
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip
```

### RDS Database

#### CloudFormation Template
```yaml
Resources:
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for RDS
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

  DBInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: production-db
      Engine: postgres
      EngineVersion: '15.3'
      DBInstanceClass: db.t3.medium
      AllocatedStorage: 100
      StorageType: gp3
      StorageEncrypted: true
      MasterUsername: admin
      MasterUserPassword: !Sub '{{resolve:secretsmanager:${DBSecret}::password}}'
      DBSubnetGroupName: !Ref DBSubnetGroup
      VPCSecurityGroups:
        - !Ref DBSecurityGroup
      BackupRetentionPeriod: 7
      PreferredBackupWindow: '03:00-04:00'
      PreferredMaintenanceWindow: 'sun:04:00-sun:05:00'
      MultiAZ: true
      EnableCloudwatchLogsExports:
        - postgresql
      DeletionProtection: true

  DBSecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      GenerateSecretString:
        SecretStringTemplate: '{"username": "admin"}'
        GenerateStringKey: 'password'
        PasswordLength: 32
        ExcludeCharacters: '"@/\'
```

## GCP Deployment

### Compute Engine

#### Instance Template
```bash
# Create instance template
gcloud compute instance-templates create web-template \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --network-interface=network=default,subnet=default \
  --metadata-from-file=startup-script=startup.sh \
  --tags=http-server,https-server \
  --service-account=my-service-account@project.iam.gserviceaccount.com \
  --scopes=cloud-platform

# Create managed instance group
gcloud compute instance-groups managed create web-group \
  --base-instance-name=web \
  --template=web-template \
  --size=3 \
  --zone=us-central1-a \
  --health-check=http-health-check \
  --initial-delay=300

# Configure autoscaling
gcloud compute instance-groups managed set-autoscaling web-group \
  --zone=us-central1-a \
  --max-num-replicas=10 \
  --min-num-replicas=2 \
  --target-cpu-utilization=0.6 \
  --cool-down-period=60
```

### Google Kubernetes Engine (GKE)

#### Cluster Creation
```bash
# Create GKE cluster
gcloud container clusters create production-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-stackdriver-kubernetes \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
  --workload-pool=project-id.svc.id.goog \
  --enable-shielded-nodes \
  --release-channel=regular

# Get credentials
gcloud container clusters get-credentials production-cluster \
  --zone=us-central1-a
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
        version: v1
    spec:
      serviceAccountName: web-app-sa
      containers:
      - name: app
        image: gcr.io/project-id/web-app:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: NODE_ENV
          value: production
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Cloud Run

#### Service Configuration
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: web-app
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: '1'
        autoscaling.knative.dev/maxScale: '100'
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      serviceAccountName: cloud-run-sa@project-id.iam.gserviceaccount.com
      containers:
      - image: gcr.io/project-id/web-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: NODE_ENV
          value: production
        resources:
          limits:
            cpu: '1'
            memory: 512Mi
```

#### Deploy Cloud Run
```bash
# Deploy from local
gcloud run deploy web-app \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 100 \
  --cpu 1 \
  --memory 512Mi \
  --timeout 300 \
  --set-env-vars NODE_ENV=production

# Deploy from container registry
gcloud run deploy web-app \
  --image gcr.io/project-id/web-app:v1.0.0 \
  --region us-central1 \
  --platform managed
```

## Azure Deployment

### Virtual Machines

#### ARM Template
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachines",
      "apiVersion": "2023-03-01",
      "name": "web-vm",
      "location": "[resourceGroup().location]",
      "properties": {
        "hardwareProfile": {
          "vmSize": "Standard_B2s"
        },
        "osProfile": {
          "computerName": "webvm",
          "adminUsername": "azureuser",
          "linuxConfiguration": {
            "disablePasswordAuthentication": true,
            "ssh": {
              "publicKeys": [
                {
                  "path": "/home/azureuser/.ssh/authorized_keys",
                  "keyData": "[parameters('sshPublicKey')]"
                }
              ]
            }
          }
        },
        "storageProfile": {
          "imageReference": {
            "publisher": "Canonical",
            "offer": "0001-com-ubuntu-server-focal",
            "sku": "20_04-lts-gen2",
            "version": "latest"
          },
          "osDisk": {
            "createOption": "FromImage",
            "managedDisk": {
              "storageAccountType": "Premium_LRS"
            }
          }
        },
        "networkProfile": {
          "networkInterfaces": [
            {
              "id": "[resourceId('Microsoft.Network/networkInterfaces', 'web-nic')]"
            }
          ]
        }
      }
    }
  ]
}
```

#### VM Scale Set
```bash
# Create VM scale set
az vmss create \
  --resource-group myResourceGroup \
  --name web-vmss \
  --image UbuntuLTS \
  --vm-sku Standard_B2s \
  --instance-count 3 \
  --vnet-name myVnet \
  --subnet mySubnet \
  --lb myLoadBalancer \
  --backend-pool-name myBackendPool \
  --upgrade-policy-mode automatic \
  --admin-username azureuser \
  --ssh-key-value ~/.ssh/id_rsa.pub

# Configure autoscale
az monitor autoscale create \
  --resource-group myResourceGroup \
  --resource web-vmss \
  --resource-type Microsoft.Compute/virtualMachineScaleSets \
  --name autoscale-config \
  --min-count 2 \
  --max-count 10 \
  --count 3

az monitor autoscale rule create \
  --resource-group myResourceGroup \
  --autoscale-name autoscale-config \
  --condition "Percentage CPU > 70 avg 5m" \
  --scale out 1
```

### Azure Kubernetes Service (AKS)

#### Cluster Creation
```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name production-aks \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --enable-addons monitoring \
  --network-plugin azure \
  --enable-managed-identity \
  --attach-acr myacr \
  --kubernetes-version 1.27.3

# Get credentials
az aks get-credentials \
  --resource-group myResourceGroup \
  --name production-aks
```

### Azure Container Instances

```bash
# Deploy container
az container create \
  --resource-group myResourceGroup \
  --name web-container \
  --image myacr.azurecr.io/web-app:latest \
  --cpu 1 \
  --memory 1.5 \
  --registry-login-server myacr.azurecr.io \
  --registry-username myacr \
  --registry-password $(az acr credential show --name myacr --query passwords[0].value -o tsv) \
  --dns-name-label web-app-unique \
  --ports 80 443 \
  --environment-variables NODE_ENV=production \
  --secure-environment-variables DB_PASSWORD=secret123
```

## Deployment Strategies

### Blue-Green Deployment

#### Using Load Balancer
```bash
# Deploy green environment
kubectl apply -f deployment-green.yaml

# Wait for green to be ready
kubectl wait --for=condition=available --timeout=300s deployment/app-green

# Switch traffic
kubectl patch service app-service \
  -p '{"spec":{"selector":{"version":"green"}}}'

# Verify and clean up blue
kubectl delete deployment app-blue
```

#### AWS Route 53 Weighted Routing
```json
{
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "app.example.com",
        "Type": "A",
        "SetIdentifier": "blue",
        "Weight": 0,
        "AliasTarget": {
          "HostedZoneId": "Z123456",
          "DNSName": "blue-lb.us-east-1.elb.amazonaws.com"
        }
      }
    },
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "app.example.com",
        "Type": "A",
        "SetIdentifier": "green",
        "Weight": 100,
        "AliasTarget": {
          "HostedZoneId": "Z123456",
          "DNSName": "green-lb.us-east-1.elb.amazonaws.com"
        }
      }
    }
  ]
}
```

### Canary Deployment

#### Kubernetes with Flagger
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: web-app
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  service:
    port: 8080
  analysis:
    interval: 1m
    threshold: 10
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
    webhooks:
    - name: load-test
      url: http://loadtester/
      timeout: 5s
```

#### AWS App Mesh
```yaml
apiVersion: appmesh.k8s.aws/v1beta2
kind: VirtualRouter
metadata:
  name: web-router
spec:
  listeners:
  - portMapping:
      port: 8080
      protocol: http
  routes:
  - name: web-route
    httpRoute:
      match:
        prefix: /
      action:
        weightedTargets:
        - virtualNodeRef:
            name: web-stable
          weight: 90
        - virtualNodeRef:
            name: web-canary
          weight: 10
```

### Rolling Update

#### Kubernetes
```yaml
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max 2 pods above desired
      maxUnavailable: 1  # Max 1 pod below desired
  minReadySeconds: 30
  progressDeadlineSeconds: 600
```

#### ECS
```json
{
  "deploymentConfiguration": {
    "maximumPercent": 200,
    "minimumHealthyPercent": 100,
    "deploymentCircuitBreaker": {
      "enable": true,
      "rollback": true
    }
  }
}
```

## Multi-Region Deployment

### Active-Active
```yaml
# Global load balancer routes to nearest region
# Each region handles production traffic
# Data replicated bidirectionally

# AWS Global Accelerator
aws globalaccelerator create-accelerator \
  --name my-app \
  --ip-address-type IPV4 \
  --enabled

# Add endpoint groups for multiple regions
aws globalaccelerator create-endpoint-group \
  --listener-arn $LISTENER_ARN \
  --endpoint-group-region us-east-1 \
  --endpoint-configurations EndpointId=$ALB_ARN,Weight=50
```

### Active-Passive
```yaml
# Primary region handles all traffic
# Secondary region on standby
# Failover on primary region failure

# Route 53 health check and failover
{
  "Type": "A",
  "SetIdentifier": "primary",
  "Failover": "PRIMARY",
  "HealthCheckId": "health-check-id",
  "AliasTarget": {
    "DNSName": "primary-lb.us-east-1.elb.amazonaws.com"
  }
}
```

## Best Practices

### Security
- Use IAM roles, not access keys
- Encrypt data at rest and in transit
- Enable VPC flow logs
- Use security groups restrictively
- Scan container images for vulnerabilities
- Rotate secrets regularly
- Enable audit logging

### Cost Optimization
- Right-size instances
- Use reserved/spot instances
- Auto-scale based on demand
- Delete unused resources
- Use S3 lifecycle policies
- Monitor and analyze costs

### High Availability
- Deploy across multiple AZs
- Use load balancers
- Implement health checks
- Auto-scaling groups
- Database replication
- Backup and disaster recovery

### Monitoring
- CloudWatch/Stackdriver metrics
- Application logs
- Distributed tracing
- Custom business metrics
- Alerting on SLOs

## Resources

- [AWS Well-Architected](https://aws.amazon.com/architecture/well-architected/)
- [GCP Best Practices](https://cloud.google.com/architecture/best-practices)
- [Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/)
