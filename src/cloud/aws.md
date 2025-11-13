# Amazon Web Services (AWS)

## Table of Contents
- [Introduction](#introduction)
- [AWS Global Infrastructure](#aws-global-infrastructure)
- [Getting Started](#getting-started)
- [Core Compute Services](#core-compute-services)
- [Storage Services](#storage-services)
- [Database Services](#database-services)
- [Networking Services](#networking-services)
- [Serverless Services](#serverless-services)
- [Container Services](#container-services)
- [Security Services](#security-services)
- [Monitoring and Management](#monitoring-and-management)
- [DevOps and CI/CD](#devops-and-cicd)
- [Machine Learning Services](#machine-learning-services)
- [Architecture Examples](#architecture-examples)
- [Cost Optimization](#cost-optimization)
- [Best Practices](#best-practices)
- [CLI Reference](#cli-reference)

## Introduction

Amazon Web Services (AWS) is the world's most comprehensive and broadly adopted cloud platform, offering over 200 fully featured services from data centers globally.

### Key Advantages
- **Market Leader**: Largest market share (~32%)
- **Mature Ecosystem**: Launched in 2006
- **Service Breadth**: 200+ services
- **Global Reach**: 30+ regions, 90+ availability zones
- **Innovation**: Rapid release of new features
- **Community**: Largest developer community

### AWS Account Structure

```
┌─────────────────────────────────────────────┐
│         AWS Organization (Root)              │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐    ┌──────────────┐      │
│  │  Production  │    │  Development │      │
│  │  OU          │    │  OU          │      │
│  ├──────────────┤    ├──────────────┤      │
│  │ Account 1    │    │ Account 3    │      │
│  │ Account 2    │    │ Account 4    │      │
│  └──────────────┘    └──────────────┘      │
│                                              │
│  ┌──────────────┐    ┌──────────────┐      │
│  │  Security    │    │  Sandbox     │      │
│  │  OU          │    │  OU          │      │
│  └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────┘
```

## AWS Global Infrastructure

### Hierarchy

```
Region
  └─ Availability Zones (AZs)
      └─ Data Centers
          └─ Edge Locations (CloudFront CDN)
```

### Key Concepts

**Region**: Geographic area with multiple AZs
- Examples: us-east-1 (Virginia), eu-west-1 (Ireland)
- Completely independent
- Data doesn't leave region unless explicitly configured

**Availability Zone**: Isolated data center(s) within a region
- 2-6 AZs per region
- Low-latency connections between AZs
- Physical separation for fault tolerance

**Edge Location**: CDN endpoint for CloudFront
- 400+ edge locations globally
- Caches content closer to users

### Region Selection Criteria

```
Factor              Consideration
──────────────────────────────────────────────
Latency             Distance to users
Compliance          Data residency laws
Services            Not all services in all regions
Cost                Pricing varies by region
```

## Getting Started

### AWS CLI Installation

```bash
# Install AWS CLI v2 (Linux/macOS)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version

# Configure AWS CLI
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json/yaml/text/table)

# Alternative: Use environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Or use AWS profiles
aws configure --profile production
aws s3 ls --profile production
```

### AWS CLI Configuration Files

```bash
# View configuration
cat ~/.aws/config
# [default]
# region = us-east-1
# output = json
#
# [profile production]
# region = us-west-2
# output = yaml

cat ~/.aws/credentials
# [default]
# aws_access_key_id = AKIAIOSFODNN7EXAMPLE
# aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
#
# [production]
# aws_access_key_id = AKIAI44QH8DHBEXAMPLE
# aws_secret_access_key = je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
```

### Basic AWS CLI Commands

```bash
# Get caller identity
aws sts get-caller-identity

# List all regions
aws ec2 describe-regions --output table

# List available services
aws help

# Get help for specific service
aws ec2 help
aws s3 help
```

## Core Compute Services

### Amazon EC2 (Elastic Compute Cloud)

Virtual servers in the cloud.

#### Instance Types

```
Category        Type         vCPU    Memory    Use Case
──────────────────────────────────────────────────────────────
General         t3.micro     2       1 GB      Development
Purpose         t3.medium    2       4 GB      Web servers
                m5.large     2       8 GB      Applications

Compute         c5.large     2       4 GB      Batch processing
Optimized       c5.xlarge    4       8 GB      High-performance

Memory          r5.large     2       16 GB     Databases
Optimized       r5.xlarge    4       32 GB     Caching

Storage         i3.large     2       15.25 GB  NoSQL databases
Optimized       d2.xlarge    4       30.5 GB   Data warehousing

GPU             p3.2xlarge   8       61 GB     ML training
Instances       g4dn.xlarge  4       16 GB     ML inference
```

#### EC2 Pricing Models

```
Model               Discount    Commitment    Use Case
─────────────────────────────────────────────────────────────
On-Demand           Baseline    None          Unpredictable
Reserved Instance   Up to 75%   1-3 years     Steady state
Spot Instance       Up to 90%   None          Fault-tolerant
Savings Plan        Up to 72%   1-3 years     Flexible
```

#### EC2 CLI Examples

```bash
# List all instances
aws ec2 describe-instances

# List instances with specific state
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress]' \
  --output table

# Launch an instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.micro \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyWebServer}]'

# Stop an instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start an instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Terminate an instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Create AMI from instance
aws ec2 create-image \
  --instance-id i-1234567890abcdef0 \
  --name "MyWebServer-Backup-$(date +%Y%m%d)" \
  --description "Backup of MyWebServer"

# List AMIs
aws ec2 describe-images --owners self

# Get instance metadata (from within instance)
curl http://169.254.169.254/latest/meta-data/
curl http://169.254.169.254/latest/meta-data/instance-id
curl http://169.254.169.254/latest/meta-data/public-ipv4
```

#### User Data Script Example

```bash
#!/bin/bash
# User data script for EC2 instance initialization

# Update system
yum update -y

# Install Apache web server
yum install -y httpd

# Start Apache
systemctl start httpd
systemctl enable httpd

# Create simple web page
echo "<h1>Hello from EC2!</h1>" > /var/www/html/index.html

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm
```

### Auto Scaling

Automatically adjust capacity to maintain performance and costs.

#### Auto Scaling Architecture

```
┌─────────────────────────────────────────────────────┐
│              Application Load Balancer              │
└──────────────────────┬──────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼────┐        ┌───▼────┐        ┌───▼────┐
│  EC2   │        │  EC2   │        │  EC2   │
│ (Min)  │        │ (Curr) │        │ (Max)  │
└────────┘        └────────┘        └────────┘
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Auto Scaling Group │
            │                     │
            │  Min: 2             │
            │  Desired: 3         │
            │  Max: 10            │
            │                     │
            │  Scale Up: CPU>70%  │
            │  Scale Down: CPU<30%│
            └─────────────────────┘
```

#### Auto Scaling CLI Examples

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name my-template \
  --version-description "Initial version" \
  --launch-template-data '{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "t3.micro",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-0123456789abcdef0"]
  }'

# Create Auto Scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --launch-template "LaunchTemplateName=my-template,Version=1" \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3 \
  --vpc-zone-identifier "subnet-12345,subnet-67890" \
  --target-group-arns arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067 \
  --health-check-type ELB \
  --health-check-grace-period 300

# Create scaling policy (target tracking)
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name my-asg \
  --policy-name cpu-target-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    },
    "TargetValue": 70.0
  }'

# Describe Auto Scaling groups
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names my-asg

# Update Auto Scaling group capacity
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name my-asg \
  --desired-capacity 5

# Delete Auto Scaling group
aws autoscaling delete-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --force-delete
```

### AWS Lambda (Serverless)

Run code without provisioning servers. Covered in detail in [Serverless Services](#serverless-services).

## Storage Services

### Amazon S3 (Simple Storage Service)

Object storage service with 99.999999999% (11 9's) durability.

#### S3 Storage Classes

```
Class                   Use Case                    Retrieval      Cost
────────────────────────────────────────────────────────────────────────
Standard                Frequently accessed         Instant        $$$
Intelligent-Tiering     Unknown/changing patterns   Instant        $$+
Standard-IA             Infrequently accessed       Instant        $$
One Zone-IA             Non-critical, infrequent    Instant        $
Glacier Instant         Archive, instant retrieval  Instant        $
Glacier Flexible        Archive, min-hour retrieval Minutes-Hours  ¢¢
Glacier Deep Archive    Long-term archive (7-10yr)  12 hours       ¢
```

#### S3 Architecture

```
┌─────────────────────────────────────────────┐
│  Bucket: my-application-bucket              │
│  Region: us-east-1                          │
├─────────────────────────────────────────────┤
│                                              │
│  /images/                                   │
│    ├─ logo.png                              │
│    └─ banner.jpg                            │
│                                              │
│  /documents/                                │
│    ├─ report.pdf                            │
│    └─ invoice.xlsx                          │
│                                              │
│  /backups/                                  │
│    └─ database-backup-2024-01-01.sql       │
│                                              │
│  Features:                                  │
│  ├─ Versioning: Enabled                    │
│  ├─ Encryption: AES-256                    │
│  ├─ Lifecycle: Move to Glacier after 90d   │
│  ├─ Replication: Cross-region enabled      │
│  └─ Access Logs: Enabled                   │
└─────────────────────────────────────────────┘
```

#### S3 CLI Examples

```bash
# Create bucket
aws s3 mb s3://my-unique-bucket-name-12345

# List buckets
aws s3 ls

# Upload file
aws s3 cp local-file.txt s3://my-bucket/
aws s3 cp local-file.txt s3://my-bucket/folder/

# Upload directory recursively
aws s3 cp ./my-directory s3://my-bucket/path/ --recursive

# Download file
aws s3 cp s3://my-bucket/file.txt ./

# Sync local directory with S3 (like rsync)
aws s3 sync ./local-dir s3://my-bucket/remote-dir/
aws s3 sync s3://my-bucket/remote-dir/ ./local-dir

# List objects in bucket
aws s3 ls s3://my-bucket/
aws s3 ls s3://my-bucket/folder/ --recursive

# Delete object
aws s3 rm s3://my-bucket/file.txt

# Delete all objects in folder
aws s3 rm s3://my-bucket/folder/ --recursive

# Make object public
aws s3api put-object-acl \
  --bucket my-bucket \
  --key file.txt \
  --acl public-read

# Generate presigned URL (temporary access)
aws s3 presign s3://my-bucket/private-file.pdf --expires-in 3600

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket my-bucket \
  --versioning-configuration Status=Enabled

# Enable server-side encryption
aws s3api put-bucket-encryption \
  --bucket my-bucket \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Set lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-bucket \
  --lifecycle-configuration file://lifecycle.json
```

#### S3 Lifecycle Policy Example

```json
{
  "Rules": [
    {
      "Id": "MoveOldFilesToGlacier",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "logs/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    },
    {
      "Id": "DeleteOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    }
  ]
}
```

#### S3 Bucket Policy Example

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-bucket/public/*"
    },
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

#### S3 SDK Example (Python/Boto3)

```python
import boto3
from botocore.exceptions import ClientError

# Create S3 client
s3 = boto3.client('s3')

# Upload file
def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    
    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"Uploaded {file_name} to {bucket}/{object_name}")
    except ClientError as e:
        print(f"Error: {e}")
        return False
    return True

# Download file
def download_file(bucket, object_name, file_name):
    try:
        s3.download_file(bucket, object_name, file_name)
        print(f"Downloaded {bucket}/{object_name} to {file_name}")
    except ClientError as e:
        print(f"Error: {e}")
        return False
    return True

# List objects
def list_objects(bucket, prefix=''):
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"{obj['Key']}: {obj['Size']} bytes")
    except ClientError as e:
        print(f"Error: {e}")

# Generate presigned URL
def create_presigned_url(bucket, object_name, expiration=3600):
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_name},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error: {e}")
        return None

# Usage
upload_file('local-file.txt', 'my-bucket', 'uploads/file.txt')
download_file('my-bucket', 'uploads/file.txt', 'downloaded-file.txt')
list_objects('my-bucket', 'uploads/')
url = create_presigned_url('my-bucket', 'uploads/file.txt')
print(f"Presigned URL: {url}")
```

### Amazon EBS (Elastic Block Store)

Block storage for EC2 instances.

#### EBS Volume Types

```
Type         IOPS          Throughput    Use Case              Cost
────────────────────────────────────────────────────────────────────
gp3          3,000-16,000  125-1000 MB/s General purpose       $$
gp2          3,000-16,000  Baseline      General purpose       $$
io2          64,000+       1,000 MB/s    Mission-critical DB   $$$$
io1          32,000+       500 MB/s      High-performance DB   $$$
st1          500           500 MB/s      Big data, logs        $
sc1          250           250 MB/s      Cold data             ¢
```

#### EBS CLI Examples

```bash
# Create EBS volume
aws ec2 create-volume \
  --volume-type gp3 \
  --size 100 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=MyVolume}]'

# List volumes
aws ec2 describe-volumes

# Attach volume to instance
aws ec2 attach-volume \
  --volume-id vol-0123456789abcdef0 \
  --instance-id i-1234567890abcdef0 \
  --device /dev/sdf

# Detach volume
aws ec2 detach-volume --volume-id vol-0123456789abcdef0

# Create snapshot
aws ec2 create-snapshot \
  --volume-id vol-0123456789abcdef0 \
  --description "Backup of MyVolume"

# List snapshots
aws ec2 describe-snapshots --owner-ids self

# Create volume from snapshot
aws ec2 create-volume \
  --snapshot-id snap-0123456789abcdef0 \
  --availability-zone us-east-1a

# Delete snapshot
aws ec2 delete-snapshot --snapshot-id snap-0123456789abcdef0

# Delete volume
aws ec2 delete-volume --volume-id vol-0123456789abcdef0
```

### Amazon EFS (Elastic File System)

Managed NFS file system for EC2.

```bash
# Create EFS file system
aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --encrypted \
  --tags Key=Name,Value=MyEFS

# Create mount target
aws efs create-mount-target \
  --file-system-id fs-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --security-groups sg-0123456789abcdef0

# Mount EFS on EC2 instance
sudo mkdir /mnt/efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
  fs-0123456789abcdef0.efs.us-east-1.amazonaws.com:/ /mnt/efs

# Add to /etc/fstab for persistent mount
echo "fs-0123456789abcdef0.efs.us-east-1.amazonaws.com:/ /mnt/efs nfs4 defaults,_netdev 0 0" | sudo tee -a /etc/fstab
```

## Database Services

### Amazon RDS (Relational Database Service)

Managed relational databases.

#### Supported Engines

```
Engine          Versions        Use Case
───────────────────────────────────────────────────────
MySQL           5.7, 8.0        Web applications
PostgreSQL      11-15           Advanced features
MariaDB         10.3-10.6       MySQL alternative
Oracle          12c, 19c        Enterprise apps
SQL Server      2016-2022       Microsoft stack
Amazon Aurora   MySQL/PG compat High performance
```

#### RDS Architecture (Multi-AZ)

```
┌─────────────────────────────────────────────────┐
│             Application Servers                  │
└───────────────────┬─────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   RDS Endpoint      │
         │   (DNS CNAME)       │
         └──────────┬──────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼────┐    Sync Repl    ┌───────▼─────┐
│Primary │◄──────────────►│  Standby    │
│Instance│                 │  Instance   │
│(AZ-A)  │                 │  (AZ-B)     │
└────────┘                 └─────────────┘
    │                            │
    │      Automatic Failover    │
    └────────────────────────────┘
```

#### RDS CLI Examples

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier mydb \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 14.7 \
  --master-username admin \
  --master-user-password MySecurePassword123 \
  --allocated-storage 20 \
  --storage-type gp3 \
  --vpc-security-group-ids sg-0123456789abcdef0 \
  --db-subnet-group-name my-db-subnet-group \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "sun:04:00-sun:05:00" \
  --multi-az \
  --storage-encrypted \
  --enable-cloudwatch-logs-exports '["postgresql"]'

# List RDS instances
aws rds describe-db-instances

# Get specific instance details
aws rds describe-db-instances \
  --db-instance-identifier mydb \
  --query 'DBInstances[0].[DBInstanceIdentifier,DBInstanceStatus,Endpoint.Address,Endpoint.Port]'

# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier mydb-replica \
  --source-db-instance-identifier mydb \
  --db-instance-class db.t3.micro \
  --availability-zone us-east-1b

# Create snapshot
aws rds create-db-snapshot \
  --db-instance-identifier mydb \
  --db-snapshot-identifier mydb-snapshot-$(date +%Y%m%d)

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mydb-restored \
  --db-snapshot-identifier mydb-snapshot-20240101

# Modify instance
aws rds modify-db-instance \
  --db-instance-identifier mydb \
  --db-instance-class db.t3.small \
  --apply-immediately

# Stop instance (up to 7 days)
aws rds stop-db-instance --db-instance-identifier mydb

# Start instance
aws rds start-db-instance --db-instance-identifier mydb

# Delete instance
aws rds delete-db-instance \
  --db-instance-identifier mydb \
  --skip-final-snapshot
  # Or with final snapshot:
  # --final-db-snapshot-identifier mydb-final-snapshot

# Connect to RDS
psql -h mydb.c9akciq32.us-east-1.rds.amazonaws.com -U admin -d postgres
mysql -h mydb.c9akciq32.us-east-1.rds.amazonaws.com -u admin -p
```

### Amazon DynamoDB

Fully managed NoSQL database.

#### DynamoDB Concepts

```
Table: Users
┌──────────────┬─────────────┬───────────┬─────────┬──────────┐
│ UserId (PK)  │ Email (SK)  │ Name      │ Age     │ Status   │
├──────────────┼─────────────┼───────────┼─────────┼──────────┤
│ user-001     │ a@ex.com    │ Alice     │ 30      │ active   │
│ user-002     │ b@ex.com    │ Bob       │ 25      │ active   │
│ user-003     │ c@ex.com    │ Charlie   │ 35      │ inactive │
└──────────────┴─────────────┴───────────┴─────────┴──────────┘

PK = Partition Key (required, determines data distribution)
SK = Sort Key (optional, enables range queries)
```

#### DynamoDB Capacity Modes

```
Mode          Billing      Use Case                Cost
─────────────────────────────────────────────────────────────
On-Demand     Per request  Unpredictable traffic   $$$$
Provisioned   Per hour     Predictable traffic     $$-$$$
  + Auto      Per hour     Variable patterns       $$-$$$
    Scaling
```

#### DynamoDB CLI Examples

```bash
# Create table
aws dynamodb create-table \
  --table-name Users \
  --attribute-definitions \
    AttributeName=UserId,AttributeType=S \
    AttributeName=Email,AttributeType=S \
  --key-schema \
    AttributeName=UserId,KeyType=HASH \
    AttributeName=Email,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --tags Key=Environment,Value=Production

# List tables
aws dynamodb list-tables

# Describe table
aws dynamodb describe-table --table-name Users

# Put item
aws dynamodb put-item \
  --table-name Users \
  --item '{
    "UserId": {"S": "user-001"},
    "Email": {"S": "alice@example.com"},
    "Name": {"S": "Alice"},
    "Age": {"N": "30"},
    "Status": {"S": "active"}
  }'

# Get item
aws dynamodb get-item \
  --table-name Users \
  --key '{
    "UserId": {"S": "user-001"},
    "Email": {"S": "alice@example.com"}
  }'

# Query items (by partition key)
aws dynamodb query \
  --table-name Users \
  --key-condition-expression "UserId = :userId" \
  --expression-attribute-values '{
    ":userId": {"S": "user-001"}
  }'

# Scan table (read all items - expensive!)
aws dynamodb scan --table-name Users

# Update item
aws dynamodb update-item \
  --table-name Users \
  --key '{
    "UserId": {"S": "user-001"},
    "Email": {"S": "alice@example.com"}
  }' \
  --update-expression "SET #status = :newStatus, Age = Age + :inc" \
  --expression-attribute-names '{"#status": "Status"}' \
  --expression-attribute-values '{
    ":newStatus": {"S": "inactive"},
    ":inc": {"N": "1"}
  }'

# Delete item
aws dynamodb delete-item \
  --table-name Users \
  --key '{
    "UserId": {"S": "user-001"},
    "Email": {"S": "alice@example.com"}
  }'

# Batch write
aws dynamodb batch-write-item --request-items file://batch-write.json

# Create global secondary index
aws dynamodb update-table \
  --table-name Users \
  --attribute-definitions AttributeName=Status,AttributeType=S \
  --global-secondary-index-updates '[{
    "Create": {
      "IndexName": "StatusIndex",
      "KeySchema": [{"AttributeName": "Status", "KeyType": "HASH"}],
      "Projection": {"ProjectionType": "ALL"},
      "ProvisionedThroughput": {
        "ReadCapacityUnits": 5,
        "WriteCapacityUnits": 5
      }
    }
  }]'

# Enable Point-in-Time Recovery
aws dynamodb update-continuous-backups \
  --table-name Users \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

#### DynamoDB SDK Example (Python/Boto3)

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

# Create DynamoDB resource
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

# Put item
def create_user(user_id, email, name, age):
    response = table.put_item(
        Item={
            'UserId': user_id,
            'Email': email,
            'Name': name,
            'Age': age,
            'Status': 'active'
        }
    )
    return response

# Get item
def get_user(user_id, email):
    response = table.get_item(
        Key={
            'UserId': user_id,
            'Email': email
        }
    )
    return response.get('Item')

# Query by partition key
def get_user_emails(user_id):
    response = table.query(
        KeyConditionExpression=Key('UserId').eq(user_id)
    )
    return response['Items']

# Query with sort key condition
def get_user_by_email_prefix(user_id, email_prefix):
    response = table.query(
        KeyConditionExpression=Key('UserId').eq(user_id) & 
                               Key('Email').begins_with(email_prefix)
    )
    return response['Items']

# Scan with filter
def get_active_users():
    response = table.scan(
        FilterExpression=Attr('Status').eq('active')
    )
    return response['Items']

# Update item
def update_user_status(user_id, email, new_status):
    response = table.update_item(
        Key={
            'UserId': user_id,
            'Email': email
        },
        UpdateExpression='SET #status = :status',
        ExpressionAttributeNames={
            '#status': 'Status'
        },
        ExpressionAttributeValues={
            ':status': new_status
        },
        ReturnValues='ALL_NEW'
    )
    return response['Attributes']

# Batch write
def batch_create_users(users):
    with table.batch_writer() as batch:
        for user in users:
            batch.put_item(Item=user)

# Usage
create_user('user-001', 'alice@example.com', 'Alice', 30)
user = get_user('user-001', 'alice@example.com')
print(user)

emails = get_user_emails('user-001')
update_user_status('user-001', 'alice@example.com', 'inactive')
```

### Amazon ElastiCache

Managed in-memory cache (Redis/Memcached).

```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id my-redis-cluster \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-nodes 1 \
  --cache-subnet-group-name my-cache-subnet-group \
  --security-group-ids sg-0123456789abcdef0

# Create Redis replication group (cluster mode)
aws elasticache create-replication-group \
  --replication-group-id my-redis-cluster \
  --replication-group-description "My Redis cluster" \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-clusters 3 \
  --automatic-failover-enabled \
  --multi-az-enabled

# Describe clusters
aws elasticache describe-cache-clusters \
  --show-cache-node-info

# Get endpoint
aws elasticache describe-cache-clusters \
  --cache-cluster-id my-redis-cluster \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint'
```

## Networking Services

### Amazon VPC (Virtual Private Cloud)

Isolated network for your AWS resources.

#### VPC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  VPC: 10.0.0.0/16                                           │
│  Region: us-east-1                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────┐  ┌──────────────────────┐   │
│  │ Public Subnet (AZ-A)      │  │ Public Subnet (AZ-B) │   │
│  │ 10.0.1.0/24               │  │ 10.0.2.0/24          │   │
│  │                           │  │                      │   │
│  │ ┌─────┐ ┌──────┐         │  │ ┌─────┐ ┌──────┐    │   │
│  │ │ NAT │ │ ALB  │         │  │ │ NAT │ │ ALB  │    │   │
│  │ └─────┘ └──────┘         │  │ └─────┘ └──────┘    │   │
│  └───────────┬───────────────┘  └──────────┬───────────┘   │
│              │                              │               │
│              │     Internet Gateway         │               │
│              └──────────────┬───────────────┘               │
│                             │                               │
│  ┌───────────────────────────┐  ┌──────────────────────┐   │
│  │ Private Subnet (AZ-A)     │  │ Private Subnet (AZ-B)│   │
│  │ 10.0.11.0/24              │  │ 10.0.12.0/24         │   │
│  │                           │  │                      │   │
│  │ ┌─────┐ ┌─────┐          │  │ ┌─────┐ ┌─────┐     │   │
│  │ │ EC2 │ │ EC2 │          │  │ │ EC2 │ │ EC2 │     │   │
│  │ └─────┘ └─────┘          │  │ └─────┘ └─────┘     │   │
│  └───────────────────────────┘  └──────────────────────┘   │
│                                                              │
│  ┌───────────────────────────┐  ┌──────────────────────┐   │
│  │ Database Subnet (AZ-A)    │  │ Database Subnet (AZ-B│   │
│  │ 10.0.21.0/24              │  │ 10.0.22.0/24         │   │
│  │                           │  │                      │   │
│  │ ┌─────────┐               │  │ ┌─────────┐         │   │
│  │ │   RDS   │               │  │ │   RDS   │         │   │
│  │ └─────────┘               │  │ └─────────┘         │   │
│  └───────────────────────────┘  └──────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### VPC CLI Examples

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# Create Internet Gateway
aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MyIGW}]'

# Attach Internet Gateway to VPC
aws ec2 attach-internet-gateway \
  --internet-gateway-id igw-0123456789abcdef0 \
  --vpc-id vpc-0123456789abcdef0

# Create public subnet
aws ec2 create-subnet \
  --vpc-id vpc-0123456789abcdef0 \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=PublicSubnet-AZ-A}]'

# Create private subnet
aws ec2 create-subnet \
  --vpc-id vpc-0123456789abcdef0 \
  --cidr-block 10.0.11.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=PrivateSubnet-AZ-A}]'

# Create route table
aws ec2 create-route-table \
  --vpc-id vpc-0123456789abcdef0 \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=PublicRouteTable}]'

# Create route to Internet Gateway
aws ec2 create-route \
  --route-table-id rtb-0123456789abcdef0 \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id igw-0123456789abcdef0

# Associate route table with subnet
aws ec2 associate-route-table \
  --route-table-id rtb-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0

# Create NAT Gateway (for private subnet internet access)
# First, allocate Elastic IP
aws ec2 allocate-address --domain vpc

# Create NAT Gateway in public subnet
aws ec2 create-nat-gateway \
  --subnet-id subnet-0123456789abcdef0 \
  --allocation-id eipalloc-0123456789abcdef0 \
  --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=MyNATGateway}]'

# Create route to NAT Gateway for private subnet
aws ec2 create-route \
  --route-table-id rtb-private-0123456789abcdef0 \
  --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id nat-0123456789abcdef0

# Create security group
aws ec2 create-security-group \
  --group-name web-server-sg \
  --description "Security group for web servers" \
  --vpc-id vpc-0123456789abcdef0

# Add inbound rules
aws ec2 authorize-security-group-ingress \
  --group-id sg-0123456789abcdef0 \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-0123456789abcdef0 \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-0123456789abcdef0 \
  --protocol tcp \
  --port 22 \
  --cidr 10.0.0.0/16

# List VPCs
aws ec2 describe-vpcs

# List subnets
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-0123456789abcdef0"

# List security groups
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=vpc-0123456789abcdef0"
```

### Elastic Load Balancing (ELB)

Distribute traffic across multiple targets.

#### Load Balancer Types

```
Type                Use Case                    OSI Layer    Cost
──────────────────────────────────────────────────────────────────
Application (ALB)   HTTP/HTTPS, path routing   Layer 7      $$
Network (NLB)       TCP/UDP, ultra performance Layer 4      $$
Gateway (GWLB)      Third-party appliances     Layer 3      $$$
Classic (CLB)       Legacy (deprecated)        Layer 4/7    $
```

#### ALB Architecture

```
                         Internet
                             │
                    ┌────────▼────────┐
                    │ Application     │
                    │ Load Balancer   │
                    │ (ALB)           │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │  Target   │      │  Target   │     │  Target   │
    │  Group 1  │      │  Group 2  │     │  Group 3  │
    │           │      │           │     │           │
    │ /api/*    │      │ /images/* │     │  /*       │
    └───────────┘      └───────────┘     └───────────┘
         │                  │                  │
    API Servers      Image Service     Web Servers
```

#### Load Balancer CLI Examples

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name my-alb \
  --subnets subnet-0123456789abcdef0 subnet-0123456789abcdef1 \
  --security-groups sg-0123456789abcdef0 \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4

# Create target group
aws elbv2 create-target-group \
  --name my-targets \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-0123456789abcdef0 \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 2

# Register targets
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067 \
  --targets Id=i-1234567890abcdef0 Id=i-0987654321abcdef0

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:region:account-id:loadbalancer/app/my-alb/50dc6c495c0c9188 \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067

# Create HTTPS listener with certificate
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:region:account-id:loadbalancer/app/my-alb/50dc6c495c0c9188 \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:region:account-id:certificate/12345678-1234-1234-1234-123456789012 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067

# Create path-based routing rule
aws elbv2 create-rule \
  --listener-arn arn:aws:elasticloadbalancing:region:account-id:listener/app/my-alb/50dc6c495c0c9188/f2f7dc8efc522ab2 \
  --priority 10 \
  --conditions Field=path-pattern,Values='/api/*' \
  --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:region:account-id:targetgroup/api-targets/73e2d6bc24d8a067

# Describe load balancers
aws elbv2 describe-load-balancers

# Describe target health
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067
```

### Amazon Route 53

Scalable DNS and domain registration.

```bash
# List hosted zones
aws route53 list-hosted-zones

# Create hosted zone
aws route53 create-hosted-zone \
  --name example.com \
  --caller-reference $(date +%s)

# Create A record
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [{"Value": "192.0.2.1"}]
      }
    }]
  }'

# Create CNAME record
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "blog.example.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "www.example.com"}]
      }
    }]
  }'

# Create alias record (to ALB)
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z35SXDOTRQ7X7K",
          "DNSName": "my-alb-1234567890.us-east-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# Health check for failover
aws route53 create-health-check \
  --health-check-config \
    IPAddress=192.0.2.1,Port=80,Type=HTTP,ResourcePath=/health,RequestInterval=30,FailureThreshold=3
```

## Serverless Services

### AWS Lambda

Run code without managing servers.

#### Lambda Architecture

```
┌────────────────────────────────────────────────┐
│            Event Sources                       │
├────────────────────────────────────────────────┤
│                                                 │
│  API Gateway │ S3 │ DynamoDB │ SQS │ EventBridge │
│                                                 │
└──────────────┬──────────┬──────────┬───────────┘
               │          │          │
          ┌────▼────┐┌────▼────┐┌───▼──────┐
          │ Lambda  ││ Lambda  ││  Lambda  │
          │Function ││Function ││ Function │
          │   1     ││    2    ││     3    │
          └────┬────┘└────┬────┘└────┬─────┘
               │          │          │
          ┌────▼──────────▼──────────▼─────┐
          │       Destinations              │
          │                                  │
          │  DynamoDB │ S3 │ SNS │ SQS     │
          └──────────────────────────────────┘
```

#### Lambda Function Example (Python)

```python
import json
import boto3

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('MyTable')

def lambda_handler(event, context):
    """
    Lambda function handler
    
    Args:
        event: Event data passed to the function
        context: Runtime information
    
    Returns:
        Response object
    """
    
    # Log the event
    print(f"Event: {json.dumps(event)}")
    
    # Example: Process S3 event
    if 'Records' in event:
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            print(f"Processing {key} from {bucket}")
            
            # Process the file
            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read().decode('utf-8')
                
                # Store metadata in DynamoDB
                table.put_item(
                    Item={
                        'file_key': key,
                        'bucket': bucket,
                        'size': response['ContentLength'],
                        'content_type': response['ContentType']
                    }
                )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps('Successfully processed file')
                }
            except Exception as e:
                print(f"Error: {str(e)}")
                return {
                    'statusCode': 500,
                    'body': json.dumps(f'Error processing file: {str(e)}')
                }
    
    # Example: Process API Gateway event
    if 'httpMethod' in event:
        http_method = event['httpMethod']
        path = event['path']
        
        if http_method == 'GET' and path == '/items':
            # Retrieve items from DynamoDB
            response = table.scan()
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response['Items'])
            }
        
        elif http_method == 'POST' and path == '/items':
            # Create new item
            body = json.loads(event['body'])
            
            table.put_item(Item=body)
            
            return {
                'statusCode': 201,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'message': 'Item created'})
            }
    
    return {
        'statusCode': 400,
        'body': json.dumps('Invalid request')
    }
```

#### Lambda CLI Examples

```bash
# Create Lambda function
zip function.zip lambda_function.py

aws lambda create-function \
  --function-name my-function \
  --runtime python3.11 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment Variables={ENV=production,DB_TABLE=MyTable}

# Update function code
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip

# Update function configuration
aws lambda update-function-configuration \
  --function-name my-function \
  --timeout 60 \
  --memory-size 512

# Invoke function synchronously
aws lambda invoke \
  --function-name my-function \
  --payload '{"key": "value"}' \
  response.json

cat response.json

# Invoke function asynchronously
aws lambda invoke \
  --function-name my-function \
  --invocation-type Event \
  --payload '{"key": "value"}' \
  response.json

# List functions
aws lambda list-functions

# Get function details
aws lambda get-function --function-name my-function

# Add S3 trigger
aws lambda add-permission \
  --function-name my-function \
  --statement-id s3-invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::my-bucket

aws s3api put-bucket-notification-configuration \
  --bucket my-bucket \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [{
      "LambdaFunctionArn": "arn:aws:lambda:region:account-id:function:my-function",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [{
            "Name": "prefix",
            "Value": "uploads/"
          }]
        }
      }
    }]
  }'

# View logs
aws logs tail /aws/lambda/my-function --follow

# Create layer
zip layer.zip -r python/

aws lambda publish-layer-version \
  --layer-name my-layer \
  --description "Common dependencies" \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.11

# Add layer to function
aws lambda update-function-configuration \
  --function-name my-function \
  --layers arn:aws:lambda:region:account-id:layer:my-layer:1

# Delete function
aws lambda delete-function --function-name my-function
```

#### Lambda Pricing

```
Component               Price (us-east-1)
─────────────────────────────────────────────────
Requests                $0.20 per 1M requests
Duration (x86)          $0.0000166667 per GB-second
Duration (ARM/Graviton) $0.0000133334 per GB-second
Free Tier               1M requests + 400,000 GB-seconds/month

Example: 1 million requests, 512 MB, 1 second each
= 1M * $0.20/1M = $0.20 (requests)
+ 1M * 0.5 GB * 1 sec * $0.0000166667 = $8.33 (duration)
= $8.53/month (minus free tier)
```

### API Gateway

Create, publish, and manage APIs.

```bash
# Create REST API
aws apigateway create-rest-api \
  --name "My API" \
  --description "My REST API" \
  --endpoint-configuration types=REGIONAL

# Get root resource
aws apigateway get-resources \
  --rest-api-id abc123

# Create resource
aws apigateway create-resource \
  --rest-api-id abc123 \
  --parent-id xyz789 \
  --path-part items

# Create method
aws apigateway put-method \
  --rest-api-id abc123 \
  --resource-id uvw456 \
  --http-method GET \
  --authorization-type NONE

# Create Lambda integration
aws apigateway put-integration \
  --rest-api-id abc123 \
  --resource-id uvw456 \
  --http-method GET \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:region:lambda:path/2015-03-31/functions/arn:aws:lambda:region:account-id:function:my-function/invocations

# Deploy API
aws apigateway create-deployment \
  --rest-api-id abc123 \
  --stage-name prod

# API URL format:
# https://abc123.execute-api.region.amazonaws.com/prod/items

# Enable API key
aws apigateway create-api-key \
  --name "My API Key" \
  --enabled

# Create usage plan
aws apigateway create-usage-plan \
  --name "Basic Plan" \
  --throttle burstLimit=100,rateLimit=50 \
  --quota limit=10000,period=MONTH

# Associate API key with usage plan
aws apigateway create-usage-plan-key \
  --usage-plan-id def456 \
  --key-id ghi789 \
  --key-type API_KEY
```

## Container Services

### Amazon ECS (Elastic Container Service)

Container orchestration service.

#### ECS Architecture

```
┌─────────────────────────────────────────────────┐
│              ECS Cluster                         │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │         ECS Service                      │   │
│  │  (Desired Count: 3)                      │   │
│  └──────────┬──────────────────────────────┘   │
│             │                                    │
│    ┌────────┼────────┐                          │
│    │        │        │                          │
│ ┌──▼──┐  ┌──▼──┐  ┌──▼──┐                      │
│ │Task │  │Task │  │Task │                      │
│ │  1  │  │  2  │  │  3  │                      │
│ └──┬──┘  └──┬──┘  └──┬──┘                      │
│    │        │        │                          │
│ ┌──▼───────▼────────▼───┐                      │
│ │    Container(s)        │                      │
│ │  ┌────────────────┐   │                      │
│ │  │  nginx:latest  │   │                      │
│ │  └────────────────┘   │                      │
│ └────────────────────────┘                      │
│                                                  │
│  Launch Type: EC2 or Fargate                    │
└─────────────────────────────────────────────────┘
```

#### ECS Task Definition Example

```json
{
  "family": "web-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "nginx",
      "image": "nginx:latest",
      "portMappings": [
        {
          "containerPort": 80,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/web-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "nginx"
        }
      }
    }
  ]
}
```

#### ECS CLI Examples

```bash
# Create cluster (Fargate)
aws ecs create-cluster --cluster-name my-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster my-cluster \
  --service-name web-service \
  --task-definition web-app:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:region:account-id:targetgroup/my-targets/73e2d6bc24d8a067,containerName=nginx,containerPort=80"

# List services
aws ecs list-services --cluster my-cluster

# Describe service
aws ecs describe-services \
  --cluster my-cluster \
  --services web-service

# Update service (e.g., change desired count)
aws ecs update-service \
  --cluster my-cluster \
  --service web-service \
  --desired-count 5

# Run standalone task
aws ecs run-task \
  --cluster my-cluster \
  --task-definition web-app:1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"

# View logs
aws logs tail /ecs/web-app --follow

# Stop task
aws ecs stop-task \
  --cluster my-cluster \
  --task arn:aws:ecs:region:account-id:task/my-cluster/abc123

# Delete service
aws ecs delete-service \
  --cluster my-cluster \
  --service web-service \
  --force

# Delete cluster
aws ecs delete-cluster --cluster my-cluster
```

### Amazon EKS (Elastic Kubernetes Service)

Managed Kubernetes service.

```bash
# Create EKS cluster (using eksctl - easier)
eksctl create cluster \
  --name my-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Or using AWS CLI (more complex)
aws eks create-cluster \
  --name my-cluster \
  --role-arn arn:aws:iam::123456789012:role/eks-service-role \
  --resources-vpc-config subnetIds=subnet-12345,subnet-67890,securityGroupIds=sg-12345

# Update kubeconfig
aws eks update-kubeconfig --name my-cluster --region us-east-1

# Verify connection
kubectl get nodes

# Deploy application
kubectl apply -f deployment.yaml

# List clusters
aws eks list-clusters

# Describe cluster
aws eks describe-cluster --name my-cluster

# Delete cluster (eksctl)
eksctl delete cluster --name my-cluster
```

### AWS Fargate

Serverless compute for containers (works with ECS and EKS).

**Benefits**:
- No EC2 instances to manage
- Pay only for resources used
- Automatic scaling
- Built-in security

**Use Cases**:
- Microservices
- Batch processing
- CI/CD tasks
- Event-driven applications

## Security Services

### AWS IAM (Identity and Access Management)

Control access to AWS resources.

#### IAM Concepts

```
┌─────────────────────────────────────────┐
│ AWS Account                              │
├─────────────────────────────────────────┤
│                                          │
│  Users          Groups         Roles    │
│  ├─ Alice      ├─ Developers  ├─ EC2   │
│  ├─ Bob        ├─ Admins      ├─ Lambda│
│  └─ Charlie    └─ Viewers     └─ ECS   │
│                                          │
│  Policies (JSON documents)               │
│  ├─ Managed Policies (AWS/Custom)       │
│  └─ Inline Policies                     │
└─────────────────────────────────────────┘
```

#### IAM Policy Example

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3ReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*"
    },
    {
      "Sid": "AllowS3ListBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::my-bucket"
    },
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

#### IAM CLI Examples

```bash
# Create user
aws iam create-user --user-name alice

# Create access key
aws iam create-access-key --user-name alice

# Create group
aws iam create-group --group-name developers

# Add user to group
aws iam add-user-to-group \
  --user-name alice \
  --group-name developers

# Create policy
aws iam create-policy \
  --policy-name S3ReadWritePolicy \
  --policy-document file://policy.json

# Attach policy to user
aws iam attach-user-policy \
  --user-name alice \
  --policy-arn arn:aws:iam::123456789012:policy/S3ReadWritePolicy

# Attach policy to group
aws iam attach-group-policy \
  --group-name developers \
  --policy-arn arn:aws:iam::aws:policy/PowerUserAccess

# Create role (for EC2)
aws iam create-role \
  --role-name EC2-S3-Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policy to role
aws iam attach-role-policy \
  --role-name EC2-S3-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name EC2-S3-Profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name EC2-S3-Profile \
  --role-name EC2-S3-Role

# Associate instance profile with EC2
aws ec2 associate-iam-instance-profile \
  --instance-id i-1234567890abcdef0 \
  --iam-instance-profile Name=EC2-S3-Profile

# List users
aws iam list-users

# List policies attached to user
aws iam list-attached-user-policies --user-name alice

# Delete user (must remove from groups and detach policies first)
aws iam remove-user-from-group --user-name alice --group-name developers
aws iam detach-user-policy --user-name alice --policy-arn arn:aws:iam::123456789012:policy/S3ReadWritePolicy
aws iam delete-user --user-name alice
```

### AWS Secrets Manager

Store and rotate secrets.

```bash
# Create secret
aws secretsmanager create-secret \
  --name prod/db/password \
  --description "Database password for production" \
  --secret-string '{"username":"admin","password":"MySecurePassword123"}'

# Get secret value
aws secretsmanager get-secret-value --secret-id prod/db/password

# Update secret
aws secretsmanager update-secret \
  --secret-id prod/db/password \
  --secret-string '{"username":"admin","password":"NewPassword456"}'

# Enable automatic rotation
aws secretsmanager rotate-secret \
  --secret-id prod/db/password \
  --rotation-lambda-arn arn:aws:lambda:region:account-id:function:my-rotation-function \
  --rotation-rules AutomaticallyAfterDays=30

# Delete secret (with recovery window)
aws secretsmanager delete-secret \
  --secret-id prod/db/password \
  --recovery-window-in-days 30
```

#### Use Secret in Lambda (Python)

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        raise

def lambda_handler(event, context):
    # Get database credentials
    db_secret = get_secret('prod/db/password')
    
    username = db_secret['username']
    password = db_secret['password']
    
    # Use credentials to connect to database
    # ...
    
    return {'statusCode': 200}
```

### AWS KMS (Key Management Service)

Manage encryption keys.

```bash
# Create KMS key
aws kms create-key \
  --description "Application data encryption key"

# Create alias
aws kms create-alias \
  --alias-name alias/app-data-key \
  --target-key-id 1234abcd-12ab-34cd-56ef-1234567890ab

# Encrypt data
aws kms encrypt \
  --key-id alias/app-data-key \
  --plaintext "sensitive data" \
  --output text \
  --query CiphertextBlob

# Decrypt data
aws kms decrypt \
  --ciphertext-blob fileb://encrypted-data \
  --output text \
  --query Plaintext | base64 --decode

# List keys
aws kms list-keys

# Enable key rotation
aws kms enable-key-rotation --key-id 1234abcd-12ab-34cd-56ef-1234567890ab
```

## Monitoring and Management

### Amazon CloudWatch

Monitoring and observability service.

#### CloudWatch Metrics

```bash
# Put custom metric
aws cloudwatch put-metric-data \
  --namespace "MyApp" \
  --metric-name "RequestCount" \
  --value 100 \
  --timestamp $(date -u +"%Y-%m-%dT%H:%M:%S")

# Get metric statistics
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
  --start-time $(date -u -d '1 hour ago' +"%Y-%m-%dT%H:%M:%S") \
  --end-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
  --period 300 \
  --statistics Average

# Create alarm
aws cloudwatch put-metric-alarm \
  --alarm-name high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
  --alarm-actions arn:aws:sns:region:account-id:my-topic

# List alarms
aws cloudwatch describe-alarms

# Delete alarm
aws cloudwatch delete-alarms --alarm-names high-cpu
```

#### CloudWatch Logs

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/lambda/my-function

# Create log stream
aws logs create-log-stream \
  --log-group-name /aws/lambda/my-function \
  --log-stream-name 2024/01/01/instance-123

# Put log events
aws logs put-log-events \
  --log-group-name /aws/lambda/my-function \
  --log-stream-name 2024/01/01/instance-123 \
  --log-events timestamp=$(date +%s000),message="Application started"

# Tail logs
aws logs tail /aws/lambda/my-function --follow

# Filter logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/my-function \
  --filter-pattern "ERROR" \
  --start-time $(date -d '1 hour ago' +%s)000

# Create metric filter
aws logs put-metric-filter \
  --log-group-name /aws/lambda/my-function \
  --filter-name ErrorCount \
  --filter-pattern "[ERROR]" \
  --metric-transformations \
    metricName=ErrorCount,metricNamespace=MyApp,metricValue=1

# Export logs to S3
aws logs create-export-task \
  --log-group-name /aws/lambda/my-function \
  --from $(date -d '1 day ago' +%s)000 \
  --to $(date +%s)000 \
  --destination my-logs-bucket \
  --destination-prefix lambda-logs/

# Set retention policy
aws logs put-retention-policy \
  --log-group-name /aws/lambda/my-function \
  --retention-in-days 30

# Delete log group
aws logs delete-log-group --log-group-name /aws/lambda/my-function
```

### AWS CloudTrail

Track user activity and API usage.

```bash
# Create trail
aws cloudtrail create-trail \
  --name my-trail \
  --s3-bucket-name my-cloudtrail-bucket

# Start logging
aws cloudtrail start-logging --name my-trail

# Lookup events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
  --max-results 10

# Get trail status
aws cloudtrail get-trail-status --name my-trail

# Stop logging
aws cloudtrail stop-logging --name my-trail

# Delete trail
aws cloudtrail delete-trail --name my-trail
```

## DevOps and CI/CD

### AWS CodeCommit

Git repository hosting.

```bash
# Create repository
aws codecommit create-repository \
  --repository-name my-repo \
  --repository-description "My application code"

# Clone repository
git clone https://git-codecommit.us-east-1.amazonaws.com/v1/repos/my-repo

# Or with SSH
git clone ssh://git-codecommit.us-east-1.amazonaws.com/v1/repos/my-repo

# List repositories
aws codecommit list-repositories

# Get repository details
aws codecommit get-repository --repository-name my-repo

# Delete repository
aws codecommit delete-repository --repository-name my-repo
```

### AWS CodeBuild

Build and test code.

#### buildspec.yml Example

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      - echo "Installing dependencies..."
      - pip install -r requirements.txt
  
  pre_build:
    commands:
      - echo "Running tests..."
      - pytest tests/
      - echo "Logging in to Amazon ECR..."
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
  
  build:
    commands:
      - echo "Building Docker image..."
      - docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .
      - docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
  
  post_build:
    commands:
      - echo "Pushing Docker image..."
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
      - echo "Build completed on `date`"

artifacts:
  files:
    - '**/*'
  name: build-output

cache:
  paths:
    - '/root/.cache/pip/**/*'
```

```bash
# Create build project
aws codebuild create-project \
  --name my-build-project \
  --source type=CODECOMMIT,location=https://git-codecommit.us-east-1.amazonaws.com/v1/repos/my-repo \
  --artifacts type=S3,location=my-build-artifacts-bucket \
  --environment type=LINUX_CONTAINER,image=aws/codebuild/standard:5.0,computeType=BUILD_GENERAL1_SMALL \
  --service-role arn:aws:iam::123456789012:role/codebuild-service-role

# Start build
aws codebuild start-build --project-name my-build-project

# Get build details
aws codebuild batch-get-builds --ids my-build-project:build-id
```

### AWS CodeDeploy

Automate application deployments.

```bash
# Create application
aws deploy create-application \
  --application-name my-app \
  --compute-platform Server

# Create deployment group
aws deploy create-deployment-group \
  --application-name my-app \
  --deployment-group-name production \
  --deployment-config-name CodeDeployDefault.OneAtATime \
  --ec2-tag-filters Key=Environment,Value=Production,Type=KEY_AND_VALUE \
  --service-role-arn arn:aws:iam::123456789012:role/CodeDeployServiceRole

# Create deployment
aws deploy create-deployment \
  --application-name my-app \
  --deployment-group-name production \
  --s3-location bucket=my-deployments-bucket,key=app-v1.0.zip,bundleType=zip

# Get deployment status
aws deploy get-deployment --deployment-id d-ABCDEF123
```

### AWS CodePipeline

Continuous delivery service.

#### Pipeline Structure

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Source     │───▶│    Build     │───▶│     Test     │───▶│    Deploy    │
│ (CodeCommit) │    │ (CodeBuild)  │    │ (CodeBuild)  │    │ (CodeDeploy) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

```bash
# Create pipeline
aws codepipeline create-pipeline --cli-input-json file://pipeline.json

# Get pipeline details
aws codepipeline get-pipeline --name my-pipeline

# Start pipeline execution
aws codepipeline start-pipeline-execution --name my-pipeline

# Get pipeline state
aws codepipeline get-pipeline-state --name my-pipeline
```

## Machine Learning Services

### Amazon SageMaker

Build, train, and deploy ML models.

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearn

# Set up
role = get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()

# Train model
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.xlarge',
    framework_version='0.23-1',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 5
    }
)

sklearn_estimator.fit({'train': 's3://bucket/train-data'})

# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Make predictions
result = predictor.predict(data)
```

### Amazon Rekognition

Image and video analysis.

```python
import boto3

rekognition = boto3.client('rekognition')

# Detect labels in image
response = rekognition.detect_labels(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'image.jpg'}},
    MaxLabels=10,
    MinConfidence=75
)

for label in response['Labels']:
    print(f"{label['Name']}: {label['Confidence']:.2f}%")

# Detect faces
response = rekognition.detect_faces(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'face.jpg'}},
    Attributes=['ALL']
)

# Compare faces
response = rekognition.compare_faces(
    SourceImage={'S3Object': {'Bucket': 'my-bucket', 'Name': 'source.jpg'}},
    TargetImage={'S3Object': {'Bucket': 'my-bucket', 'Name': 'target.jpg'}},
    SimilarityThreshold=80
)
```

### Amazon Comprehend

Natural language processing.

```python
import boto3

comprehend = boto3.client('comprehend')

text = "Amazon Web Services is a great cloud platform."

# Detect sentiment
sentiment = comprehend.detect_sentiment(Text=text, LanguageCode='en')
print(f"Sentiment: {sentiment['Sentiment']}")

# Detect entities
entities = comprehend.detect_entities(Text=text, LanguageCode='en')
for entity in entities['Entities']:
    print(f"{entity['Text']}: {entity['Type']}")

# Detect key phrases
phrases = comprehend.detect_key_phrases(Text=text, LanguageCode='en')
for phrase in phrases['KeyPhrases']:
    print(phrase['Text'])
```

## Architecture Examples

### Three-Tier Web Application

```
                         Internet
                             │
                    ┌────────▼────────┐
                    │   CloudFront    │  CDN
                    │   (Optional)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Route 53       │  DNS
                    └────────┬────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                          VPC                                 │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Public Subnet (AZ-A)    Public Subnet (AZ-B)        │  │
│  │  ┌─────────────────┐     ┌─────────────────┐         │  │
│  │  │  Application    │     │  Application    │         │  │
│  │  │  Load Balancer  │     │  Load Balancer  │         │  │
│  │  └────────┬────────┘     └────────┬────────┘         │  │
│  └───────────┼──────────────────────┼───────────────────┘  │
│              │                      │                       │
│  ┌───────────▼──────────────────────▼───────────────────┐  │
│  │  Private Subnet (AZ-A)   Private Subnet (AZ-B)       │  │
│  │  ┌─────────────┐          ┌─────────────┐            │  │
│  │  │ Auto Scaling│          │ Auto Scaling│            │  │
│  │  │   Group     │          │   Group     │            │  │
│  │  │  ┌───┐ ┌───┐          │  ┌───┐ ┌───┐            │  │
│  │  │  │EC2│ │EC2│          │  │EC2│ │EC2│            │  │
│  │  │  └─┬─┘ └─┬─┘          │  └─┬─┘ └─┬─┘            │  │
│  │  └────┼─────┼────────────┘────┼─────┼──────────────┘  │
│  │       │     │                 │     │                  │
│  │  ┌────▼─────▼─────────────────▼─────▼──────────────┐  │
│  │  │  Database Subnet (AZ-A)   Database Subnet (AZ-B)│  │
│  │  │  ┌──────────────┐          ┌──────────────┐     │  │
│  │  │  │ RDS Primary  │◄────────▶│ RDS Standby  │     │  │
│  │  │  └──────────────┘          └──────────────┘     │  │
│  │  │                                                  │  │
│  │  │  ┌──────────────┐                               │  │
│  │  │  │ ElastiCache  │                               │  │
│  │  │  └──────────────┘                               │  │
│  │  └──────────────────────────────────────────────────┘  │
│  │                                                          │
│  │  Additional Services:                                   │
│  │  ├─ S3: Static assets                                   │
│  │  ├─ CloudWatch: Monitoring                              │
│  │  ├─ CloudTrail: Audit logs                              │
│  │  └─ WAF: Web application firewall                       │
└──────────────────────────────────────────────────────────────┘
```

### Serverless Microservices

```
                        ┌─────────────┐
                        │   Users     │
                        └──────┬──────┘
                               │
                      ┌────────▼────────┐
                      │  CloudFront +   │
                      │  S3 (Frontend)  │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  API Gateway    │
                      └────────┬────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
   │ Lambda   │          │ Lambda   │          │ Lambda   │
   │ User Svc │          │ Order Svc│          │ Pay Svc  │
   └────┬─────┘          └────┬─────┘          └────┬─────┘
        │                     │                     │
   ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
   │DynamoDB  │          │DynamoDB  │          │DynamoDB  │
   │Users     │          │Orders    │          │Payments  │
   └──────────┘          └──────────┘          └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                       ┌──────▼──────┐
                       │  EventBridge│
                       │     SNS     │
                       └─────────────┘
```

## Cost Optimization

### Cost Optimization Strategies

```
┌──────────────────────────────────────────────────────────┐
│ AWS Cost Optimization Checklist                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ Compute                                                   │
│ ☐ Use Reserved Instances for steady workloads            │
│ ☐ Use Spot Instances for fault-tolerant workloads        │
│ ☐ Right-size instances based on metrics                  │
│ ☐ Use Savings Plans for flexible commitments             │
│ ☐ Stop development/test instances off-hours              │
│ ☐ Use Lambda/Fargate for serverless workloads            │
│ ☐ Enable EC2 Auto Scaling                                │
│                                                           │
│ Storage                                                   │
│ ☐ Use S3 Lifecycle policies                              │
│ ☐ Move infrequent data to S3-IA or Glacier               │
│ ☐ Delete unattached EBS volumes                          │
│ ☐ Delete old snapshots                                   │
│ ☐ Use S3 Intelligent-Tiering                             │
│ ☐ Enable EBS volume encryption only when needed          │
│                                                           │
│ Database                                                  │
│ ☐ Use Aurora Serverless for variable workloads           │
│ ☐ Stop RDS instances when not in use                     │
│ ☐ Use DynamoDB On-Demand for unpredictable traffic       │
│ ☐ Use read replicas efficiently                          │
│ ☐ Right-size RDS instances                               │
│                                                           │
│ Network                                                   │
│ ☐ Use CloudFront to reduce data transfer costs           │
│ ☐ Use VPC endpoints to avoid NAT Gateway costs           │
│ ☐ Consolidate data transfer within same region           │
│ ☐ Use Direct Connect for high volume transfers           │
│                                                           │
│ Monitoring                                                │
│ ☐ Set up AWS Budgets with alerts                         │
│ ☐ Use Cost Explorer to analyze spending                  │
│ ☐ Enable Cost Allocation Tags                            │
│ ☐ Use Trusted Advisor cost optimization checks           │
│ ☐ Review AWS Cost Anomaly Detection                      │
└──────────────────────────────────────────────────────────┘
```

### AWS Cost Management CLI

```bash
# Set up budget
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "Monthly-Budget",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "email@example.com"
    }]
  }]'

# Get cost and usage
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity DAILY \
  --metrics BlendedCost

# Get cost forecast
aws ce get-cost-forecast \
  --time-period Start=2024-02-01,End=2024-02-29 \
  --metric BLENDED_COST \
  --granularity MONTHLY
```

## Best Practices

### Security Best Practices

```
1. Identity and Access
   ├─ Enable MFA for all users
   ├─ Use IAM roles instead of access keys
   ├─ Implement least privilege principle
   ├─ Rotate credentials regularly
   └─ Use AWS SSO for centralized access

2. Network Security
   ├─ Use VPC with public/private subnets
   ├─ Implement security groups properly
   ├─ Use Network ACLs as additional layer
   ├─ Enable VPC Flow Logs
   └─ Use AWS WAF for web applications

3. Data Protection
   ├─ Enable encryption at rest
   ├─ Use TLS/SSL for data in transit
   ├─ Regular backups and snapshots
   ├─ Enable versioning on S3
   └─ Use KMS for key management

4. Monitoring and Logging
   ├─ Enable CloudTrail for all regions
   ├─ Use CloudWatch for monitoring
   ├─ Set up security alerts
   ├─ Regular security audits
   └─ Use AWS Config for compliance

5. Incident Response
   ├─ Have incident response plan
   ├─ Use AWS Systems Manager
   ├─ Enable automated responses
   └─ Regular disaster recovery drills
```

### Performance Best Practices

```
1. Compute
   ├─ Choose appropriate instance types
   ├─ Use Auto Scaling
   ├─ Implement load balancing
   ├─ Consider serverless for variable workloads
   └─ Use placement groups for HPC

2. Storage
   ├─ Use EBS-optimized instances
   ├─ Choose correct EBS volume type
   ├─ Use S3 Transfer Acceleration
   ├─ Implement caching (CloudFront, ElastiCache)
   └─ Use S3 multipart upload

3. Database
   ├─ Use read replicas for read-heavy workloads
   ├─ Enable query caching
   ├─ Use connection pooling
   ├─ Implement proper indexing
   └─ Consider Aurora for better performance

4. Network
   ├─ Use CloudFront CDN
   ├─ Enable enhanced networking
   ├─ Use VPC endpoints
   ├─ Implement Route 53 routing policies
   └─ Consider Direct Connect
```

### Reliability Best Practices

```
1. High Availability
   ├─ Deploy across multiple AZs
   ├─ Use Multi-AZ for databases
   ├─ Implement auto-scaling
   ├─ Use Elastic Load Balancing
   └─ Consider multi-region for critical workloads

2. Backup and Recovery
   ├─ Automated backups for RDS
   ├─ Regular EBS snapshots
   ├─ Enable S3 versioning
   ├─ Cross-region replication
   └─ Test recovery procedures

3. Monitoring
   ├─ Set up CloudWatch alarms
   ├─ Use health checks
   ├─ Monitor key metrics
   ├─ Implement automated responses
   └─ Use AWS X-Ray for tracing

4. Testing
   ├─ Regular load testing
   ├─ Chaos engineering
   ├─ Failover testing
   └─ Disaster recovery drills
```

## CLI Reference

### Common CLI Patterns

```bash
# Use --query for filtering output
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
  --output table

# Use --filters for filtering resources
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
           "Name=tag:Environment,Values=production"

# Use --output for different formats
aws ec2 describe-instances --output json
aws ec2 describe-instances --output yaml
aws ec2 describe-instances --output table
aws ec2 describe-instances --output text

# Use JMESPath for complex queries
aws ec2 describe-instances \
  --query 'Reservations[].Instances[?State.Name==`running`].[InstanceId,PrivateIpAddress]'

# Paginate results
aws s3api list-objects-v2 \
  --bucket my-bucket \
  --max-items 100 \
  --page-size 10

# Wait for resource to be ready
aws ec2 wait instance-running --instance-ids i-1234567890abcdef0

# Generate skeleton for complex commands
aws ec2 run-instances --generate-cli-skeleton > template.json
# Edit template.json
aws ec2 run-instances --cli-input-json file://template.json
```

### Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

alias ec2-list='aws ec2 describe-instances --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress,Tags[?Key=='\''Name'\''].Value|[0]]" --output table'

alias ec2-running='aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].[InstanceId,InstanceType,PublicIpAddress]" --output table'

alias s3-buckets='aws s3 ls'

alias lambda-list='aws lambda list-functions --query "Functions[].[FunctionName,Runtime,LastModified]" --output table'

alias rds-list='aws rds describe-db-instances --query "DBInstances[].[DBInstanceIdentifier,DBInstanceStatus,Engine,DBInstanceClass]" --output table'
```

## Certification Paths

### AWS Certification Roadmap

```
Foundational
    │
    └─ AWS Certified Cloud Practitioner
        │
        ├─ Associate Level
        │   ├─ Solutions Architect Associate
        │   ├─ Developer Associate
        │   └─ SysOps Administrator Associate
        │
        └─ Professional Level
            ├─ Solutions Architect Professional
            └─ DevOps Engineer Professional
            
        Specialty (Optional)
        ├─ Security Specialty
        ├─ Machine Learning Specialty
        ├─ Advanced Networking Specialty
        ├─ Database Specialty
        └─ Data Analytics Specialty
```

## Resources

### Official Documentation
- AWS Documentation: https://docs.aws.amazon.com
- AWS CLI Reference: https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html
- AWS SDK Documentation: https://aws.amazon.com/tools/

### Learning Resources
- AWS Training and Certification: https://aws.amazon.com/training/
- AWS Free Tier: https://aws.amazon.com/free/
- AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/
- AWS Samples: https://github.com/aws-samples
- AWS Workshops: https://workshops.aws/

### Community
- r/aws: Reddit community
- AWS Forums: https://forums.aws.amazon.com/
- AWS re:Post: https://repost.aws/
- AWS User Groups: https://aws.amazon.com/developer/community/usergroups/

### Tools
- AWS CLI: Command-line interface
- AWS SDKs: Python (Boto3), JavaScript, Java, .NET, etc.
- AWS CDK: Infrastructure as code using programming languages
- Terraform: Multi-cloud infrastructure as code
- LocalStack: Local AWS cloud emulator

---

**Updated**: January 2025
