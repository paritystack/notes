# Microsoft Azure

## Table of Contents
- [Introduction](#introduction)
- [Azure Global Infrastructure](#azure-global-infrastructure)
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
- [AI and Machine Learning](#ai-and-machine-learning)
- [Architecture Examples](#architecture-examples)
- [Azure vs AWS Comparison](#azure-vs-aws-comparison)
- [Cost Optimization](#cost-optimization)
- [Best Practices](#best-practices)
- [CLI Reference](#cli-reference)

## Introduction

Microsoft Azure is a cloud computing platform providing 200+ services for building, deploying, and managing applications through Microsoft's global network of data centers.

### Key Advantages
- **Enterprise Integration**: Seamless integration with Microsoft products (Office 365, Active Directory, Dynamics)
- **Hybrid Cloud**: Industry-leading hybrid cloud capabilities with Azure Arc
- **Global Reach**: 60+ regions (more than any other cloud provider)
- **Compliance**: Most comprehensive compliance offerings
- **Windows Workloads**: Best platform for .NET and Windows-based applications
- **Developer Tools**: Excellent integration with Visual Studio and GitHub

### Azure Account Hierarchy

```
┌─────────────────────────────────────────────────┐
│        Azure Entra ID (Azure AD) Tenant          │
│        (Organization-wide identity)              │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Management Groups │
         └─────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │   Subscriptions    │
         │  ├─ Production     │
         │  ├─ Development    │
         │  └─ Testing        │
         └─────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │  Resource Groups   │
         │  ├─ RG-Web         │
         │  ├─ RG-Database    │
         │  └─ RG-Network     │
         └─────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │     Resources      │
         │  ├─ VMs            │
         │  ├─ Storage        │
         │  └─ Databases      │
         └────────────────────┘
```

## Azure Global Infrastructure

### Hierarchy

```
Geography (e.g., United States)
  └─ Region (e.g., East US, West US)
      └─ Availability Zones (3 per region)
          └─ Data Centers
              └─ Edge Locations (Azure Front Door)
```

### Azure Regions

**Azure has 60+ regions worldwide** - more than any other cloud provider

**Paired Regions**: Each region is paired with another region for disaster recovery
- Example: East US ↔ West US
- Example: North Europe ↔ West Europe

### Availability Zones

- 3 or more physically separate zones within a region
- Each zone has independent power, cooling, networking
- < 2ms latency between zones
- Not all regions have Availability Zones

### Region Selection Criteria

```
Factor              Consideration
────────────────────────────────────────────────
Latency             Distance to users
Compliance          Data residency requirements
Services            Service availability varies
Cost                Pricing differs by region
Paired Region       Consider DR requirements
```

## Getting Started

### Azure CLI Installation

```bash
# Install Azure CLI (Linux)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Azure CLI (macOS)
brew update && brew install azure-cli

# Install Azure CLI (Windows - PowerShell)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# Verify installation
az --version

# Login to Azure
az login

# Login with specific tenant
az login --tenant TENANT_ID

# Login with service principal
az login --service-principal \
  --username APP_ID \
  --password PASSWORD \
  --tenant TENANT_ID

# Set default subscription
az account set --subscription "My Subscription"

# List subscriptions
az account list --output table

# Show current subscription
az account show
```

### Azure PowerShell

```powershell
# Install Azure PowerShell
Install-Module -Name Az -Repository PSGallery -Force

# Connect to Azure
Connect-AzAccount

# Set subscription
Set-AzContext -SubscriptionId "subscription-id"

# List subscriptions
Get-AzSubscription

# List resource groups
Get-AzResourceGroup
```

### Basic Azure CLI Commands

```bash
# Get help
az help
az vm help

# List all resource groups
az group list --output table

# List all resources
az resource list --output table

# List available locations
az account list-locations --output table

# List available VM sizes
az vm list-sizes --location eastus --output table

# Interactive mode
az interactive
```

## Core Compute Services

### Azure Virtual Machines

Cloud-based virtual servers.

#### VM Series and Sizes

```
Series      vCPU    Memory    Use Case                      AWS Equivalent
────────────────────────────────────────────────────────────────────────────
B-Series    1-20    0.5-80GB  Burstable, dev/test          t3
D-Series    2-96    8-384GB   General purpose              m5
F-Series    2-72    4-144GB   Compute optimized            c5
E-Series    2-96    16-672GB  Memory optimized             r5
M-Series    128-416 2-12TB    Largest memory               x1e
N-Series    6-24    112-448GB GPU instances                p3/g4
```

#### VM Pricing Models

```
Model                   Discount    Commitment    Use Case
───────────────────────────────────────────────────────────────
Pay-as-you-go          Baseline    None          Short-term
Reserved Instances     Up to 72%   1-3 years     Steady state
Spot VMs               Up to 90%   None          Fault-tolerant
Azure Hybrid Benefit   Up to 85%   None          Existing licenses
```

#### VM CLI Examples

```bash
# Create resource group
az group create \
  --name myResourceGroup \
  --location eastus

# List available VM images
az vm image list --output table
az vm image list --publisher MicrosoftWindowsServer --output table

# Create Linux VM
az vm create \
  --resource-group myResourceGroup \
  --name myVM \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard \
  --tags Environment=Production Owner=IT

# Create Windows VM
az vm create \
  --resource-group myResourceGroup \
  --name myWindowsVM \
  --image Win2022Datacenter \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --admin-password 'SecurePassword123!'

# List VMs
az vm list --output table

# Get VM details
az vm show \
  --resource-group myResourceGroup \
  --name myVM \
  --show-details

# Start VM
az vm start \
  --resource-group myResourceGroup \
  --name myVM

# Stop VM (deallocate to stop billing)
az vm deallocate \
  --resource-group myResourceGroup \
  --name myVM

# Restart VM
az vm restart \
  --resource-group myResourceGroup \
  --name myVM

# Resize VM
az vm resize \
  --resource-group myResourceGroup \
  --name myVM \
  --size Standard_D4s_v3

# Delete VM
az vm delete \
  --resource-group myResourceGroup \
  --name myVM \
  --yes

# Open port
az vm open-port \
  --resource-group myResourceGroup \
  --name myVM \
  --port 80 \
  --priority 1001

# Run command on VM
az vm run-command invoke \
  --resource-group myResourceGroup \
  --name myVM \
  --command-id RunShellScript \
  --scripts "sudo apt-get update && sudo apt-get install -y nginx"

# Create VM from snapshot
az vm create \
  --resource-group myResourceGroup \
  --name myRestoredVM \
  --attach-os-disk myOSDisk \
  --os-type Linux

# Get VM instance metadata (from within VM)
curl -H Metadata:true "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
```

#### Custom Script Extension

```bash
# Add custom script extension (Linux)
az vm extension set \
  --resource-group myResourceGroup \
  --vm-name myVM \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings '{"fileUris": ["https://raw.githubusercontent.com/user/repo/script.sh"],"commandToExecute": "./script.sh"}'

# Add custom script extension (Windows)
az vm extension set \
  --resource-group myResourceGroup \
  --vm-name myWindowsVM \
  --name CustomScriptExtension \
  --publisher Microsoft.Compute \
  --settings '{"fileUris": ["https://example.com/script.ps1"],"commandToExecute": "powershell -ExecutionPolicy Unrestricted -File script.ps1"}'
```

### Azure Virtual Machine Scale Sets (VMSS)

Auto-scaling groups of identical VMs.

#### VMSS Architecture

```
┌─────────────────────────────────────────────────┐
│          Azure Load Balancer                     │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
    ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
    │ VM 1  │  │ VM 2  │  │ VM 3  │
    └───────┘  └───────┘  └───────┘
        │          │          │
        └──────────┼──────────┘
                   │
        ┌──────────▼──────────┐
        │ Virtual Machine     │
        │ Scale Set (VMSS)    │
        │                     │
        │ Min: 2              │
        │ Current: 3          │
        │ Max: 10             │
        │                     │
        │ Scale Rules:        │
        │ CPU > 75%: +1 VM    │
        │ CPU < 25%: -1 VM    │
        └─────────────────────┘
```

#### VMSS CLI Examples

```bash
# Create VMSS
az vmss create \
  --resource-group myResourceGroup \
  --name myScaleSet \
  --image Ubuntu2204 \
  --instance-count 3 \
  --vm-sku Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys \
  --load-balancer myLoadBalancer \
  --upgrade-policy-mode Automatic

# List VMSS
az vmss list --output table

# Scale manually
az vmss scale \
  --resource-group myResourceGroup \
  --name myScaleSet \
  --new-capacity 5

# Create autoscale profile
az monitor autoscale create \
  --resource-group myResourceGroup \
  --resource myScaleSet \
  --resource-type Microsoft.Compute/virtualMachineScaleSets \
  --name myAutoscaleProfile \
  --min-count 2 \
  --max-count 10 \
  --count 3

# Create autoscale rule (scale out)
az monitor autoscale rule create \
  --resource-group myResourceGroup \
  --autoscale-name myAutoscaleProfile \
  --condition "Percentage CPU > 75 avg 5m" \
  --scale out 1

# Create autoscale rule (scale in)
az monitor autoscale rule create \
  --resource-group myResourceGroup \
  --autoscale-name myAutoscaleProfile \
  --condition "Percentage CPU < 25 avg 5m" \
  --scale in 1

# List VMSS instances
az vmss list-instances \
  --resource-group myResourceGroup \
  --name myScaleSet \
  --output table

# Update VMSS image
az vmss update \
  --resource-group myResourceGroup \
  --name myScaleSet \
  --set virtualMachineProfile.storageProfile.imageReference.version=latest

# Start rolling upgrade
az vmss update-instances \
  --resource-group myResourceGroup \
  --name myScaleSet \
  --instance-ids '*'

# Delete VMSS
az vmss delete \
  --resource-group myResourceGroup \
  --name myScaleSet
```

### Azure App Service

PaaS for web applications.

```bash
# Create App Service Plan
az appservice plan create \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --sku B1 \
  --is-linux

# Create Web App
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name myUniqueWebApp123 \
  --runtime "NODE:18-lts"

# Deploy from GitHub
az webapp deployment source config \
  --name myUniqueWebApp123 \
  --resource-group myResourceGroup \
  --repo-url https://github.com/user/repo \
  --branch main \
  --manual-integration

# Deploy from local Git
az webapp deployment source config-local-git \
  --name myUniqueWebApp123 \
  --resource-group myResourceGroup

# Deploy ZIP file
az webapp deployment source config-zip \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123 \
  --src app.zip

# Set environment variables
az webapp config appsettings set \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123 \
  --settings DB_HOST=mydb.database.windows.net DB_NAME=mydb

# Enable HTTPS only
az webapp update \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123 \
  --https-only true

# Scale up (change plan)
az appservice plan update \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --sku P1V2

# Scale out (add instances)
az appservice plan update \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --number-of-workers 3

# View logs
az webapp log tail \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123

# Restart web app
az webapp restart \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123

# Delete web app
az webapp delete \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123
```

## Storage Services

### Azure Blob Storage

Object storage service (equivalent to AWS S3).

#### Blob Storage Types

```
Type            Use Case                        Performance    Cost
────────────────────────────────────────────────────────────────────
Block Blobs     Text and binary data           Standard/Premium  $$
Append Blobs    Logging data                   Standard         $$
Page Blobs      VHD files, random access       Premium          $$$
```

#### Blob Access Tiers

```
Tier        Access Frequency    Retrieval Time    Cost
─────────────────────────────────────────────────────────
Hot         Frequent            Immediate         $$$
Cool        Infrequent (30d+)   Immediate         $$
Cold        Rare (90d+)         Immediate         $
Archive     Rarely (180d+)      Hours             ¢
```

#### Blob Storage Architecture

```
┌─────────────────────────────────────────────────┐
│  Storage Account: mystorageaccount              │
│  Location: eastus                                │
│  Replication: LRS/GRS/RA-GRS                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  Container: images (Blob Container)             │
│    ├─ logo.png                                  │
│    ├─ banner.jpg                                │
│    └─ photos/                                   │
│        ├─ photo1.jpg                            │
│        └─ photo2.jpg                            │
│                                                  │
│  Container: documents                           │
│    ├─ report.pdf                                │
│    └─ invoice.xlsx                              │
│                                                  │
│  Container: backups                             │
│    └─ database-backup.sql                      │
│                                                  │
│  File Share: fileshare (Azure Files)           │
│    ├─ shared/                                   │
│    └─ config/                                   │
│                                                  │
│  Table Storage (NoSQL)                          │
│  Queue Storage (Message Queue)                  │
└─────────────────────────────────────────────────┘
```

#### Blob Storage CLI Examples

```bash
# Create storage account
az storage account create \
  --name mystorageaccount123 \
  --resource-group myResourceGroup \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

# Get connection string
az storage account show-connection-string \
  --name mystorageaccount123 \
  --resource-group myResourceGroup

# Export connection string
export AZURE_STORAGE_CONNECTION_STRING="<connection-string>"

# Create container
az storage container create \
  --name mycontainer \
  --account-name mystorageaccount123 \
  --public-access off

# Upload blob
az storage blob upload \
  --container-name mycontainer \
  --name myfile.txt \
  --file ./local-file.txt \
  --account-name mystorageaccount123

# Upload directory
az storage blob upload-batch \
  --destination mycontainer \
  --source ./local-directory \
  --account-name mystorageaccount123

# Download blob
az storage blob download \
  --container-name mycontainer \
  --name myfile.txt \
  --file ./downloaded-file.txt \
  --account-name mystorageaccount123

# List blobs
az storage blob list \
  --container-name mycontainer \
  --account-name mystorageaccount123 \
  --output table

# Copy blob
az storage blob copy start \
  --source-container mycontainer \
  --source-blob myfile.txt \
  --destination-container backup \
  --destination-blob myfile-backup.txt \
  --account-name mystorageaccount123

# Generate SAS token
az storage blob generate-sas \
  --container-name mycontainer \
  --name myfile.txt \
  --account-name mystorageaccount123 \
  --permissions r \
  --expiry 2024-12-31T23:59:59Z

# Set blob tier
az storage blob set-tier \
  --container-name mycontainer \
  --name myfile.txt \
  --tier Cool \
  --account-name mystorageaccount123

# Delete blob
az storage blob delete \
  --container-name mycontainer \
  --name myfile.txt \
  --account-name mystorageaccount123

# Enable versioning
az storage account blob-service-properties update \
  --account-name mystorageaccount123 \
  --resource-group myResourceGroup \
  --enable-versioning true

# Set lifecycle management policy
az storage account management-policy create \
  --account-name mystorageaccount123 \
  --resource-group myResourceGroup \
  --policy @policy.json
```

#### Lifecycle Management Policy Example

```json
{
  "rules": [
    {
      "enabled": true,
      "name": "MoveToArchive",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 365
            }
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["logs/"]
        }
      }
    }
  ]
}
```

#### Blob Storage SDK Example (Python)

```python
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta

# Create blob service client
connection_string = "DefaultEndpointsProtocol=https;AccountName=..."
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create container
def create_container(container_name):
    container_client = blob_service_client.create_container(container_name)
    return container_client

# Upload blob
def upload_blob(container_name, blob_name, data):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {blob_name}")

# Upload file
def upload_file(container_name, file_path, blob_name=None):
    if blob_name is None:
        blob_name = file_path.split('/')[-1]
    
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file_path} as {blob_name}")

# Download blob
def download_blob(container_name, blob_name, file_path):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    
    with open(file_path, "wb") as file:
        data = blob_client.download_blob()
        file.write(data.readall())
    print(f"Downloaded {blob_name} to {file_path}")

# List blobs
def list_blobs(container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    
    for blob in blob_list:
        print(f"{blob.name}: {blob.size} bytes")

# Generate SAS URL
def generate_sas_url(container_name, blob_name, account_name, account_key):
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    
    url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return url

# Delete blob
def delete_blob(container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    blob_client.delete_blob()
    print(f"Deleted {blob_name}")

# Usage
create_container("mycontainer")
upload_file("mycontainer", "./local-file.txt")
download_blob("mycontainer", "local-file.txt", "./downloaded.txt")
list_blobs("mycontainer")
```

### Azure Files

Managed SMB/NFS file shares.

```bash
# Create file share
az storage share create \
  --name myfileshare \
  --account-name mystorageaccount123 \
  --quota 100

# Upload file to share
az storage file upload \
  --share-name myfileshare \
  --source ./local-file.txt \
  --account-name mystorageaccount123

# List files
az storage file list \
  --share-name myfileshare \
  --account-name mystorageaccount123 \
  --output table

# Mount file share (Linux)
sudo mkdir /mnt/azure
sudo mount -t cifs //mystorageaccount123.file.core.windows.net/myfileshare /mnt/azure \
  -o vers=3.0,username=mystorageaccount123,password=<storage-key>,dir_mode=0777,file_mode=0777

# Mount file share (Windows)
net use Z: \\mystorageaccount123.file.core.windows.net\myfileshare /user:Azure\mystorageaccount123 <storage-key>

# Add to /etc/fstab (Linux)
echo "//mystorageaccount123.file.core.windows.net/myfileshare /mnt/azure cifs vers=3.0,username=mystorageaccount123,password=<storage-key>,dir_mode=0777,file_mode=0777 0 0" | sudo tee -a /etc/fstab
```

### Azure Disk Storage

Managed disks for VMs (equivalent to AWS EBS).

#### Disk Types

```
Type            IOPS        Throughput      Use Case              Cost
─────────────────────────────────────────────────────────────────────
Ultra Disk      160K+       4,000 MB/s      Mission-critical      $$$$
Premium SSD v2  80K         1,200 MB/s      Production DBs        $$$
Premium SSD     20K         900 MB/s        Production            $$
Standard SSD    6K          750 MB/s        Web servers           $
Standard HDD    2K          500 MB/s        Backup, dev/test      ¢
```

```bash
# Create managed disk
az disk create \
  --resource-group myResourceGroup \
  --name myDataDisk \
  --size-gb 128 \
  --sku Premium_LRS

# Attach disk to VM
az vm disk attach \
  --resource-group myResourceGroup \
  --vm-name myVM \
  --name myDataDisk

# Detach disk
az vm disk detach \
  --resource-group myResourceGroup \
  --vm-name myVM \
  --name myDataDisk

# Create snapshot
az snapshot create \
  --resource-group myResourceGroup \
  --name mySnapshot \
  --source myDataDisk

# Create disk from snapshot
az disk create \
  --resource-group myResourceGroup \
  --name myRestoredDisk \
  --source mySnapshot

# Increase disk size
az disk update \
  --resource-group myResourceGroup \
  --name myDataDisk \
  --size-gb 256
```

## Database Services

### Azure SQL Database

Managed SQL Server database.

#### Service Tiers

```
Tier          vCores    Memory    Max DB Size    Use Case          Cost
────────────────────────────────────────────────────────────────────────
Serverless    0.5-40    3-120GB   4TB           Variable load     $$
General       2-80      10.4-408GB 4TB           Balanced          $$
Purpose
Business      2-128     20.8-625GB 4TB           Mission-critical  $$$$
Critical
Hyperscale    2-128     20.8-625GB 100TB         Large databases   $$$
```

#### SQL Database CLI Examples

```bash
# Create SQL Server
az sql server create \
  --name myuniquesqlserver123 \
  --resource-group myResourceGroup \
  --location eastus \
  --admin-user sqladmin \
  --admin-password 'SecurePassword123!'

# Configure firewall rule
az sql server firewall-rule create \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name AllowMyIP \
  --start-ip-address 1.2.3.4 \
  --end-ip-address 1.2.3.4

# Allow Azure services
az sql server firewall-rule create \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Create database
az sql db create \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myDatabase \
  --service-objective S0 \
  --backup-storage-redundancy Local

# Create serverless database
az sql db create \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myServerlessDB \
  --edition GeneralPurpose \
  --compute-model Serverless \
  --family Gen5 \
  --capacity 2 \
  --auto-pause-delay 60

# List databases
az sql db list \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --output table

# Scale database
az sql db update \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myDatabase \
  --service-objective S2

# Create read replica
az sql db replica create \
  --name myDatabase \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --partner-server myuniquesqlserver-replica \
  --partner-resource-group myResourceGroup

# Create backup
az sql db export \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myDatabase \
  --admin-user sqladmin \
  --admin-password 'SecurePassword123!' \
  --storage-key-type StorageAccessKey \
  --storage-key "<storage-key>" \
  --storage-uri "https://mystorageaccount.blob.core.windows.net/backups/mydb.bacpac"

# Restore database
az sql db restore \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myRestoredDB \
  --source-database myDatabase \
  --time "2024-01-01T00:00:00Z"

# Delete database
az sql db delete \
  --resource-group myResourceGroup \
  --server myuniquesqlserver123 \
  --name myDatabase \
  --yes

# Connect to database
sqlcmd -S myuniquesqlserver123.database.windows.net -d myDatabase -U sqladmin -P 'SecurePassword123!'
```

#### SQL Database Connection Example (Python)

```python
import pyodbc

# Connection string
server = 'myuniquesqlserver123.database.windows.net'
database = 'myDatabase'
username = 'sqladmin'
password = 'SecurePassword123!'
driver = '{ODBC Driver 18 for SQL Server}'

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Connect to database
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE users (
        id INT PRIMARY KEY IDENTITY,
        name NVARCHAR(100),
        email NVARCHAR(100),
        created_at DATETIME DEFAULT GETDATE()
    )
''')

# Insert data
cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ('Alice', 'alice@example.com'))
conn.commit()

# Query data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(f"ID: {row.id}, Name: {row.name}, Email: {row.email}")

# Close connection
cursor.close()
conn.close()
```

### Azure Cosmos DB

Globally distributed NoSQL database.

#### Cosmos DB APIs

```
API              Type           Use Case                AWS Equivalent
──────────────────────────────────────────────────────────────────────
Core (SQL)       Document       General purpose         DynamoDB
MongoDB          Document       MongoDB compatibility   DocumentDB
Cassandra        Wide-column    Cassandra workloads     Keyspaces
Gremlin          Graph          Graph relationships     Neptune
Table            Key-value      Simple key-value        DynamoDB
```

#### Cosmos DB CLI Examples

```bash
# Create Cosmos DB account
az cosmosdb create \
  --name mycosmosaccount123 \
  --resource-group myResourceGroup \
  --locations regionName=eastus failoverPriority=0 \
  --locations regionName=westus failoverPriority=1 \
  --default-consistency-level Session \
  --enable-automatic-failover true

# Create database (SQL API)
az cosmosdb sql database create \
  --account-name mycosmosaccount123 \
  --resource-group myResourceGroup \
  --name myDatabase

# Create container
az cosmosdb sql container create \
  --account-name mycosmosaccount123 \
  --resource-group myResourceGroup \
  --database-name myDatabase \
  --name myContainer \
  --partition-key-path "/userId" \
  --throughput 400

# Get connection string
az cosmosdb keys list \
  --name mycosmosaccount123 \
  --resource-group myResourceGroup \
  --type connection-strings

# List databases
az cosmosdb sql database list \
  --account-name mycosmosaccount123 \
  --resource-group myResourceGroup

# Update throughput
az cosmosdb sql container throughput update \
  --account-name mycosmosaccount123 \
  --resource-group myResourceGroup \
  --database-name myDatabase \
  --name myContainer \
  --throughput 1000
```

#### Cosmos DB SDK Example (Python)

```python
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Initialize client
endpoint = "https://mycosmosaccount123.documents.azure.com:443/"
key = "<primary-key>"
client = CosmosClient(endpoint, key)

# Get database and container
database = client.get_database_client("myDatabase")
container = database.get_container_client("myContainer")

# Create item
item = {
    'id': 'user-001',
    'userId': 'user-001',
    'name': 'Alice',
    'email': 'alice@example.com',
    'age': 30
}
container.create_item(body=item)

# Read item
item = container.read_item(item='user-001', partition_key='user-001')
print(item)

# Query items
query = "SELECT * FROM c WHERE c.age > @age"
parameters = [{"name": "@age", "value": 25}]

items = list(container.query_items(
    query=query,
    parameters=parameters,
    enable_cross_partition_query=True
))

for item in items:
    print(f"{item['name']}: {item['age']} years old")

# Update item
item['age'] = 31
container.replace_item(item='user-001', body=item)

# Delete item
container.delete_item(item='user-001', partition_key='user-001')
```

### Azure Database for PostgreSQL/MySQL

Managed open-source databases.

```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --name mypostgresserver123 \
  --resource-group myResourceGroup \
  --location eastus \
  --admin-user myadmin \
  --admin-password 'SecurePassword123!' \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32

# Create MySQL server
az mysql flexible-server create \
  --name mymysqlserver123 \
  --resource-group myResourceGroup \
  --location eastus \
  --admin-user myadmin \
  --admin-password 'SecurePassword123!' \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32

# Configure firewall
az postgres flexible-server firewall-rule create \
  --resource-group myResourceGroup \
  --name mypostgresserver123 \
  --rule-name AllowMyIP \
  --start-ip-address 1.2.3.4 \
  --end-ip-address 1.2.3.4

# Connect to PostgreSQL
psql "host=mypostgresserver123.postgres.database.azure.com port=5432 dbname=postgres user=myadmin password=SecurePassword123! sslmode=require"

# Connect to MySQL
mysql -h mymysqlserver123.mysql.database.azure.com -u myadmin -p
```

### Azure Cache for Redis

Managed Redis cache.

```bash
# Create Redis cache
az redis create \
  --resource-group myResourceGroup \
  --name myrediscache123 \
  --location eastus \
  --sku Basic \
  --vm-size c0

# Get access keys
az redis list-keys \
  --resource-group myResourceGroup \
  --name myrediscache123

# Get hostname
az redis show \
  --resource-group myResourceGroup \
  --name myrediscache123 \
  --query hostName

# Connect to Redis
redis-cli -h myrediscache123.redis.cache.windows.net -p 6380 -a <primary-key> --tls
```

## Networking Services

### Azure Virtual Network (VNet)

Isolated network (equivalent to AWS VPC).

#### VNet Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  VNet: my-vnet (10.0.0.0/16)                                │
│  Region: eastus                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │ Public Subnet         │  │ Public Subnet            │   │
│  │ 10.0.1.0/24          │  │ 10.0.2.0/24              │   │
│  │ (AZ 1)               │  │ (AZ 2)                   │   │
│  │                       │  │                          │   │
│  │ ┌─────────────────┐  │  │ ┌─────────────────┐     │   │
│  │ │ Load Balancer   │  │  │ │ Load Balancer   │     │   │
│  │ └─────────────────┘  │  │ └─────────────────┘     │   │
│  └───────────┬───────────┘  └──────────┬───────────────┘   │
│              │                         │                    │
│              │   Azure Gateway         │                    │
│              └──────────┬──────────────┘                    │
│                         │                                   │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │ Private Subnet        │  │ Private Subnet           │   │
│  │ 10.0.11.0/24         │  │ 10.0.12.0/24             │   │
│  │ (AZ 1)               │  │ (AZ 2)                   │   │
│  │                       │  │                          │   │
│  │ ┌─────┐ ┌─────┐      │  │ ┌─────┐ ┌─────┐         │   │
│  │ │ VM  │ │ VM  │      │  │ │ VM  │ │ VM  │         │   │
│  │ └─────┘ └─────┘      │  │ └─────┘ └─────┘         │   │
│  └───────────────────────┘  └──────────────────────────┘   │
│                                                              │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │ Database Subnet       │  │ Database Subnet          │   │
│  │ 10.0.21.0/24         │  │ 10.0.22.0/24             │   │
│  │                       │  │                          │   │
│  │ ┌──────────┐          │  │ ┌──────────┐            │   │
│  │ │  SQL DB  │          │  │ │  SQL DB  │            │   │
│  │ └──────────┘          │  │ └──────────┘            │   │
│  └───────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### VNet CLI Examples

```bash
# Create VNet
az network vnet create \
  --resource-group myResourceGroup \
  --name myVNet \
  --address-prefix 10.0.0.0/16 \
  --location eastus

# Create subnet
az network vnet subnet create \
  --resource-group myResourceGroup \
  --vnet-name myVNet \
  --name PublicSubnet \
  --address-prefixes 10.0.1.0/24

az network vnet subnet create \
  --resource-group myResourceGroup \
  --vnet-name myVNet \
  --name PrivateSubnet \
  --address-prefixes 10.0.11.0/24

# List VNets
az network vnet list --output table

# List subnets
az network vnet subnet list \
  --resource-group myResourceGroup \
  --vnet-name myVNet \
  --output table

# Create Network Security Group (NSG)
az network nsg create \
  --resource-group myResourceGroup \
  --name myNSG

# Add NSG rule
az network nsg rule create \
  --resource-group myResourceGroup \
  --nsg-name myNSG \
  --name AllowHTTP \
  --priority 100 \
  --source-address-prefixes '*' \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 80 \
  --access Allow \
  --protocol Tcp \
  --direction Inbound

az network nsg rule create \
  --resource-group myResourceGroup \
  --nsg-name myNSG \
  --name AllowSSH \
  --priority 110 \
  --source-address-prefixes 'VirtualNetwork' \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 22 \
  --access Allow \
  --protocol Tcp \
  --direction Inbound

# Associate NSG with subnet
az network vnet subnet update \
  --resource-group myResourceGroup \
  --vnet-name myVNet \
  --name PublicSubnet \
  --network-security-group myNSG

# Create NAT Gateway
az network public-ip create \
  --resource-group myResourceGroup \
  --name myNATGatewayIP \
  --sku Standard \
  --allocation-method Static

az network nat gateway create \
  --resource-group myResourceGroup \
  --name myNATGateway \
  --public-ip-addresses myNATGatewayIP \
  --idle-timeout 10

# Associate NAT Gateway with subnet
az network vnet subnet update \
  --resource-group myResourceGroup \
  --vnet-name myVNet \
  --name PrivateSubnet \
  --nat-gateway myNATGateway

# VNet peering
az network vnet peering create \
  --resource-group myResourceGroup \
  --name myVNet-to-VNet2 \
  --vnet-name myVNet \
  --remote-vnet myVNet2 \
  --allow-vnet-access
```

### Azure Load Balancer

Distribute traffic across resources.

#### Load Balancer Types

```
Type                SKU         OSI Layer    Use Case              Cost
───────────────────────────────────────────────────────────────────────
Load Balancer       Basic       Layer 4      Internal/Public       Free
Load Balancer       Standard    Layer 4      Production            $$
Application         Standard    Layer 7      HTTP/HTTPS routing    $$
Gateway
```

```bash
# Create public IP
az network public-ip create \
  --resource-group myResourceGroup \
  --name myPublicIP \
  --sku Standard

# Create load balancer
az network lb create \
  --resource-group myResourceGroup \
  --name myLoadBalancer \
  --sku Standard \
  --public-ip-address myPublicIP \
  --frontend-ip-name myFrontEnd \
  --backend-pool-name myBackEndPool

# Create health probe
az network lb probe create \
  --resource-group myResourceGroup \
  --lb-name myLoadBalancer \
  --name myHealthProbe \
  --protocol tcp \
  --port 80 \
  --interval 15 \
  --threshold 2

# Create load balancer rule
az network lb rule create \
  --resource-group myResourceGroup \
  --lb-name myLoadBalancer \
  --name myHTTPRule \
  --protocol tcp \
  --frontend-port 80 \
  --backend-port 80 \
  --frontend-ip-name myFrontEnd \
  --backend-pool-name myBackEndPool \
  --probe-name myHealthProbe

# Add VM to backend pool
az network nic ip-config address-pool add \
  --resource-group myResourceGroup \
  --nic-name myNIC \
  --ip-config-name ipconfig1 \
  --lb-name myLoadBalancer \
  --address-pool myBackEndPool
```

### Azure Application Gateway

Layer 7 load balancer with WAF.

```bash
# Create Application Gateway
az network application-gateway create \
  --name myAppGateway \
  --resource-group myResourceGroup \
  --location eastus \
  --vnet-name myVNet \
  --subnet PublicSubnet \
  --capacity 2 \
  --sku Standard_v2 \
  --public-ip-address myPublicIP \
  --servers 10.0.11.4 10.0.11.5

# Create path-based routing rule
az network application-gateway url-path-map create \
  --gateway-name myAppGateway \
  --resource-group myResourceGroup \
  --name myPathMap \
  --paths /images/* \
  --http-settings appGatewayBackendHttpSettings \
  --address-pool imagesBackendPool

# Enable Web Application Firewall (WAF)
az network application-gateway waf-config set \
  --gateway-name myAppGateway \
  --resource-group myResourceGroup \
  --enabled true \
  --firewall-mode Prevention \
  --rule-set-version 3.0
```

### Azure DNS

DNS hosting service.

```bash
# Create DNS zone
az network dns zone create \
  --resource-group myResourceGroup \
  --name example.com

# Create A record
az network dns record-set a add-record \
  --resource-group myResourceGroup \
  --zone-name example.com \
  --record-set-name www \
  --ipv4-address 1.2.3.4

# Create CNAME record
az network dns record-set cname set-record \
  --resource-group myResourceGroup \
  --zone-name example.com \
  --record-set-name blog \
  --cname www.example.com

# List records
az network dns record-set list \
  --resource-group myResourceGroup \
  --zone-name example.com

# Get nameservers
az network dns zone show \
  --resource-group myResourceGroup \
  --name example.com \
  --query nameServers
```

## Serverless Services

### Azure Functions

Serverless compute (equivalent to AWS Lambda).

#### Function Runtime Versions

```
Runtime       Languages                    Timeout (Consumption)
───────────────────────────────────────────────────────────────
4.x (Current) C#, Java, JavaScript,       10 minutes (default)
              Python, PowerShell, TypeScript
```

#### Function Triggers

```
Trigger Type        Use Case
────────────────────────────────────────────────────
HTTP                REST APIs, webhooks
Timer               Scheduled tasks
Blob Storage        File processing
Queue Storage       Async processing
Event Grid          Event-driven workflows
Event Hub           Real-time data streams
Service Bus         Enterprise messaging
Cosmos DB           Database change feed
```

#### Function Example (Python)

```python
import logging
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
            name = req_body.get('name')
        except ValueError:
            pass
    
    if name:
        return func.HttpResponse(
            f"Hello, {name}!",
            status_code=200
        )
    else:
        return func.HttpResponse(
            "Please pass a name parameter",
            status_code=400
        )

# Blob trigger example
def main(myblob: func.InputStream):
    logging.info(f"Processing blob: {myblob.name}")
    logging.info(f"Blob size: {myblob.length} bytes")
    
    # Process the blob
    content = myblob.read()
    # Do something with content

# Timer trigger example
def main(mytimer: func.TimerRequest) -> None:
    logging.info('Timer trigger function executed.')
    
    if mytimer.past_due:
        logging.info('The timer is past due!')
    
    # Perform scheduled task
    perform_maintenance()

# Queue trigger example
def main(msg: func.QueueMessage) -> None:
    logging.info(f'Processing queue message: {msg.get_body().decode("utf-8")}')
    
    # Process message
    process_order(msg.get_json())
```

#### function.json Configuration

```json
{
  "bindings": [
    {
      "authLevel": "function",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get", "post"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

#### Azure Functions CLI Examples

```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Create function app locally
func init myFunctionApp --python
cd myFunctionApp

# Create new function
func new --name HttpTrigger --template "HTTP trigger"

# Run locally
func start

# Create function app in Azure
az functionapp create \
  --resource-group myResourceGroup \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name myuniquefunctionapp123 \
  --storage-account mystorageaccount123 \
  --os-type Linux

# Deploy to Azure
func azure functionapp publish myuniquefunctionapp123

# View logs
func azure functionapp logstream myuniquefunctionapp123

# Set application settings
az functionapp config appsettings set \
  --name myuniquefunctionapp123 \
  --resource-group myResourceGroup \
  --settings "DB_CONNECTION_STRING=Server=..."

# Enable managed identity
az functionapp identity assign \
  --name myuniquefunctionapp123 \
  --resource-group myResourceGroup

# List functions
az functionapp function list \
  --name myuniquefunctionapp123 \
  --resource-group myResourceGroup

# Delete function app
az functionapp delete \
  --name myuniquefunctionapp123 \
  --resource-group myResourceGroup
```

#### Azure Functions Pricing

```
Plan             Price                   Timeout     Scaling
──────────────────────────────────────────────────────────────
Consumption      $0.20/million requests  10 min      Automatic
                 + $0.000016/GB-s
Premium          $0.169/vCPU hour       Unlimited    Automatic
                 + $0.0123/GB hour
Dedicated        App Service Plan cost  Unlimited    Manual/Auto

Free Tier: 1M requests + 400,000 GB-s/month
```

### Azure Logic Apps

Workflow automation (similar to AWS Step Functions).

```bash
# Create Logic App
az logic workflow create \
  --resource-group myResourceGroup \
  --location eastus \
  --name myLogicApp \
  --definition @workflow.json

# List Logic Apps
az logic workflow list \
  --resource-group myResourceGroup

# Show Logic App
az logic workflow show \
  --resource-group myResourceGroup \
  --name myLogicApp

# Run Logic App
az logic workflow run trigger \
  --resource-group myResourceGroup \
  --name myLogicApp \
  --trigger-name manual
```

## Container Services

### Azure Container Instances (ACI)

Serverless containers (similar to AWS Fargate).

```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name mycontainer \
  --image nginx:latest \
  --cpu 1 \
  --memory 1.5 \
  --dns-name-label myuniquecontainer123 \
  --ports 80

# List containers
az container list --output table

# Get container logs
az container logs \
  --resource-group myResourceGroup \
  --name mycontainer

# Execute command in container
az container exec \
  --resource-group myResourceGroup \
  --name mycontainer \
  --exec-command "/bin/bash"

# Delete container
az container delete \
  --resource-group myResourceGroup \
  --name mycontainer \
  --yes

# Create container with environment variables
az container create \
  --resource-group myResourceGroup \
  --name myapp \
  --image myregistry.azurecr.io/myapp:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    'DB_HOST'='mydb.database.windows.net' \
    'DB_NAME'='mydb' \
  --secure-environment-variables \
    'DB_PASSWORD'='SecurePassword123!' \
  --registry-login-server myregistry.azurecr.io \
  --registry-username myregistry \
  --registry-password <password>
```

### Azure Kubernetes Service (AKS)

Managed Kubernetes.

```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name myAKSCluster \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --enable-managed-identity \
  --generate-ssh-keys \
  --network-plugin azure \
  --enable-addons monitoring

# Get credentials
az aks get-credentials \
  --resource-group myResourceGroup \
  --name myAKSCluster

# Verify connection
kubectl get nodes

# Scale cluster
az aks scale \
  --resource-group myResourceGroup \
  --name myAKSCluster \
  --node-count 5

# Upgrade cluster
az aks upgrade \
  --resource-group myResourceGroup \
  --name myAKSCluster \
  --kubernetes-version 1.28.0

# Enable cluster autoscaler
az aks update \
  --resource-group myResourceGroup \
  --name myAKSCluster \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10

# List available versions
az aks get-versions --location eastus --output table

# Delete cluster
az aks delete \
  --resource-group myResourceGroup \
  --name myAKSCluster \
  --yes
```

### Azure Container Registry (ACR)

Docker registry (similar to AWS ECR).

```bash
# Create container registry
az acr create \
  --resource-group myResourceGroup \
  --name myuniqueregistry123 \
  --sku Basic

# Login to registry
az acr login --name myuniqueregistry123

# Tag image
docker tag myapp:latest myuniqueregistry123.azurecr.io/myapp:v1.0

# Push image
docker push myuniqueregistry123.azurecr.io/myapp:v1.0

# List images
az acr repository list --name myuniqueregistry123 --output table

# List tags
az acr repository show-tags \
  --name myuniqueregistry123 \
  --repository myapp \
  --output table

# Delete image
az acr repository delete \
  --name myuniqueregistry123 \
  --image myapp:v1.0 \
  --yes
```

## Security Services

### Azure Active Directory (Azure Entra ID)

Identity and access management.

```bash
# Create user
az ad user create \
  --display-name "Alice Smith" \
  --user-principal-name alice@contoso.com \
  --password SecurePassword123!

# Create group
az ad group create \
  --display-name Developers \
  --mail-nickname developers

# Add user to group
az ad group member add \
  --group Developers \
  --member-id <user-object-id>

# Create service principal
az ad sp create-for-rbac \
  --name myServicePrincipal \
  --role Contributor \
  --scopes /subscriptions/<subscription-id>

# List users
az ad user list --output table

# List groups
az ad group list --output table
```

### Azure Key Vault

Secrets management (similar to AWS Secrets Manager).

```bash
# Create Key Vault
az keyvault create \
  --name myuniquekeyvault123 \
  --resource-group myResourceGroup \
  --location eastus

# Set secret
az keyvault secret set \
  --vault-name myuniquekeyvault123 \
  --name dbpassword \
  --value "SecurePassword123!"

# Get secret
az keyvault secret show \
  --vault-name myuniquekeyvault123 \
  --name dbpassword \
  --query value \
  --output tsv

# List secrets
az keyvault secret list \
  --vault-name myuniquekeyvault123 \
  --output table

# Delete secret
az keyvault secret delete \
  --vault-name myuniquekeyvault123 \
  --name dbpassword

# Set access policy
az keyvault set-policy \
  --name myuniquekeyvault123 \
  --upn alice@contoso.com \
  --secret-permissions get list set delete

# Create certificate
az keyvault certificate create \
  --vault-name myuniquekeyvault123 \
  --name mycert \
  --policy "$(az keyvault certificate get-default-policy)"

# Import certificate
az keyvault certificate import \
  --vault-name myuniquekeyvault123 \
  --name imported-cert \
  --file certificate.pfx
```

#### Use Key Vault in Application (Python)

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Create client
credential = DefaultAzureCredential()
vault_url = "https://myuniquekeyvault123.vault.azure.net"
client = SecretClient(vault_url=vault_url, credential=credential)

# Get secret
secret = client.get_secret("dbpassword")
print(f"Secret value: {secret.value}")

# Set secret
client.set_secret("newsecret", "newvalue")

# List secrets
secrets = client.list_properties_of_secrets()
for secret in secrets:
    print(f"Secret name: {secret.name}")

# Delete secret
client.begin_delete_secret("newsecret").wait()
```

### Azure RBAC (Role-Based Access Control)

```bash
# List role definitions
az role definition list --output table

# Assign role to user
az role assignment create \
  --assignee alice@contoso.com \
  --role Contributor \
  --scope /subscriptions/<subscription-id>/resourceGroups/myResourceGroup

# Assign role to service principal
az role assignment create \
  --assignee <service-principal-object-id> \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/mystorageaccount123

# List role assignments
az role assignment list \
  --assignee alice@contoso.com \
  --output table

# Remove role assignment
az role assignment delete \
  --assignee alice@contoso.com \
  --role Contributor \
  --scope /subscriptions/<subscription-id>/resourceGroups/myResourceGroup

# Create custom role
az role definition create --role-definition @custom-role.json
```

#### Custom Role Definition Example

```json
{
  "Name": "Custom VM Operator",
  "Description": "Can start and stop VMs",
  "Actions": [
    "Microsoft.Compute/virtualMachines/start/action",
    "Microsoft.Compute/virtualMachines/restart/action",
    "Microsoft.Compute/virtualMachines/deallocate/action",
    "Microsoft.Compute/virtualMachines/read"
  ],
  "NotActions": [],
  "AssignableScopes": [
    "/subscriptions/<subscription-id>"
  ]
}
```

## Monitoring and Management

### Azure Monitor

Monitoring and observability (similar to CloudWatch).

```bash
# Create action group
az monitor action-group create \
  --name myActionGroup \
  --resource-group myResourceGroup \
  --short-name myAG \
  --email-receiver name=admin email=admin@example.com

# Create metric alert
az monitor metrics alert create \
  --name high-cpu \
  --resource-group myResourceGroup \
  --scopes /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachines/myVM \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action myActionGroup

# List alerts
az monitor metrics alert list \
  --resource-group myResourceGroup

# Query metrics
az monitor metrics list \
  --resource /subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Compute/virtualMachines/myVM \
  --metric "Percentage CPU" \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --interval PT1H
```

### Azure Log Analytics

Log collection and analysis.

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group myResourceGroup \
  --workspace-name myWorkspace \
  --location eastus

# Query logs (KQL - Kusto Query Language)
az monitor log-analytics query \
  --workspace myWorkspace \
  --analytics-query "AzureActivity | where TimeGenerated > ago(1h) | summarize count() by OperationName"

# Example KQL queries
# All logs from last hour
"AzureActivity | where TimeGenerated > ago(1h)"

# Count errors by resource
"AzureDiagnostics | where Level == 'Error' | summarize count() by Resource"

# VM performance - CPU over 80%
"Perf | where CounterName == '% Processor Time' and CounterValue > 80"

# Failed login attempts
"SigninLogs | where ResultType != 0 | project TimeGenerated, UserPrincipalName, ResultType, ResultDescription"
```

### Azure Application Insights

Application performance monitoring.

```python
from applicationinsights import TelemetryClient

# Initialize client
tc = TelemetryClient('<instrumentation-key>')

# Track event
tc.track_event('UserLogin', {'user': 'alice@example.com'})

# Track metric
tc.track_metric('request_duration', 125.5)

# Track exception
try:
    result = 1 / 0
except Exception as e:
    tc.track_exception()

# Track request
tc.track_request('GET /api/users', 'https://myapi.com/api/users', True, 200, 125)

# Track dependency
tc.track_dependency('SQL', 'mydb.database.windows.net', 'SELECT * FROM users', 45, True, 'Query')

# Flush telemetry
tc.flush()
```

```bash
# Enable Application Insights for web app
az webapp config appsettings set \
  --resource-group myResourceGroup \
  --name myUniqueWebApp123 \
  --settings "APPINSIGHTS_INSTRUMENTATIONKEY=<instrumentation-key>"
```

## DevOps and CI/CD

### Azure DevOps

Complete DevOps platform.

#### Azure Pipelines YAML Example

```yaml
# azure-pipelines.yml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  buildConfiguration: 'Release'

stages:
- stage: Build
  jobs:
  - job: BuildJob
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/ --junitxml=junit/test-results.xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      inputs:
        testResultsFiles: '**/test-results.xml'
    
    - script: |
        docker build -t myapp:$(Build.BuildId) .
      displayName: 'Build Docker image'
    
    - task: Docker@2
      inputs:
        containerRegistry: 'myACR'
        repository: 'myapp'
        command: 'push'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  dependsOn: Build
  jobs:
  - deployment: DeployJob
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureWebAppContainer@1
            inputs:
              azureSubscription: 'myServiceConnection'
              appName: 'myUniqueWebApp123'
              containers: 'myregistry.azurecr.io/myapp:$(Build.BuildId)'
```

### Azure CLI for DevOps

```bash
# Create Azure DevOps project
az devops project create --name MyProject --org https://dev.azure.com/myorg

# Create pipeline
az pipelines create \
  --name MyPipeline \
  --repository https://github.com/user/repo \
  --branch main \
  --yml-path azure-pipelines.yml

# Run pipeline
az pipelines run --name MyPipeline

# List pipelines
az pipelines list --output table

# Show pipeline runs
az pipelines runs list --pipeline-name MyPipeline --output table
```

## AI and Machine Learning

### Azure OpenAI Service

Access to OpenAI models (GPT-4, GPT-3.5, DALL-E, Whisper).

```python
import openai

# Configure
openai.api_type = "azure"
openai.api_base = "https://myopenai.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "<api-key>"

# Generate completion
response = openai.ChatCompletion.create(
    engine="gpt-4",  # deployment name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain cloud computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=800
)

print(response.choices[0].message.content)

# Generate image
response = openai.Image.create(
    prompt="A futuristic cloud data center",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']
```

### Azure Cognitive Services

Pre-built AI services.

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Text Analytics
endpoint = "https://myservice.cognitiveservices.azure.com/"
key = "<api-key>"

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Sentiment analysis
documents = ["I love Azure!", "This is terrible."]
result = client.analyze_sentiment(documents)

for doc in result:
    print(f"Sentiment: {doc.sentiment}, Confidence: {doc.confidence_scores}")

# Entity recognition
result = client.recognize_entities(["Microsoft was founded by Bill Gates."])
for doc in result:
    for entity in doc.entities:
        print(f"Entity: {entity.text}, Category: {entity.category}")

# Key phrase extraction
result = client.extract_key_phrases(["Azure is a cloud computing platform."])
for doc in result:
    print(f"Key phrases: {doc.key_phrases}")
```

### Azure Machine Learning

End-to-end ML platform.

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig

# Connect to workspace
ws = Workspace.from_config()

# Create experiment
experiment = Experiment(workspace=ws, name='my-experiment')

# Configure training run
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    compute_target='cpu-cluster',
    environment='AzureML-sklearn-1.0'
)

# Submit run
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Register model
model = run.register_model(
    model_name='my-model',
    model_path='outputs/model.pkl'
)

# Deploy model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(
    entry_script='score.py',
    environment='AzureML-sklearn-1.0'
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name='my-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```

## Architecture Examples

### Three-Tier Web Application

```
                         Internet
                             │
                    ┌────────▼────────┐
                    │  Azure Front    │  CDN
                    │  Door           │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Azure DNS      │
                    └────────┬────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Virtual Network                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Public Subnet (AZ-1)    Public Subnet (AZ-2)       │  │
│  │  ┌──────────────────┐    ┌──────────────────┐      │  │
│  │  │ Application      │    │ Application      │      │  │
│  │  │ Gateway + WAF    │    │ Gateway + WAF    │      │  │
│  │  └────────┬─────────┘    └────────┬─────────┘      │  │
│  └───────────┼──────────────────────┼─────────────────┘  │
│              │                      │                      │
│  ┌───────────▼──────────────────────▼─────────────────┐  │
│  │  Private Subnet (AZ-1)   Private Subnet (AZ-2)     │  │
│  │  ┌────────────────┐       ┌────────────────┐       │  │
│  │  │ VMSS           │       │ VMSS           │       │  │
│  │  │  ┌──┐ ┌──┐     │       │  ┌──┐ ┌──┐     │       │  │
│  │  │  │VM│ │VM│     │       │  │VM│ │VM│     │       │  │
│  │  │  └──┘ └──┘     │       │  └──┘ └──┘     │       │  │
│  │  └────────┬───────┘       └────────┬───────┘       │  │
│  └───────────┼──────────────────────┼─────────────────┘  │
│              │                      │                      │
│  ┌───────────▼──────────────────────▼─────────────────┐  │
│  │  Database Subnet (AZ-1)  Database Subnet (AZ-2)    │  │
│  │  ┌──────────────┐        ┌──────────────┐          │  │
│  │  │ Azure SQL    │◄──────▶│ Azure SQL    │          │  │
│  │  │ Primary      │        │ Secondary    │          │  │
│  │  └──────────────┘        └──────────────┘          │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────┐                   │  │
│  │  │ Azure Cache for Redis        │                   │  │
│  │  └──────────────────────────────┘                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Additional Services:                                       │
│  ├─ Blob Storage: Static assets                             │
│  ├─ Key Vault: Secrets management                           │
│  ├─ Monitor: Monitoring and alerts                          │
│  └─ Application Insights: APM                               │
└──────────────────────────────────────────────────────────────┘
```

### Serverless Microservices

```
                        ┌─────────────┐
                        │   Users     │
                        └──────┬──────┘
                               │
                      ┌────────▼────────┐
                      │  Azure Front    │
                      │  Door + Blob    │
                      │  (Frontend)     │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  API Management │
                      └────────┬────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼──────┐         ┌────▼──────┐         ┌────▼──────┐
   │ Function  │         │ Function  │         │ Function  │
   │ User Svc  │         │ Order Svc │         │ Pay Svc   │
   └────┬──────┘         └────┬──────┘         └────┬──────┘
        │                     │                     │
   ┌────▼──────┐         ┌────▼──────┐         ┌────▼──────┐
   │Cosmos DB  │         │Cosmos DB  │         │Cosmos DB  │
   │Users      │         │Orders     │         │Payments   │
   └───────────┘         └───────────┘         └───────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                       ┌──────▼──────┐
                       │  Event Grid │
                       │  Service Bus│
                       └─────────────┘
```

## Azure vs AWS Comparison

### Service Mapping

```
Service Category      Azure                      AWS
─────────────────────────────────────────────────────────────────
Compute
  VMs                 Virtual Machines           EC2
  Auto-scaling        VMSS                       Auto Scaling
  Serverless          Functions                  Lambda
  Containers          AKS / ACI                  EKS / ECS / Fargate
  PaaS                App Service                Elastic Beanstalk

Storage
  Object              Blob Storage               S3
  Block               Managed Disks              EBS
  File                Azure Files                EFS
  Archive             Archive Storage            Glacier

Database
  Relational          SQL Database               RDS
  NoSQL Document      Cosmos DB                  DynamoDB
  Cache               Cache for Redis            ElastiCache
  Data Warehouse      Synapse Analytics          Redshift

Networking
  Virtual Network     VNet                       VPC
  Load Balancer       Load Balancer / App GW     ELB / ALB / NLB
  CDN                 Front Door / CDN           CloudFront
  DNS                 Azure DNS                  Route 53
  VPN                 VPN Gateway                VPN Gateway

Security
  Identity            Azure AD (Entra ID)        IAM / Cognito
  Secrets             Key Vault                  Secrets Manager
  Encryption          Key Vault                  KMS

Monitoring
  Metrics             Azure Monitor              CloudWatch
  Logs                Log Analytics              CloudWatch Logs
  APM                 Application Insights       X-Ray
  Audit               Activity Log               CloudTrail

DevOps
  CI/CD               Azure DevOps / Pipelines   CodePipeline
  Repository          Azure Repos                CodeCommit
  Container Registry  ACR                        ECR

AI/ML
  Pre-trained Models  Cognitive Services         AI Services
  ML Platform         Machine Learning           SageMaker
  GenAI               OpenAI Service             Bedrock
```

### Key Differences

```
Aspect                Azure                      AWS
─────────────────────────────────────────────────────────────────
Market Share          ~23%                       ~32%
Launch Year           2010                       2006
Focus                 Enterprise / Hybrid        Startups / Flexibility
Integration           Microsoft stack            Broad ecosystem
Regions               60+ regions                30+ regions
Pricing               Per-minute billing         Per-second billing
Support               Strong enterprise          Extensive documentation
Compliance            Most certifications        Extensive certifications
Hybrid Cloud          Azure Arc (best-in-class)  Outposts
Windows Workloads     Native integration         Good support
```

### When to Choose Azure

```
✓ Heavy Microsoft stack usage (Windows, .NET, SQL Server)
✓ Enterprise Active Directory integration needed
✓ Hybrid cloud requirements (on-premises + cloud)
✓ Existing Microsoft licensing (Azure Hybrid Benefit)
✓ Office 365 / Dynamics 365 integration
✓ Strong compliance requirements
✓ European data centers needed
✓ .NET development team
```

### When to Choose AWS

```
✓ Largest service selection needed
✓ Startup with flexible requirements
✓ Open-source technologies focus
✓ Mature ecosystem and tooling important
✓ Broadest region availability needed
✓ Extensive third-party integrations
✓ Strong serverless requirements
✓ Largest community and resources
```

## Cost Optimization

### Azure Cost Management

```bash
# Create budget
az consumption budget create \
  --budget-name myBudget \
  --amount 1000 \
  --category Cost \
  --time-grain Monthly \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# View cost analysis
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# Get cost forecast
az consumption forecast list

# Enable auto-shutdown for VMs
az vm auto-shutdown \
  --resource-group myResourceGroup \
  --name myVM \
  --time 1900 \
  --timezone "Pacific Standard Time"
```

### Cost Optimization Strategies

```
┌──────────────────────────────────────────────────────────┐
│ Azure Cost Optimization Checklist                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ Compute                                                   │
│ ☐ Use Reserved Instances (up to 72% discount)            │
│ ☐ Use Spot VMs for fault-tolerant workloads              │
│ ☐ Right-size VMs based on metrics                        │
│ ☐ Use Azure Hybrid Benefit for Windows/SQL               │
│ ☐ Deallocate VMs when not in use                         │
│ ☐ Use Azure Functions for event-driven workloads         │
│ ☐ Enable auto-shutdown for dev/test VMs                  │
│                                                           │
│ Storage                                                   │
│ ☐ Use lifecycle management policies                      │
│ ☐ Move infrequent data to Cool/Archive tiers             │
│ ☐ Delete unused disks and snapshots                      │
│ ☐ Use LRS instead of GRS when possible                   │
│ ☐ Enable blob versioning only when needed                │
│                                                           │
│ Database                                                  │
│ ☐ Use serverless for SQL Database with variable load     │
│ ☐ Right-size database tiers                              │
│ ☐ Use Cosmos DB autoscale                                │
│ ☐ Implement connection pooling                           │
│ ☐ Pause dev/test databases when not in use               │
│                                                           │
│ Network                                                   │
│ ☐ Use Azure Front Door to reduce data transfer           │
│ ☐ Use VNet peering instead of VPN when possible          │
│ ☐ Consolidate data transfer within same region           │
│ ☐ Use private endpoints to avoid data transfer costs     │
│                                                           │
│ Monitoring                                                │
│ ☐ Set up Azure Cost Management + Billing alerts          │
│ ☐ Use Azure Advisor cost recommendations                 │
│ ☐ Review Advisor score regularly                         │
│ ☐ Use tags for cost allocation                           │
│ ☐ Review Underutilized Resources report                  │
└──────────────────────────────────────────────────────────┘
```

### Azure Pricing Calculator

Use Azure Pricing Calculator: https://azure.microsoft.com/pricing/calculator/

### Example Monthly Costs

```
Service                   Configuration           Monthly Cost (Approx)
─────────────────────────────────────────────────────────────────────
VM (B2s)                  2 vCPU, 4GB, Linux     $30
Managed Disk (128GB)      Premium SSD            $20
SQL Database (S0)         10 DTUs                $15
Cosmos DB                 400 RU/s               $24
Blob Storage (100GB)      Hot tier               $2
Data Transfer             50GB outbound          $4
App Service (B1)          1 core, 1.75GB         $55
Functions                 1M requests            $0.20
                                                 ─────────
                          Total:                 ~$150/month
```

## Best Practices

### Security Best Practices

```
1. Identity and Access
   ├─ Use Azure AD (Entra ID) for all authentication
   ├─ Enable MFA for all users
   ├─ Use managed identities instead of service principals
   ├─ Implement RBAC with least privilege
   ├─ Use Azure AD Privileged Identity Management (PIM)
   └─ Enable Conditional Access policies

2. Network Security
   ├─ Use Network Security Groups (NSGs)
   ├─ Implement Azure Firewall or third-party NVA
   ├─ Use private endpoints for PaaS services
   ├─ Enable DDoS Protection Standard for production
   ├─ Use Application Gateway with WAF
   └─ Enable VNet service endpoints

3. Data Protection
   ├─ Enable encryption at rest for all services
   ├─ Use Azure Key Vault for secrets
   ├─ Enable TLS 1.2+ for data in transit
   ├─ Implement backup and disaster recovery
   ├─ Enable soft delete for Key Vault and Storage
   └─ Use customer-managed keys when required

4. Monitoring and Compliance
   ├─ Enable Azure Security Center (Defender for Cloud)
   ├─ Use Azure Sentinel for SIEM
   ├─ Enable Azure Monitor and Log Analytics
   ├─ Implement Azure Policy for governance
   ├─ Use Azure Blueprints for compliance
   └─ Regular security assessments

5. Application Security
   ├─ Use Web Application Firewall (WAF)
   ├─ Implement API Management security features
   ├─ Enable Application Insights
   ├─ Use Azure Front Door for global apps
   └─ Regular vulnerability scanning
```

### Reliability Best Practices

```
1. High Availability
   ├─ Deploy across Availability Zones
   ├─ Use zone-redundant services
   ├─ Implement auto-scaling
   ├─ Use Azure Load Balancer / Application Gateway
   └─ Consider multi-region for critical workloads

2. Disaster Recovery
   ├─ Define RPO and RTO requirements
   ├─ Use Azure Site Recovery
   ├─ Implement geo-redundant storage
   ├─ Regular backup and restore testing
   └─ Document DR procedures

3. Monitoring
   ├─ Use Azure Monitor for all resources
   ├─ Set up alerts for critical metrics
   ├─ Implement health checks
   ├─ Use Application Insights for APM
   └─ Create dashboards for visibility

4. Resilience
   ├─ Implement retry logic
   ├─ Use circuit breaker pattern
   ├─ Implement graceful degradation
   ├─ Use queue-based load leveling
   └─ Regular chaos engineering tests
```

## CLI Reference

### Common CLI Patterns

```bash
# Use --output for different formats
az vm list --output table
az vm list --output json
az vm list --output yaml
az vm list --output tsv

# Use --query for filtering (JMESPath)
az vm list --query "[].{name:name, powerState:powerState}"
az vm list --query "[?powerState=='VM running'].name"

# Use --resource-group shorthand
az vm list -g myResourceGroup

# Use --verbose for debugging
az vm create --verbose ...

# Get help
az vm --help
az vm create --help

# Interactive mode
az interactive

# Configure defaults
az configure --defaults group=myResourceGroup location=eastus

# Show defaults
az configure --list-defaults
```

### Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

alias azvm='az vm list --output table'
alias azrunning='az vm list --query "[?powerState=='\''VM running'\''].{name:name, resourceGroup:resourceGroup}" --output table'
alias azstorage='az storage account list --output table'
alias azsql='az sql db list --output table'
alias azgroup='az group list --output table'
```

## Certification Paths

### Azure Certification Roadmap

```
Foundational
    │
    └─ AZ-900: Azure Fundamentals
        │
        ├─ Associate Level
        │   ├─ AZ-104: Azure Administrator
        │   ├─ AZ-204: Azure Developer
        │   └─ AZ-400: DevOps Engineer
        │
        └─ Expert Level
            ├─ AZ-305: Azure Solutions Architect
            └─ AZ-400: DevOps Engineer (with AZ-104/204)
            
        Specialty (Optional)
        ├─ AZ-500: Security Technologies
        ├─ AI-102: AI Engineer
        ├─ DP-203: Data Engineer
        └─ AZ-700: Network Engineer
```

## Resources

### Official Documentation
- Azure Documentation: https://docs.microsoft.com/azure
- Azure CLI Reference: https://docs.microsoft.com/cli/azure/
- Azure SDK Documentation: https://azure.github.io/azure-sdk/

### Learning Resources
- Microsoft Learn: https://learn.microsoft.com/training/
- Azure Free Account: https://azure.microsoft.com/free/
- Azure Architecture Center: https://docs.microsoft.com/azure/architecture/
- Azure Samples: https://github.com/Azure-Samples
- Azure Friday: https://azure.microsoft.com/resources/videos/azure-friday/

### Community
- r/AZURE: Reddit community
- Microsoft Q&A: https://docs.microsoft.com/answers/
- Azure Community Support: https://azure.microsoft.com/support/community/
- Azure User Groups: https://www.meetup.com/pro/azureug

### Tools
- Azure CLI: Command-line interface
- Azure PowerShell: PowerShell modules
- Azure SDKs: Python, JavaScript, Java, .NET, Go
- Bicep: Azure-native IaC
- Terraform: Multi-cloud IaC
- Azure Storage Explorer: GUI for storage
- Azure Data Studio: Database management

### Pricing
- Azure Pricing Calculator: https://azure.microsoft.com/pricing/calculator/
- Azure Cost Management: https://azure.microsoft.com/services/cost-management/
- Total Cost of Ownership (TCO) Calculator: https://azure.microsoft.com/pricing/tco/

---

**Updated**: January 2025
