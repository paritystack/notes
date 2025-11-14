# Terraform

Infrastructure as Code (IaC) tool for provisioning and managing cloud infrastructure across multiple providers.

## Core Concepts

### Infrastructure as Code (IaC)
- **Declarative Configuration**: Define desired state, Terraform handles the rest
- **Version Control**: Infrastructure code stored in Git
- **Reproducibility**: Same config creates identical infrastructure
- **Collaboration**: Team-based infrastructure management

### Key Components

#### Providers
Plugins that interact with cloud platforms and services:
```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

#### Resources
Infrastructure components to create and manage:
```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"

  tags = {
    Name = "WebServer"
    Environment = "production"
  }
}
```

#### Data Sources
Query existing infrastructure:
```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}
```

#### Variables
Parameterize configurations:
```hcl
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
```

#### Outputs
Export values for reference:
```hcl
output "instance_ip" {
  description = "Public IP of instance"
  value       = aws_instance.web.public_ip
}

output "instance_id" {
  description = "ID of instance"
  value       = aws_instance.web.id
}
```

#### Modules
Reusable infrastructure components:
```hcl
module "vpc" {
  source = "./modules/vpc"

  vpc_cidr = "10.0.0.0/16"
  environment = var.environment
}

module "web_cluster" {
  source = "terraform-aws-modules/ec2-instance/aws"
  version = "5.0.0"

  name           = "web-cluster"
  instance_count = 3

  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  subnet_id     = module.vpc.public_subnet_ids[0]
}
```

## State Management

### Local State
Default storage in `terraform.tfstate`:
```hcl
# terraform.tfstate stores current infrastructure state
# NOT recommended for team environments
```

### Remote State
Store state remotely for collaboration:

#### S3 Backend
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

#### Terraform Cloud
```hcl
terraform {
  cloud {
    organization = "my-org"

    workspaces {
      name = "production"
    }
  }
}
```

### State Locking
Prevent concurrent modifications:
```bash
# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name           = "terraform-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

## Common Operations

### Initialize
Download providers and modules:
```bash
# Initialize working directory
terraform init

# Upgrade providers
terraform init -upgrade

# Reconfigure backend
terraform init -reconfigure
```

### Plan
Preview infrastructure changes:
```bash
# Show execution plan
terraform plan

# Save plan to file
terraform plan -out=tfplan

# Show specific resource changes
terraform plan -target=aws_instance.web

# Plan with variable file
terraform plan -var-file="prod.tfvars"
```

### Apply
Create or update infrastructure:
```bash
# Apply with confirmation
terraform apply

# Apply saved plan
terraform apply tfplan

# Apply without confirmation (CI/CD)
terraform apply -auto-approve

# Apply specific resource
terraform apply -target=aws_instance.web
```

### Destroy
Remove infrastructure:
```bash
# Destroy all resources
terraform destroy

# Destroy specific resource
terraform destroy -target=aws_instance.web

# Destroy without confirmation
terraform destroy -auto-approve
```

### State Operations
```bash
# List resources in state
terraform state list

# Show resource details
terraform state show aws_instance.web

# Remove resource from state
terraform state rm aws_instance.web

# Move resource in state
terraform state mv aws_instance.web aws_instance.app

# Pull remote state
terraform state pull

# Push local state
terraform state push terraform.tfstate
```

### Import
Import existing resources:
```bash
# Import EC2 instance
terraform import aws_instance.web i-1234567890abcdef0

# Import with module
terraform import module.vpc.aws_vpc.main vpc-1234567890abcdef0
```

### Workspace Management
Multiple environments with same config:
```bash
# List workspaces
terraform workspace list

# Create workspace
terraform workspace new staging

# Switch workspace
terraform workspace select production

# Show current workspace
terraform workspace show

# Delete workspace
terraform workspace delete staging
```

## Common Patterns

### Multi-Environment Setup
```hcl
# environments/prod/main.tf
module "infrastructure" {
  source = "../../modules/infrastructure"

  environment    = "production"
  instance_type  = "t3.large"
  instance_count = 5
  enable_backup  = true
}

# environments/dev/main.tf
module "infrastructure" {
  source = "../../modules/infrastructure"

  environment    = "development"
  instance_type  = "t3.micro"
  instance_count = 1
  enable_backup  = false
}
```

### Module Composition
```hcl
# modules/web-app/main.tf
module "networking" {
  source = "../networking"

  vpc_cidr    = var.vpc_cidr
  environment = var.environment
}

module "security" {
  source = "../security"

  vpc_id      = module.networking.vpc_id
  environment = var.environment
}

module "compute" {
  source = "../compute"

  vpc_id            = module.networking.vpc_id
  subnet_ids        = module.networking.private_subnet_ids
  security_group_id = module.security.web_sg_id
}
```

### Dynamic Blocks
```hcl
resource "aws_security_group" "web" {
  name   = "web-sg"
  vpc_id = var.vpc_id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}

# Usage
ingress_rules = [
  {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  },
  {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
]
```

### Conditional Resources
```hcl
resource "aws_instance" "web" {
  count = var.create_instance ? 1 : 0

  ami           = var.ami_id
  instance_type = var.instance_type
}

# Or with for_each
resource "aws_s3_bucket" "logs" {
  for_each = var.enable_logging ? { "logs" = true } : {}

  bucket = "app-logs-${each.key}"
}
```

### Remote State Data Source
```hcl
data "terraform_remote_state" "networking" {
  backend = "s3"

  config = {
    bucket = "terraform-state"
    key    = "networking/terraform.tfstate"
    region = "us-east-1"
  }
}

resource "aws_instance" "app" {
  ami           = var.ami_id
  instance_type = "t3.micro"
  subnet_id     = data.terraform_remote_state.networking.outputs.subnet_id
}
```

### Locals for Computed Values
```hcl
locals {
  common_tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
    Project     = var.project_name
  }

  name_prefix = "${var.project_name}-${var.environment}"

  instance_count = var.environment == "production" ? 5 : 2
}

resource "aws_instance" "app" {
  count = local.instance_count

  ami           = var.ami_id
  instance_type = "t3.micro"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-app-${count.index}"
    }
  )
}
```

## Best Practices

### Code Organization
```
terraform/
├── environments/
│   ├── prod/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars
│   └── dev/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── ec2/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── shared/
    ├── providers.tf
    └── backend.tf
```

### Version Constraints
```hcl
terraform {
  required_version = "~> 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}
```

### Variable Validation
```hcl
variable "environment" {
  description = "Environment name"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string

  validation {
    condition     = can(regex("^t[23]\\.", var.instance_type))
    error_message = "Only t2 and t3 instance types allowed."
  }
}
```

### Sensitive Data
```hcl
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

output "db_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}
```

### Naming Conventions
```hcl
# Use descriptive resource names
resource "aws_instance" "web_server" { }      # Good
resource "aws_instance" "instance1" { }       # Bad

# Use consistent naming patterns
locals {
  name_prefix = "${var.project}-${var.environment}"
}

# Tag all resources
tags = {
  Name        = "${local.name_prefix}-web"
  Environment = var.environment
  ManagedBy   = "Terraform"
  CostCenter  = var.cost_center
}
```

### State Management
- **Always** use remote state for teams
- **Enable** state locking with DynamoDB
- **Encrypt** state files (sensitive data)
- **Never** commit state files to Git
- **Backup** state regularly
- **Use** workspaces for isolation

### Security
```hcl
# Don't hardcode credentials
provider "aws" {
  region = var.region
  # Uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
}

# Use IAM roles instead
provider "aws" {
  region = var.region
  assume_role {
    role_arn = var.assume_role_arn
  }
}

# Encrypt sensitive data
resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Terraform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.6.0

      - name: Terraform Init
        run: terraform init

      - name: Terraform Format
        run: terraform fmt -check

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        run: terraform plan -out=tfplan
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve tfplan
```

## Troubleshooting

### Common Issues

#### State Lock Errors
```bash
# Force unlock (use with caution)
terraform force-unlock <lock-id>
```

#### Resource Already Exists
```bash
# Import existing resource
terraform import aws_instance.web i-1234567890abcdef0
```

#### Drift Detection
```bash
# Check for drift
terraform plan -refresh-only

# Update state with actual infrastructure
terraform apply -refresh-only
```

#### Debug Mode
```bash
# Enable verbose logging
export TF_LOG=DEBUG
terraform plan

# Log to file
export TF_LOG_PATH=terraform.log
terraform apply
```

## Tools & Utilities

- **terraform fmt**: Format code
- **terraform validate**: Validate syntax
- **terraform graph**: Visualize dependencies
- **tflint**: Linting tool
- **terragrunt**: DRY configurations
- **infracost**: Cost estimation
- **checkov**: Security scanning
- **terraform-docs**: Generate documentation

## Resources

- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [Terraform Registry](https://registry.terraform.io/)
- [Best Practices](https://www.terraform-best-practices.com/)
