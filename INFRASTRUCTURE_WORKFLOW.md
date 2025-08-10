# Infrastructure as Code - Safe Workflow

## 🔒 Core Principle: Plan-Only Agent, Human-Approved Applies

The `infra-pr` agent **NEVER** runs `terraform apply`. It only:
1. Creates plans
2. Analyzes risks
3. Opens PRs
4. Waits for human approval

## 🚀 Complete Infrastructure Workflow

### 1. Request Infrastructure Change

```bash
# Via GitHub comment
@claude update RDS instance size in infra/, run terraform plan, and open a PR with risks

# Via CLI
claude infra-pr "Increase RDS instance from db.t3.medium to db.t3.large"

# Via slash command
claude /infra-plan production "Increase RDS instance size"
```

### 2. Agent Creates Plan (Never Applies)

The `infra-pr` agent:
```bash
# Validates configuration
terraform validate
terraform fmt -check
tflint

# Generates plan
terraform plan -out=tfplan.binary

# Analyzes risks
# - Checks for deletions (HIGH RISK)
# - Checks for replacements (HIGH RISK)
# - Checks for database changes (MEDIUM RISK)

# Estimates costs
infracost breakdown --path .

# Creates PR with all information
```

### 3. PR Created with Full Context

```markdown
## 🏗️ Infrastructure Change Request

### Summary
Increase RDS instance size from db.t3.medium to db.t3.large

### Risk Assessment: MEDIUM
- 📊 1 database resource affected
- ⚠️ Potential 5-10 minute downtime during resize
- 💰 Cost increase: $73/month → $146/month

### Terraform Plan
```
~ aws_db_instance.main
    ~ instance_class = "db.t3.medium" -> "db.t3.large"
```

### Required Approvals
- [ ] DevOps review
- [ ] Budget owner approval (>$50/month increase)
- [ ] Database team review

### Rollback Plan
terraform apply -replace="aws_db_instance.main" -var="instance_class=db.t3.medium"
```

### 4. Human Review & Approval

GitHub Environment Protection Rules:
- **Development**: No approval needed
- **Staging**: 1 reviewer required
- **Production**: 2 reviewers + manual approval

### 5. Apply After Approval

Once PR is approved and merged:
```yaml
# GitHub Actions automatically:
1. Runs terraform apply in staging
2. Waits for manual approval for production
3. Runs terraform apply in production
4. Notifies team of completion
```

## 🎯 Real-World Examples

### Example 1: Safe RDS Resize

```bash
# Developer request
@claude increase production RDS to db.r5.xlarge for Black Friday traffic

# Claude infra-pr agent responds:
"I'll create a plan for RDS resize. This is a HIGH RISK change that requires multiple approvals."

# Creates PR #123 with:
- Full plan showing instance resize
- Risk: HIGH (database change in production)
- Cost: +$200/month
- Downtime: ~10 minutes during maintenance window
- Rollback: Resize back to original

# Requires:
- 2 DevOps approvals
- 1 Management approval (cost > $100)
- DBA team review
```

### Example 2: Add Security Group Rule

```bash
# Request
claude /infra-plan staging "Allow port 443 from CloudFlare IPs"

# Agent creates PR with:
- Plan showing new security group rules
- Risk: LOW (additive change only)
- No cost impact
- No downtime
- Auto-approved after 1 review
```

### Example 3: Accidental Deletion Prevention

```bash
# Developer accidentally requests
@claude remove unused VPC in production

# Claude infra-pr agent:
1. Runs plan
2. Detects 15 resources will be DELETED
3. Sets risk level: CRITICAL
4. Creates PR with big warnings:

"⚠️ CRITICAL: This will DELETE 15 resources including:
- 1 VPC
- 4 Subnets
- 2 NAT Gateways (DATA LOSS RISK)
- 8 Security Groups

This requires:
- 3 DevOps approvals
- CTO approval
- Documented migration plan"

# Deletion prevented by review process!
```

## 🔐 GitHub Environments Configuration

### Setup Production Environment

1. Go to Settings → Environments → New Environment → "production"

2. Add Protection Rules:
```yaml
Required reviewers: 2
- @devops-lead
- @infrastructure-team

Deployment branches: main only

Environment secrets:
- AWS_ROLE_ARN: arn:aws:iam::123:role/terraform-prod
- SLACK_WEBHOOK: https://hooks.slack.com/...

Wait timer: 5 minutes (allows cancellation)
```

3. Add Environment URL:
```
https://production.example.com
```

## 🛡️ Security Controls

### OIDC/Workload Identity Federation

Instead of static AWS keys:
```yaml
# GitHub Actions uses OIDC
- uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
    role-session-name: GitHubActions-${{ github.run_id }}
    aws-region: us-east-1
```

### State File Protection

```hcl
# S3 backend with encryption and versioning
terraform {
  backend "s3" {
    bucket         = "terraform-state-prod"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:..."
    dynamodb_table = "terraform-state-lock"
    
    # Require MFA for state changes
    mfa_delete = true
  }
}
```

### Policy as Code (OPA)

```rego
# policies/terraform.rego
package terraform.aws

# Deny public RDS instances
deny[msg] {
  resource := input.resource_changes[_]
  resource.type == "aws_db_instance"
  resource.change.after.publicly_accessible == true
  msg := sprintf("RDS instance %v cannot be public", [resource.address])
}

# Deny EC2 without encryption
deny[msg] {
  resource := input.resource_changes[_]
  resource.type == "aws_instance"
  resource.change.after.root_block_device[_].encrypted == false
  msg := sprintf("EC2 instance %v must have encrypted storage", [resource.address])
}

# Require tagging
deny[msg] {
  resource := input.resource_changes[_]
  required_tags := ["Environment", "Owner", "CostCenter"]
  missing := required_tags[_]
  not resource.change.after.tags[missing]
  msg := sprintf("Resource %v missing required tag: %v", [resource.address, missing])
}
```

## 📊 Monitoring Applied Changes

After apply, monitor:

```bash
# CloudWatch dashboard
@claude create CloudWatch dashboard for recent infra changes

# Terraform state history
terraform state list
terraform state show aws_db_instance.main

# AWS Config for compliance
aws configservice get-compliance-details-by-resource \
  --resource-type AWS::RDS::DBInstance \
  --resource-id db-instance-1
```

## 🔄 Drift Detection

Scheduled drift detection:
```yaml
# .github/workflows/drift-detection.yml
name: Drift Detection
on:
  schedule:
    - cron: '0 8 * * MON'  # Weekly on Monday

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Check for drift
        run: |
          terraform plan -detailed-exitcode
          if [ $? -eq 2 ]; then
            echo "Drift detected!"
            # Create issue
            gh issue create --title "Infrastructure Drift Detected" \
              --body "Terraform detected drift from desired state"
          fi
```

## 💡 Best Practices

1. **Never bypass the PR process** - Even for "simple" changes
2. **Always include rollback plan** - Document how to undo
3. **Test in staging first** - Production follows staging
4. **Use workspaces/environments** - Separate state files
5. **Monitor after apply** - Watch metrics for 30 minutes
6. **Document decisions** - Why did we make this change?

## 🚨 Emergency Procedures

### If Apply Fails

```bash
# 1. Don't panic
# 2. Check partial application
terraform state list

# 3. Try targeted apply
terraform apply -target=aws_instance.broken

# 4. If still broken, rollback
terraform apply -replace="aws_instance.broken" tfplan-previous.binary

# 5. Notify team
@claude notify team of failed infrastructure change with rollback steps
```

### Break Glass Procedure

For true emergencies only:
```bash
# Requires 2 senior engineers
EMERGENCY=true terraform apply -auto-approve

# Must create incident report after
@claude create incident report for emergency infrastructure change
```

This workflow ensures infrastructure changes are safe, reviewed, and reversible!