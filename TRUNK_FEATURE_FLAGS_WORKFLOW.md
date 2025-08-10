# Trunk-Based Development with Feature Flags

This document explains how to use trunk-based development with feature flags for safe, continuous deployment.

## 🎯 Overview

The trunk + feature flags workflow enables:
- **Continuous Integration**: Merge to main daily
- **Safe Deployment**: Features hidden behind flags
- **Progressive Rollout**: Gradual feature enablement
- **Automatic Cleanup**: Claude removes stale flags
- **Zero Downtime**: Instant rollback via flags

## 🌲 Trunk-Based Development

### Core Principles

1. **Single Branch**: Everyone works on `main` (trunk)
2. **Short-Lived Features**: Branches live <24 hours
3. **Feature Flags**: Incomplete features behind flags
4. **Continuous Deployment**: Deploy trunk multiple times daily
5. **No Long-Lived Branches**: No develop, staging, or release branches

### Workflow

```bash
# 1. Start feature with flag
/feature payment-redesign

# This creates:
# - Feature branch: feature/payment-redesign
# - Feature flag: payment-redesign (0% rollout)
# - Wires flag in code

# 2. Work on feature (can be incomplete)
claude "Implement new payment UI behind payment-redesign flag"

# 3. Merge to trunk quickly (same day)
/trunk-merge payment-redesign

# 4. Continue work on trunk
git checkout main
claude "Complete payment validation behind payment-redesign flag"

# 5. Progressive rollout
/flag-rollout payment-redesign 5    # 5% of users
/flag-rollout payment-redesign 25   # 25% of users
/flag-rollout payment-redesign 100  # All users

# 6. Automatic cleanup (after 30 days at 100%)
# Claude automatically creates cleanup PR
```

## 🚩 Feature Flag Lifecycle

### 1. Creation Phase

```bash
# Create flag with initial setup
/flag-create dark-mode 0

# Claude automatically:
# - Creates flag configuration
# - Wires flag checks in code
# - Creates tests for both paths
# - Sets up monitoring
# - Schedules cleanup
```

Generated flag configuration:
```yaml
# config/flags.yaml
flags:
  dark-mode:
    description: "Dark theme support"
    created: 2024-01-08
    owner: "@team-frontend"
    rollout:
      type: "percentage"
      value: 0
      strategy: "gradual"
    sunset_date: 2024-03-08
    cleanup_after_days: 30
```

Generated code:
```typescript
// src/components/Theme.tsx
import { useFeatureFlag } from '@/lib/feature-flags';

export function ThemeProvider({ children }) {
  const isDarkModeEnabled = useFeatureFlag('dark-mode');
  
  if (isDarkModeEnabled) {
    // New dark mode implementation
    return <DarkThemeProvider>{children}</DarkThemeProvider>;
  }
  
  // Existing light theme (preserved)
  return <LightThemeProvider>{children}</LightThemeProvider>;
}
```

### 2. Rollout Phase

```bash
# Gradual rollout with monitoring
/flag-rollout dark-mode 5   # Start with 5%
# Monitor for 1 hour...

/flag-rollout dark-mode 25  # Increase to 25%
# Monitor for 4 hours...

/flag-rollout dark-mode 50  # Half of users
# Monitor for 1 day...

/flag-rollout dark-mode 100 # Full rollout
```

Each rollout change:
- Updates configuration
- Triggers monitoring
- Sends notifications
- Checks error rates
- Can auto-rollback

### 3. Monitoring Phase

```bash
# Monitor flag performance
/flag-monitor dark-mode 300

# Output:
# 📊 Monitoring flag dark-mode for 300s...
# [0s] Error rate: 0.01%, Latency: 95ms
# [60s] Error rate: 0.02%, Latency: 102ms
# [120s] Error rate: 0.01%, Latency: 98ms
# ✅ Flag is stable
```

Automatic monitoring includes:
- Error rate tracking
- Performance comparison
- User engagement metrics
- Automatic rollback on issues

### 4. Cleanup Phase

After 30 days at 100% rollout, Claude automatically:

1. **Detects stale flag**
```bash
# Runs daily via cron/GitHub Actions
/flag-cleanup

# Finds:
# - dark-mode: 100% for 35 days ✓
# - payment-redesign: 100% for 28 days (wait 2 more days)
# - experimental-ai: 15% for 60 days (stuck, needs review)
```

2. **Creates cleanup PR**
```markdown
PR #123: cleanup: Remove dark-mode feature flag

## Removing Feature Flag: `dark-mode`

### Reason
- Flag has been at 100% rollout for 35 days
- No issues reported
- All metrics positive

### Changes
- Removed flag checks from 12 locations
- Using dark mode code path (winner)
- Removed flag configuration
- Removed old light theme code

### Testing
- All tests passing
- No functional changes (flag was at 100%)
```

3. **Waits for approval**
```
🔔 Approval Required: Feature Flag Cleanup

Flag: dark-mode
Status: Ready for removal
PR: #123

[Approve] [Reject] [Postpone]
```

## 🔔 Notification System

### Approval Notifications

When Claude needs approval, you get notified via:

#### Slack
```
🔔 Approval Required: Flag Rollout

Flag: payment-redesign
Current: 25%
Proposed: 50%
Metrics: ✅ Stable

[Approve] [Reject] [View Dashboard]
```

#### GitHub Issue
```markdown
## 🔔 Approval Required: Flag Cleanup

**Flag**: checkout-v2
**Action**: Remove flag and old code
**PR**: #456

### Checklist
- [ ] Review PR changes
- [ ] Confirm metrics are stable
- [ ] Approve removal

### Links
- [View PR](#456)
- [Flag Dashboard](https://flags.example.com/checkout-v2)
```

#### Email/PagerDuty
For critical production flags

### Configuration

```json
// .claude/notifications.json
{
  "slack": {
    "enabled": true,
    "webhook_url": "$SLACK_WEBHOOK",
    "channels": {
      "flag_rollout": "#deployments",
      "flag_cleanup": "#engineering",
      "flag_issues": "#oncall"
    }
  },
  "github": {
    "enabled": true,
    "create_issue": true,
    "assign_to": ["team-lead", "flag-owner"]
  },
  "email": {
    "enabled": false,
    "recipients": ["team@example.com"]
  }
}
```

## 📊 Commands Reference

### Flag Management
```bash
/flag-create <name> [rollout%]    # Create new flag
/flag-rollout <name> <percentage> # Change rollout
/flag-status                      # Show all flags
/flag-cleanup                     # Remove stale flags
/flag-monitor <name> [duration]   # Monitor performance
/flag-wire <name> <file>         # Add flag to code
/flag-test <name>                # Test both paths
```

### Trunk-Based Flow
```bash
/feature <name>                   # Start feature with flag
/trunk-merge <feature>           # Merge to trunk with flag
```

## 🎯 Best Practices

### 1. Flag Naming
```
Format: <team>-<feature>-<version>
Good: payments-checkout-v2
Bad: flag1, test, new-thing
```

### 2. Rollout Strategy
```
Canary: 1% → 5% → 25% → 50% → 100%
Time: 1hr → 4hr → 1day → 2day → done
```

### 3. Flag Hygiene
- Maximum age: 90 days
- Cleanup after: 30 days at 100%
- Required: sunset date
- Monitor: performance impact

### 4. Testing
```bash
# Always test both paths
/flag-test payment-redesign

# Runs:
# - Tests with flag OFF
# - Tests with flag ON
# - Performance comparison
# - Error rate check
```

## 🔄 Complete Example

### Day 1: Start Feature
```bash
# Morning: Start feature
/feature ai-suggestions

# Afternoon: Basic implementation
claude "Add AI suggestions to search with ai-suggestions flag"

# Evening: Merge to trunk (incomplete but safe)
/trunk-merge ai-suggestions
```

### Day 2-3: Iterate on Trunk
```bash
# Continue development
git checkout main
claude "Improve AI suggestion ranking"
git commit -m "feat: Better AI ranking (flag: ai-suggestions)"
git push
```

### Day 4: Start Rollout
```bash
# Begin canary rollout
/flag-rollout ai-suggestions 1
/flag-monitor ai-suggestions

# Looks good, increase
/flag-rollout ai-suggestions 10
```

### Week 2: Full Rollout
```bash
/flag-rollout ai-suggestions 100
```

### Day 45: Automatic Cleanup
```
🔔 Claude: Flag ai-suggestions ready for cleanup
Created PR #789: Remove ai-suggestions flag

[Review PR]
```

## 🚨 Rollback Scenarios

### Instant Rollback
```bash
# Problem detected!
/flag-rollout payment-v2 0  # Instant disable

# Or partial rollback
/flag-rollout payment-v2 10  # Reduce to 10%
```

### Automatic Rollback
Configured in flag:
```yaml
flags:
  payment-v2:
    rollback_conditions:
      - error_rate > 0.05  # 5% errors
      - latency_p95 > 2000  # 2s latency
      - conversion_drop > 0.1  # 10% drop
```

## 📈 Metrics & Monitoring

### Flag Dashboard
```
Flag: checkout-v2
Status: 75% rollout
Age: 12 days
Metrics:
  - Error Rate: 0.02% ✅
  - Latency: +12ms ⚠️
  - Conversion: +2.3% ✅
  - Usage: 145k/day
Next: Increase to 100%
```

### Automated Reports
Daily flag report:
```
Active Flags: 8
Ready for Cleanup: 2
Stuck Flags: 1 (needs attention)
Total Cleaned This Month: 5
```

## 🔒 Safety Features

1. **Gradual Rollout**: Never 0→100
2. **Automatic Monitoring**: After each change
3. **Required Tests**: Both paths must pass
4. **Sunset Dates**: Prevent eternal flags
5. **Approval Gates**: For production changes
6. **Instant Rollback**: One command
7. **Automatic Cleanup**: Prevents tech debt

Remember: **Ship early, ship often, ship safely!**