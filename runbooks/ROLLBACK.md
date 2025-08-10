# Rollback Runbook

## When to Use
Execute this runbook when:
- Production deployment causes errors >5%
- Critical functionality is broken
- Performance degradation >50%
- Security vulnerability discovered

## Prerequisites
- [ ] Kubernetes access configured
- [ ] GitHub CLI authenticated
- [ ] Monitoring dashboard access
- [ ] PagerDuty access (if P0)

## Rollback Steps

### 1. Assess Situation (2 min)
```bash
# Check current status
kubectl get pods -n production
kubectl top pods -n production

# Check error rates
curl https://metrics.example.com/api/errors/5m

# Get current version
kubectl get deployment app -n production -o jsonpath='{.spec.template.spec.containers[0].image}'
```

### 2. Initiate Rollback (1 min)

#### Option A: Quick Rollback (Kubernetes)
```bash
# Rollback to previous version
kubectl rollout undo deployment/app -n production

# Monitor rollback
kubectl rollout status deployment/app -n production
```

#### Option B: Specific Version Rollback
```bash
# List deployment history
kubectl rollout history deployment/app -n production

# Rollback to specific revision
kubectl rollout undo deployment/app -n production --to-revision=42
```

#### Option C: Emergency Git Revert
```bash
# Revert last merge
git revert -m 1 HEAD
git push origin main

# This triggers CI/CD pipeline
```

### 3. Verify Rollback (3 min)
```bash
# Check pod status
kubectl get pods -n production -w

# Verify version
kubectl get deployment app -n production -o jsonpath='{.spec.template.spec.containers[0].image}'

# Run health checks
./scripts/health-check.sh production

# Check metrics
curl https://api.example.com/health
```

### 4. Monitor Stability (5 min)
```bash
# Watch error rates
watch -n 5 'curl -s https://metrics.example.com/api/errors/1m'

# Monitor logs
kubectl logs -f deployment/app -n production --tail=100

# Check user reports
# Monitor #support channel in Slack
```

### 5. Communicate Status

#### Internal Communication
```bash
# Update Slack
curl -X POST $SLACK_WEBHOOK -d '{
  "text": "🔄 Rollback completed. Service stable. Investigating root cause.",
  "channel": "#engineering"
}'

# Update incident
gh issue comment $INCIDENT_ID --body "Rollback completed at $(date)"
```

#### External Communication
```bash
# Update status page
curl -X POST https://status.example.com/api/incidents \
  -H "Authorization: Bearer $STATUS_TOKEN" \
  -d '{
    "status": "monitoring",
    "message": "Issue resolved. Monitoring for stability."
  }'
```

## Decision Matrix

| Metric | Threshold | Action | Severity |
|--------|-----------|---------|---------|
| Error Rate | >10% | Immediate rollback | P0 |
| Error Rate | 5-10% | Rollback within 5 min | P1 |
| Latency P95 | >5s | Immediate rollback | P1 |
| Memory Usage | >95% | Scale first, then rollback | P2 |
| CPU Usage | >90% | Scale first, then rollback | P2 |
| 500 Errors | >100/min | Immediate rollback | P0 |

## Post-Rollback Actions

### Immediate (within 30 min)
- [ ] Create incident report
- [ ] Disable problematic feature flags
- [ ] Block further deployments
- [ ] Schedule RCA meeting

### Short-term (within 24 hours)
- [ ] Complete RCA document
- [ ] Create fix PR
- [ ] Update tests
- [ ] Review deployment process

### Long-term (within 1 week)
- [ ] Update runbook based on learnings
- [ ] Add missing monitors
- [ ] Improve CI/CD checks
- [ ] Team retrospective

## Common Issues and Solutions

### Issue: Rollback Stuck
```bash
# Force rollback
kubectl delete pods -l app=api -n production
kubectl scale deployment/app --replicas=0 -n production
kubectl scale deployment/app --replicas=10 -n production
```

### Issue: Database Migration Incompatible
```bash
# Run reverse migration
kubectl exec -it deploy/migration-job -n production -- npm run migrate:down

# Or restore from backup
./scripts/restore-database.sh --point-in-time "30 minutes ago"
```

### Issue: Cache Inconsistency
```bash
# Clear all caches
kubectl exec -it deploy/redis -n production -- redis-cli FLUSHALL

# Restart app to rebuild cache
kubectl rollout restart deployment/app -n production
```

## Escalation Path

1. **On-call Engineer** - First responder
2. **Team Lead** - If rollback fails
3. **Platform Team** - Infrastructure issues
4. **CTO** - Customer-facing P0 >30min

## Tools and Links

- 📊 [Monitoring Dashboard](https://grafana.example.com)
- 🚨 [PagerDuty](https://example.pagerduty.com)
- 📝 [Incident Template](/.github/ISSUE_TEMPLATE/incident.yml)
- 💬 [Slack #incidents](https://slack.com/channels/incidents)
- 📈 [Status Page](https://status.example.com)

## Rollback Verification Checklist

- [ ] All pods running
- [ ] Health endpoints responding
- [ ] Error rate <0.1%
- [ ] P95 latency <500ms
- [ ] No customer complaints
- [ ] Monitoring alerts cleared
- [ ] Status page updated
- [ ] Team notified

---
*Last updated: 2024-01-08*
*Next review: 2024-02-08*