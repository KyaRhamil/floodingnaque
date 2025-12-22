# Floodingnaque Security Incident Response Runbook

## Document Information
- **Version:** 1.0
- **Last Updated:** 2024-12-22
- **Classification:** Internal Use Only
- **Owner:** Security Team

---

## 1. Overview

This runbook provides step-by-step procedures for responding to security incidents in the Floodingnaque application. All team members with system access should be familiar with these procedures.

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P1 - Critical** | Active breach, data exfiltration, system compromise | Immediate (< 15 min) | Active attack, ransomware, database breach |
| **P2 - High** | Vulnerability being exploited, potential data exposure | < 1 hour | SQL injection attempt, authentication bypass |
| **P3 - Medium** | Security weakness identified, no active exploitation | < 24 hours | Outdated dependency, misconfiguration |
| **P4 - Low** | Minor security improvement needed | < 1 week | Information disclosure, missing headers |

---

## 2. Initial Response Checklist

### Immediate Actions (First 15 Minutes)

- [ ] **Assess the situation** - Determine what happened
- [ ] **Notify the incident commander** - Escalate to security lead
- [ ] **Preserve evidence** - Do NOT delete logs or modify systems
- [ ] **Document timeline** - Start an incident log with timestamps
- [ ] **Determine severity** - Use the severity matrix above

### Communication Chain

1. **First Responder** → **Security Lead** → **CTO/Engineering Lead**
2. For P1 incidents, immediately notify all stakeholders
3. Use secure communication channels (not the compromised system)

---

## 3. Incident Response Procedures

### 3.1 Unauthorized Access / Authentication Breach

**Indicators:**
- Unusual login patterns or locations
- Multiple failed login attempts followed by success
- API key abuse or unexpected API usage spikes

**Response Steps:**

```bash
# 1. Check recent authentication logs
docker logs floodingnaque-api-prod --since 1h | grep -i "auth\|login\|401\|403"

# 2. Check rate limiting logs
docker logs floodingnaque-api-prod --since 1h | grep -i "rate limit\|429"

# 3. Review active sessions in Redis
redis-cli -h <REDIS_HOST> KEYS "session:*"

# 4. Immediately rotate compromised credentials
# Generate new API keys
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 5. Force all sessions to re-authenticate
redis-cli -h <REDIS_HOST> FLUSHDB  # CAUTION: Clears all Redis data
```

**Containment:**
1. Revoke compromised API keys immediately
2. Block suspicious IP addresses in Nginx
3. Enable enhanced logging
4. Consider temporary rate limit reduction

### 3.2 Data Breach / Data Exfiltration

**Indicators:**
- Large data exports or unusual query patterns
- Unauthorized access to sensitive endpoints
- Database queries for bulk user data

**Response Steps:**

```bash
# 1. Check database query logs (if enabled)
docker exec floodingnaque-api-prod cat /app/logs/floodingnaque.log | grep -i "SELECT\|export\|data"

# 2. Review export endpoint usage
docker logs floodingnaque-api-prod | grep "/export\|/api/v1/data"

# 3. Check Supabase audit logs
# Access Supabase Dashboard > Logs > API

# 4. Identify affected data scope
# Query database for access patterns
```

**Containment:**
1. Temporarily disable export endpoints if needed
2. Add additional authentication requirements
3. Enable database audit logging
4. Notify affected users if personal data involved

### 3.3 DDoS Attack / Service Disruption

**Indicators:**
- Sudden traffic spike
- Increased latency or timeouts
- 502/503 errors from Nginx
- Container resource exhaustion

**Response Steps:**

```bash
# 1. Check current traffic patterns
docker stats --no-stream

# 2. Review Nginx access logs for patterns
docker exec floodingnaque-nginx-prod tail -1000 /var/log/nginx/floodingnaque_access.log | \
  awk '{print $1}' | sort | uniq -c | sort -rn | head -20

# 3. Check rate limiting effectiveness
docker logs floodingnaque-api-prod | grep "429\|rate limit"

# 4. Enable aggressive rate limiting
# Edit nginx config to reduce limits temporarily
```

**Containment:**
1. Enable Cloudflare "Under Attack" mode if available
2. Block attacking IP ranges in firewall
3. Scale up containers if needed
4. Consider temporary geographic blocking

### 3.4 Malware / Container Compromise

**Indicators:**
- Unexpected processes running in containers
- Modified system files
- Outbound connections to unknown IPs
- Cryptocurrency mining activity

**Response Steps:**

```bash
# 1. DO NOT STOP THE CONTAINER - preserve evidence first

# 2. Capture container state
docker commit floodingnaque-api-prod evidence-$(date +%Y%m%d-%H%M%S)

# 3. Check running processes
docker exec floodingnaque-api-prod ps aux

# 4. Check network connections
docker exec floodingnaque-api-prod netstat -tulpn

# 5. Check file modifications
docker exec floodingnaque-api-prod find /app -mmin -60 -type f

# 6. After evidence collection, isolate the container
docker network disconnect floodingnaque-production floodingnaque-api-prod
```

**Containment:**
1. Isolate affected container from network
2. Spin up fresh container from known-good image
3. Rotate all secrets and credentials
4. Scan other containers for similar compromise

### 3.5 Dependency Vulnerability (CVE)

**Indicators:**
- CVE announced for a dependency
- pip-audit or Trivy scan alerts
- Security advisory from vendor

**Response Steps:**

```bash
# 1. Run vulnerability scan
cd backend
pip-audit -r requirements.txt --strict

# 2. Check if vulnerability is exploitable in our context
# Review the CVE details and affected code paths

# 3. Test the patch in staging
pip install --upgrade <affected-package>
python -m pytest tests/

# 4. Deploy updated dependencies
docker build -t floodingnaque-api:patched .
docker-compose -f docker-compose-production.yml up -d --build
```

**Risk Assessment Questions:**
1. Is the vulnerable code path used in our application?
2. Is the vulnerability remotely exploitable?
3. Is there a known exploit in the wild?
4. What data could be accessed if exploited?

---

## 4. Evidence Preservation

### What to Preserve
- [ ] Application logs (`/app/logs/`)
- [ ] Container logs (`docker logs`)
- [ ] Nginx access and error logs
- [ ] Database query logs
- [ ] Network traffic captures
- [ ] System state (running processes, connections)

### How to Preserve

```bash
# Create evidence directory
mkdir -p /evidence/incident-$(date +%Y%m%d)
cd /evidence/incident-$(date +%Y%m%d)

# Capture container logs
docker logs floodingnaque-api-prod > api-logs.txt 2>&1
docker logs floodingnaque-nginx-prod > nginx-logs.txt 2>&1

# Capture container state
docker inspect floodingnaque-api-prod > container-state.json

# Capture network state
docker exec floodingnaque-api-prod netstat -tulpn > network-state.txt

# Create forensic image of container
docker commit floodingnaque-api-prod evidence:$(date +%Y%m%d-%H%M%S)

# Hash all evidence files
sha256sum * > evidence-hashes.txt
```

---

## 5. Communication Templates

### Internal Notification (Slack/Email)

```
🚨 SECURITY INCIDENT - [SEVERITY]

What: [Brief description]
When: [Timestamp]
Status: [Investigating/Contained/Resolved]
Impact: [Affected systems/users]
Action Required: [What team members should do]

Incident Commander: [Name]
Next Update: [Time]
```

### External Notification (If Required)

```
Subject: Security Notification - [Date]

Dear [Customer/User],

We are writing to inform you of a security incident that may have 
affected your account/data on [Date].

What Happened:
[Description of incident]

What Information Was Involved:
[Types of data potentially affected]

What We Are Doing:
[Actions taken to address the incident]

What You Can Do:
[Recommended actions for the user]

For More Information:
[Contact details]

Sincerely,
Floodingnaque Security Team
```

---

## 6. Recovery Procedures

### Post-Incident Checklist

- [ ] Confirm threat is neutralized
- [ ] Restore systems from clean state
- [ ] Rotate all potentially compromised credentials
- [ ] Update security controls to prevent recurrence
- [ ] Conduct post-incident review
- [ ] Update this runbook with lessons learned
- [ ] Communicate resolution to stakeholders

### Credential Rotation

```bash
# Generate new secrets
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_hex(32))"
python -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"

# Update .env.production with new values
# Restart all services
docker-compose -f docker-compose-production.yml down
docker-compose -f docker-compose-production.yml up -d
```

---

## 7. Useful Commands Reference

### Log Analysis

```bash
# Search for specific patterns in logs
docker logs floodingnaque-api-prod 2>&1 | grep -E "ERROR|CRITICAL|401|403|500"

# Count requests by IP
cat access.log | awk '{print $1}' | sort | uniq -c | sort -rn

# Find suspicious user agents
grep -i "sqlmap\|nikto\|scanner\|bot" access.log
```

### Database Queries

```sql
-- Check recent failed logins (if logging enabled)
SELECT * FROM api_requests 
WHERE endpoint LIKE '%login%' 
AND status_code = 401 
ORDER BY timestamp DESC LIMIT 100;

-- Check high-volume API users
SELECT api_key_hash, COUNT(*) as requests 
FROM api_requests 
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY api_key_hash 
ORDER BY requests DESC;
```

### Blocking IPs

```bash
# Block IP in Nginx
echo "deny 192.168.1.100;" >> /etc/nginx/conf.d/blocklist.conf
nginx -s reload

# Block IP range with iptables (if not using container)
iptables -A INPUT -s 192.168.1.0/24 -j DROP
```

---

## 8. Contacts

| Role | Name | Contact | Backup |
|------|------|---------|--------|
| Security Lead | TBD | TBD | TBD |
| Engineering Lead | TBD | TBD | TBD |
| DevOps Lead | TBD | TBD | TBD |
| Legal/Compliance | TBD | TBD | TBD |

### External Contacts

- **Supabase Support:** support@supabase.io
- **Redis Cloud Support:** support@redis.com
- **Cloudflare Security:** security@cloudflare.com
- **Local CERT:** (Add your country's CERT contact)

---

## 9. Post-Incident Review Template

### Incident Summary
- **Incident ID:** INC-YYYY-MM-DD-XXX
- **Severity:** P1/P2/P3/P4
- **Duration:** Start to Resolution
- **Impact:** Affected users/systems

### Timeline
| Time | Event | Action Taken |
|------|-------|--------------|
| HH:MM | Event description | Action taken |

### Root Cause Analysis
- What was the root cause?
- Why did existing controls fail?
- What made detection possible?

### Lessons Learned
1. What went well?
2. What could be improved?
3. What actions will prevent recurrence?

### Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Description | Name | Date | Open/Complete |

---

**Remember:** Security is everyone's responsibility. When in doubt, escalate.
