# Production Deployment Guide

## Pre-Deployment Checklist

### 1. Infrastructure Requirements

**Minimum Specifications:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 10 Mbps uplink

**Recommended Specifications:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 100 Mbps uplink

**For Edge Nodes:**
- Jetson Orin Nano (8GB)
- Or any Linux device with Python 3.11+

### 2. Security Configuration

**Environment Variables (Production):**
```bash
# Generate strong secret key
SECRET_KEY=$(openssl rand -hex 32)

# Strong database password
POSTGRES_PASSWORD=$(openssl rand -base64 24)

# Update .env file
cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://reasoner:${POSTGRES_PASSWORD}@postgres:5432/reasoner_db
SECRET_KEY=${SECRET_KEY}
DEBUG=false
ENABLE_PROMETHEUS_METRICS=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_PERIOD=60
EOF
```

**SSL/TLS Setup:**
```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key \
  -out ssl/certificate.crt

# Update nginx configuration to use SSL
# (See production nginx.conf)
```

### 3. Database Backup Strategy

**Automated Backups:**
```bash
# Add to crontab
0 2 * * * docker exec postgres pg_dump -U reasoner reasoner_db > /backups/reasoner_$(date +\%Y\%m\%d).sql

# Retention: Keep 7 days
find /backups -name "reasoner_*.sql" -mtime +7 -delete
```

**Restore from Backup:**
```bash
docker exec -i postgres psql -U reasoner reasoner_db < backup_file.sql
```

## Step-by-Step Deployment

### Step 1: Server Setup

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get install -y docker-compose

# Create user for deployment
useradd -m -s /bin/bash reasoner
usermod -aG docker reasoner
```

### Step 2: Application Deployment

```bash
# As reasoner user
su - reasoner

# Clone/copy application
cd /opt
# Extract reasoner-platform-complete.zip here

# Configure environment
cd reasoner-platform
cp .env.example .env
nano .env  # Update with production values

# Start services
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f
```

### Step 3: Initialize Database

```bash
# Load initial formulas
docker-compose exec backend python -m app.core.init_db

# Verify database
docker-compose exec postgres psql -U reasoner reasoner_db -c "\dt"
```

### Step 4: Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check API
curl http://localhost:8000/api/v1/formulas?limit=10

# Check dashboard
curl http://localhost:3000
```

### Step 5: Configure Reverse Proxy

**Nginx Configuration (External Server):**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/certificate.crt;
    ssl_certificate_key /etc/ssl/private/private.key;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
}

# Rate limit zone
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
```

### Step 6: Monitoring Setup

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'reasoner-backend'
    static_configs:
      - targets: ['localhost:8000']
```

**Docker Compose Addition:**
```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

## High Availability Setup

### Load Balancer Configuration

**HAProxy:**
```
frontend reasoner_frontend
    bind *:80
    mode http
    default_backend reasoner_backend

backend reasoner_backend
    balance roundrobin
    option httpchk GET /health
    server backend1 10.0.1.10:8000 check
    server backend2 10.0.1.11:8000 check
    server backend3 10.0.1.12:8000 check
```

### Database Replication

**PostgreSQL Streaming Replication:**
```sql
-- On primary
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'strong_password';

-- postgresql.conf
wal_level = replica
max_wal_senders = 3
```

**On replica:**
```bash
pg_basebackup -h primary_host -D /var/lib/postgresql/data -U replicator -P
```

## Scaling Strategies

### Horizontal Scaling (Multiple Backends)

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3
    environment:
      WORKERS: 2  # Per container
```

### Vertical Scaling (More Resources)

```yaml
services:
  backend:
    environment:
      WORKERS: 8
      MAX_PARALLEL_WORKERS: 16
      FORMULA_CACHE_SIZE: 5000
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
```

### Edge Deployment

**On each edge node:**
```bash
# Install Python and dependencies
apt-get install python3.11 python3-pip

# Copy edge processor
cd /opt/reasoner-edge
pip install -r requirements.txt

# Run as service
cat > /etc/systemd/system/reasoner-edge.service << EOF
[Unit]
Description=Reasoner Edge Processor
After=network.target

[Service]
Type=simple
User=reasoner
WorkingDirectory=/opt/reasoner-edge
ExecStart=/usr/bin/python3 edge_processor.py edge_node_$(hostname) https://backend.company.com
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable reasoner-edge
systemctl start reasoner-edge
```

## Performance Optimization

### Database Optimization

```sql
-- Add indexes
CREATE INDEX CONCURRENTLY idx_formulas_confidence_domain 
ON formulas(confidence_score, domain);

CREATE INDEX CONCURRENTLY idx_executions_timestamp 
ON formula_executions(execution_timestamp DESC);

-- Analyze tables
ANALYZE formulas;
ANALYZE formula_executions;
```

### Caching Strategy

**Redis for API Caching:**
```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

**In application:**
```python
# Add to requirements.txt
redis==5.0.1

# Use for frequent queries
from redis import Redis
cache = Redis(host='redis', port=6379, decode_responses=True)
```

## Disaster Recovery

### Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker exec postgres pg_dump -U reasoner reasoner_db | gzip > "${BACKUP_DIR}/db_${DATE}.sql.gz"

# Application data
tar -czf "${BACKUP_DIR}/data_${DATE}.tar.gz" /opt/reasoner-platform/data

# Upload to S3 (optional)
# aws s3 cp "${BACKUP_DIR}/" s3://your-bucket/backups/ --recursive

# Cleanup old backups (>30 days)
find "${BACKUP_DIR}" -name "*.gz" -mtime +30 -delete
```

### Recovery Procedures

**Full System Recovery:**
```bash
# 1. Fresh installation
docker-compose down -v

# 2. Restore database
gunzip < db_backup.sql.gz | docker exec -i postgres psql -U reasoner reasoner_db

# 3. Restore data
tar -xzf data_backup.tar.gz -C /opt/reasoner-platform/

# 4. Restart services
docker-compose up -d
```

## Security Hardening

### Firewall Rules

```bash
# UFW configuration
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

### Fail2Ban Configuration

```ini
# /etc/fail2ban/jail.local
[reasoner-api]
enabled = true
port = http,https
filter = reasoner-api
logpath = /var/log/nginx/access.log
maxretry = 10
bantime = 3600
```

### Regular Updates

```bash
# Automated security updates
apt-get install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades
```

## Monitoring & Alerts

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:8000/health"
SLACK_WEBHOOK="your-webhook-url"

response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $response -ne 200 ]; then
    curl -X POST $SLACK_WEBHOOK -H 'Content-Type: application/json' \
      -d '{"text":"⚠️ Reasoner API is down! Status: '$response'"}'
fi
```

### Log Aggregation

**Using Loki:**
```yaml
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
  
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml
```

## Troubleshooting

### Common Issues

**Issue: High CPU usage**
```bash
# Check which container
docker stats

# Reduce workers
# Edit docker-compose.yml
environment:
  WORKERS: 2
  MAX_PARALLEL_WORKERS: 4
```

**Issue: Database connection pool exhausted**
```bash
# Increase pool size
environment:
  DB_POOL_SIZE: 40
  DB_MAX_OVERFLOW: 20
```

**Issue: Slow formula execution**
```bash
# Enable caching
environment:
  FORMULA_CACHE_SIZE: 2000

# Check formula complexity
docker-compose exec backend python -c "
from app.services.reasoner import reasoner_engine
# Profile slow formulas
"
```

## Maintenance Windows

**Recommended Schedule:**
- **Weekly:** Database optimization (VACUUM, ANALYZE)
- **Monthly:** Security updates
- **Quarterly:** Full backup testing
- **Annually:** Disaster recovery drill

**Maintenance Commands:**
```bash
# Put in maintenance mode
docker-compose scale backend=0

# Perform maintenance
docker-compose exec postgres psql -U reasoner -c "VACUUM ANALYZE;"

# Restore service
docker-compose scale backend=3
```

## Compliance & Audit

### Audit Logging

```python
# Enable detailed logging
environment:
  LOG_LEVEL: INFO
  LOG_FORMAT: json
  
# Configure log retention
volumes:
  - ./logs:/app/logs
```

### Data Privacy

- Ensure GDPR compliance for EU users
- Implement data retention policies
- Add user consent mechanisms
- Enable data export/deletion

## Support Contacts

- **Technical Issues:** Check logs first (`docker-compose logs`)
- **Security Issues:** Follow disclosure policy
- **Performance Issues:** Review monitoring dashboards

## Conclusion

This deployment guide covers production deployment, scaling, security, and maintenance. Follow these procedures for a robust, secure installation.

For specific environment needs, adjust configurations accordingly. Always test in staging before production deployment.
