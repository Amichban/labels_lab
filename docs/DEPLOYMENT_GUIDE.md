# Label Computation System Deployment Guide

> Comprehensive guide for deploying the Label Computation System in production environments using Docker and Kubernetes.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Production Configuration](#production-configuration)
5. [Security Considerations](#security-considerations)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & Performance](#scaling--performance)
8. [Backup & Recovery](#backup--recovery)

## Deployment Overview

The Label Computation System consists of several components that must be deployed and configured together:

### Core Components
- **Labels API**: FastAPI application serving REST endpoints
- **ClickHouse**: Time-series database for market data and computed labels
- **Redis**: High-performance cache for hot data and active levels
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Metrics visualization and dashboards

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
│                   (NGINX/ALB)                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
           ┌──────────┼──────────┐
           │          │          │
      ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
      │API Pod │ │API Pod │ │API Pod │  (Auto-scaling)
      └────┬───┘ └────┬───┘ └────┬───┘
           │          │          │
           └──────────┼──────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼───┐    ┌────▼────┐   ┌────▼────┐
   │ Redis  │    │ClickHouse│   │Prometheus│
   │Cluster │    │ Cluster  │   │ Server  │
   └────────┘    └─────────┘   └─────────┘
```

### Deployment Options

1. **Docker Compose** (Development & Small Production)
2. **Docker Swarm** (Medium Scale Production)
3. **Kubernetes** (Enterprise & High Scale)
4. **Cloud Managed Services** (AWS ECS/EKS, GCP GKE, Azure AKS)

## Docker Deployment

### Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- Minimum 8GB RAM, 4 CPU cores
- 100GB+ SSD storage for data persistence
- Network access to market data sources

### Quick Start (Development)

1. **Clone and prepare environment**:
```bash
git clone <repository-url>
cd label-computation-system

# Copy environment template
cp .env.example .env
```

2. **Configure environment variables** (`.env`):
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
ENVIRONMENT=production

# Database Configuration
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_secure_clickhouse_password
CLICKHOUSE_DATABASE=quantx

# Cache Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_redis_password
REDIS_DB=0
CACHE_TTL_SECONDS=3600

# Performance Tuning
BATCH_CHUNK_SIZE=10000
PARALLEL_WORKERS=8
ENABLE_VALIDATION=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

3. **Deploy with Docker Compose**:
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f labels-api

# Initialize database schema
docker-compose exec labels-api python scripts/create_tables.py
```

4. **Verify deployment**:
```bash
# Health check
curl http://localhost:8000/v1/health

# Compute test label
curl -X POST http://localhost:8000/v1/labels/compute \
  -H "Content-Type: application/json" \
  -d @docs/api/examples/requests/compute-labels.json
```

### Production Docker Compose

For production deployment, create a separate `docker-compose.prod.yml`:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  labels-api:
    image: label-computation-system:1.0.0
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    environment:
      - API_WORKERS=8
      - DEBUG=false
      - LOG_LEVEL=INFO
      - ENABLE_PROFILING=false
    volumes:
      - /var/log/labels-api:/app/logs
    networks:
      - labels-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health/live"]
      interval: 15s
      timeout: 5s
      retries: 5

  clickhouse:
    image: clickhouse/clickhouse-server:23.8
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
    environment:
      - CLICKHOUSE_DB=quantx
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./config/clickhouse/production:/etc/clickhouse-server/config.d
    ulimits:
      nofile:
        soft: 262144
        hard: 262144

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 3gb
      --maxmemory-policy allkeys-lru
      --save 300 100
      --tcp-keepalive 60
      --timeout 300
    volumes:
      - redis_data:/data

  # Reverse proxy with SSL termination
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/ssl:/etc/nginx/ssl
    depends_on:
      - labels-api

networks:
  labels-network:
    driver: overlay
    attachable: true
```

Deploy with:
```bash
docker stack deploy -c docker-compose.prod.yml labels-stack
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.24+ (EKS, GKE, AKS, or on-premises)
- kubectl configured for cluster access
- Helm 3.0+ for package management
- Persistent volume support for data storage

### Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: label-computation
  labels:
    name: label-computation
    environment: production
```

### ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: labels-config
  namespace: label-computation
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "8"
  ENVIRONMENT: "production"
  CLICKHOUSE_HOST: "clickhouse-service"
  CLICKHOUSE_PORT: "9000"
  CLICKHOUSE_DATABASE: "quantx"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  REDIS_DB: "0"
  CACHE_TTL_SECONDS: "3600"
  BATCH_CHUNK_SIZE: "10000"
  PARALLEL_WORKERS: "8"
  PROMETHEUS_ENABLED: "true"
  LOG_LEVEL: "INFO"

---
apiVersion: v1
kind: Secret
metadata:
  name: labels-secrets
  namespace: label-computation
type: Opaque
data:
  clickhouse-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  jwt-secret: <base64-encoded-jwt-secret>
```

### ClickHouse Deployment

```yaml
# clickhouse-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: clickhouse
  namespace: label-computation
spec:
  serviceName: clickhouse-headless
  replicas: 3
  selector:
    matchLabels:
      app: clickhouse
  template:
    metadata:
      labels:
        app: clickhouse
    spec:
      containers:
      - name: clickhouse
        image: clickhouse/clickhouse-server:23.8
        ports:
        - containerPort: 9000
          name: native
        - containerPort: 8123
          name: http
        env:
        - name: CLICKHOUSE_DB
          value: "quantx"
        - name: CLICKHOUSE_USER
          value: "default"
        - name: CLICKHOUSE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: labels-secrets
              key: clickhouse-password
        volumeMounts:
        - name: clickhouse-data
          mountPath: /var/lib/clickhouse
        - name: clickhouse-config
          mountPath: /etc/clickhouse-server/config.d
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /ping
            port: 8123
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ping
            port: 8123
          initialDelaySeconds: 15
          periodSeconds: 10
      volumes:
      - name: clickhouse-config
        configMap:
          name: clickhouse-config
  volumeClaimTemplates:
  - metadata:
      name: clickhouse-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 500Gi

---
apiVersion: v1
kind: Service
metadata:
  name: clickhouse-service
  namespace: label-computation
spec:
  selector:
    app: clickhouse
  ports:
  - name: native
    port: 9000
    targetPort: 9000
  - name: http
    port: 8123
    targetPort: 8123
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: clickhouse-headless
  namespace: label-computation
spec:
  clusterIP: None
  selector:
    app: clickhouse
  ports:
  - name: native
    port: 9000
  - name: http
    port: 8123
```

### Redis Deployment

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: label-computation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: labels-secrets
              key: redis-password
        command:
        - redis-server
        - --requirepass
        - $(REDIS_PASSWORD)
        - --maxmemory
        - 3gb
        - --maxmemory-policy
        - allkeys-lru
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - --raw
            - incr
            - ping
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: label-computation
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: "fast-ssd"
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: label-computation
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

### Labels API Deployment

```yaml
# labels-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: labels-api
  namespace: label-computation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: labels-api
  template:
    metadata:
      labels:
        app: labels-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/v1/metrics"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: labels-api
        image: label-computation-system:1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: labels-config
        env:
        - name: CLICKHOUSE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: labels-secrets
              key: clickhouse-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: labels-secrets
              key: redis-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: labels-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /v1/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /v1/health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: labels-api-service
  namespace: label-computation
spec:
  selector:
    app: labels-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: labels-api-ingress
  namespace: label-computation
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.labelcompute.com
    secretName: labels-api-tls
  rules:
  - host: api.labelcompute.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: labels-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: labels-api-hpa
  namespace: label-computation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: labels-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Deployment Commands

```bash
# Apply all Kubernetes manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f clickhouse-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f labels-api-deployment.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n label-computation
kubectl get services -n label-computation
kubectl get ingress -n label-computation

# Initialize database schema
kubectl exec -n label-computation deploy/labels-api -- python scripts/create_tables.py

# View logs
kubectl logs -n label-computation -f deploy/labels-api
```

## Production Configuration

### Environment-Specific Configuration

Create separate configuration files for different environments:

```bash
# Production environment (.env.prod)
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
API_WORKERS=8
ENABLE_PROFILING=false

# Database connection pooling
CLICKHOUSE_MAX_CONNECTIONS=50
CLICKHOUSE_QUERY_TIMEOUT=30

# Cache optimization
REDIS_MAX_CONNECTIONS=100
CACHE_TTL_SECONDS=3600
CACHE_PRELOAD_ENABLED=true

# Performance tuning
BATCH_CHUNK_SIZE=25000
PARALLEL_WORKERS=16
MAX_CONCURRENT_REQUESTS=1000

# Security
CORS_ORIGINS=https://dashboard.labelcompute.com
RATE_LIMIT_PER_MINUTE=1000
JWT_EXPIRATION_HOURS=24

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_EXPORT_INTERVAL=15
HEALTH_CHECK_INTERVAL=30
```

### Database Optimization

**ClickHouse Production Configuration** (`config/clickhouse/production/config.xml`):

```xml
<?xml version="1.0"?>
<clickhouse>
    <!-- Performance settings -->
    <max_connections>4000</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>1000</max_concurrent_queries>
    <max_server_memory_usage_to_ram_ratio>0.8</max_server_memory_usage_to_ram_ratio>
    
    <!-- Memory settings -->
    <max_memory_usage>8000000000</max_memory_usage>
    <max_bytes_before_external_group_by>4000000000</max_bytes_before_external_group_by>
    
    <!-- Query optimization -->
    <max_execution_time>300</max_execution_time>
    <max_query_size>1048576</max_query_size>
    
    <!-- Compression -->
    <compression>
        <case>
            <method>lz4</method>
        </case>
    </compression>
    
    <!-- Distributed settings -->
    <distributed_aggregation_memory_efficient>1</distributed_aggregation_memory_efficient>
    
    <!-- Logging -->
    <logger>
        <level>warning</level>
        <log>/var/log/clickhouse-server/clickhouse-server.log</log>
        <errorlog>/var/log/clickhouse-server/clickhouse-server.err.log</errorlog>
        <size>100M</size>
        <count>5</count>
    </logger>
</clickhouse>
```

### Load Balancing with NGINX

**NGINX Configuration** (`config/nginx/nginx.conf`):

```nginx
upstream labels_api {
    least_conn;
    server labels-api-1:8000 max_fails=3 fail_timeout=30s;
    server labels-api-2:8000 max_fails=3 fail_timeout=30s;
    server labels-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.labelcompute.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json text/plain;

    location / {
        proxy_pass http://labels_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health check
        proxy_next_upstream error timeout http_500 http_502 http_503;
    }

    location /health {
        access_log off;
        proxy_pass http://labels_api;
    }
}
```

## Security Considerations

### Authentication & Authorization

1. **JWT Token Configuration**:
```python
# JWT settings in production
JWT_SECRET = "your-256-bit-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
JWT_REFRESH_ENABLED = True
```

2. **API Key Management**:
```python
# Environment-specific API keys
API_KEYS = {
    "production": ["prod_key_1", "prod_key_2"],
    "staging": ["staging_key_1"],
}
```

### Network Security

1. **VPC/Network Isolation**:
   - Deploy in private subnets
   - Use security groups/network policies
   - Restrict database access to API pods only

2. **TLS/SSL Configuration**:
   - Use Let's Encrypt for SSL certificates
   - Enable HTTP/2 for better performance
   - Implement proper certificate rotation

3. **Secrets Management**:
```yaml
# Use external secret management
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "labels-api"
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "labels_rules.yml"
    
    scrape_configs:
      - job_name: 'labels-api'
        static_configs:
          - targets: ['labels-api-service:8000']
        metrics_path: '/v1/metrics'
        scrape_interval: 15s
        
      - job_name: 'clickhouse'
        static_configs:
          - targets: ['clickhouse-service:9363']
        
      - job_name: 'redis'
        static_configs:
          - targets: ['redis-exporter:9121']
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
```

### Grafana Dashboards

Create monitoring dashboards for:

1. **API Performance**:
   - Request rate and latency percentiles
   - Error rates by endpoint
   - Cache hit rates

2. **Label Computation Metrics**:
   - Computation time by label type
   - Batch processing throughput
   - Validation failure rates

3. **Infrastructure Health**:
   - ClickHouse query performance
   - Redis memory usage
   - Pod resource utilization

### Alerting Rules

```yaml
# alerts.yml
groups:
- name: labels-api-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(label_computation_duration_seconds_bucket[5m])) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High label computation latency"
      description: "P99 latency is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(label_computation_errors_total[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High error rate in label computation"

  - alert: CacheHitRateLow
    expr: label_cache_hit_rate < 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Cache hit rate below threshold"
```

## Scaling & Performance

### Horizontal Scaling

1. **API Scaling**:
   - Use HPA based on CPU, memory, and custom metrics
   - Scale between 3-20 replicas based on load
   - Implement graceful shutdown handling

2. **Database Scaling**:
   - Use ClickHouse cluster with sharding
   - Implement read replicas for query load
   - Consider data partitioning by instrument/date

3. **Cache Scaling**:
   - Deploy Redis Cluster for horizontal scaling
   - Use Redis Sentinel for high availability
   - Implement cache warming strategies

### Performance Optimization

```python
# Production performance settings
PERFORMANCE_CONFIG = {
    "batch_size": 25000,
    "parallel_workers": 16,
    "connection_pool_size": 50,
    "query_timeout": 30,
    "cache_ttl": 3600,
    "preload_instruments": ["EURUSD", "GBPUSD", "USDJPY"],
    "warm_cache_hours": 24,
}
```

## Backup & Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# ClickHouse backup script
BACKUP_DIR="/backups/clickhouse/$(date +%Y-%m-%d)"
RETENTION_DAYS=30

# Create backup
clickhouse-client --query="BACKUP DATABASE quantx TO Disk('backups', '${BACKUP_DIR}')"

# Cleanup old backups
find /backups/clickhouse -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \;
```

### Disaster Recovery

1. **Multi-Region Deployment**:
   - Deploy in multiple availability zones
   - Use cross-region database replication
   - Implement automatic failover

2. **Data Recovery Procedures**:
   - Regular backup testing
   - Point-in-time recovery capability
   - RTO/RPO targets: 4 hours/1 hour

### Deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database schema initialized
- [ ] Monitoring dashboards created
- [ ] Alerting rules configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

This deployment guide provides comprehensive instructions for deploying the Label Computation System in production environments. Choose the deployment method that best fits your infrastructure requirements and scaling needs.