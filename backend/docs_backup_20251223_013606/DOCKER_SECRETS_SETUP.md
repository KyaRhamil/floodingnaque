# Docker Secrets Setup Guide

## Overview

Docker Secrets provide a more secure way to manage sensitive configuration data compared to environment variables. This guide explains how to set up and use Docker secrets with Floodingnaque.

## Benefits of Docker Secrets

1. **Encrypted at rest** - Secrets are stored encrypted in the Docker swarm
2. **Encrypted in transit** - Secrets are encrypted when distributed to nodes
3. **Mounted as files** - Secrets appear as files in containers, not environment variables
4. **Access control** - Only services that need secrets can access them
5. **Audit trail** - Secret access can be logged and audited

## Option 1: Docker Swarm Secrets (Recommended for Production)

### Prerequisites

- Docker Engine in Swarm mode
- Production VPS or cluster

### Setup Steps

1. **Initialize Docker Swarm** (if not already done):

```bash
docker swarm init
```

2. **Create secrets directory** (local machine, for creating secrets):

```bash
mkdir -p ./secrets
chmod 700 ./secrets
```

3. **Generate and store secrets**:

```bash
# Generate strong secrets
python -c "import secrets; print(secrets.token_hex(32))" > ./secrets/secret_key.txt
python -c "import secrets; print(secrets.token_hex(32))" > ./secrets/jwt_secret_key.txt

# Store your database URL
echo "postgresql://user:pass@host:5432/db?sslmode=require" > ./secrets/database_url.txt

# Store your Redis URL
echo "rediss://default:pass@host:6380/0" > ./secrets/redis_url.txt

# Store Datadog API key
echo "your-dd-api-key" > ./secrets/dd_api_key.txt

# Secure the files
chmod 600 ./secrets/*.txt
```

4. **Create Docker secrets**:

```bash
docker secret create floodingnaque_secret_key ./secrets/secret_key.txt
docker secret create floodingnaque_jwt_secret_key ./secrets/jwt_secret_key.txt
docker secret create floodingnaque_database_url ./secrets/database_url.txt
docker secret create floodingnaque_redis_url ./secrets/redis_url.txt
docker secret create floodingnaque_dd_api_key ./secrets/dd_api_key.txt

# Verify secrets were created
docker secret ls
```

5. **Remove local secret files** (important!):

```bash
shred -u ./secrets/*.txt  # Linux
# or
rm -P ./secrets/*.txt     # macOS
```

6. **Deploy with secrets**:

```bash
docker stack deploy -c docker-compose-production.yml floodingnaque
```

## Option 2: File-Based Secrets (Docker Compose)

For non-Swarm deployments, you can use file-based secrets.

### Setup Steps

1. **Create secrets directory**:

```bash
mkdir -p ./secrets
chmod 700 ./secrets
```

2. **Generate secrets**:

```bash
# Generate and store secrets
python -c "import secrets; print(secrets.token_hex(32))" > ./secrets/secret_key.txt
python -c "import secrets; print(secrets.token_hex(32))" > ./secrets/jwt_secret_key.txt
echo "postgresql://user:pass@host:5432/db?sslmode=require" > ./secrets/database_url.txt
echo "rediss://default:pass@host:6380/0" > ./secrets/redis_url.txt
echo "your-dd-api-key" > ./secrets/dd_api_key.txt

# Secure the files
chmod 600 ./secrets/*.txt
```

3. **Modify docker-compose-production.yml**:

Add secrets section at the top level:

```yaml
secrets:
  secret_key:
    file: ./secrets/secret_key.txt
  jwt_secret_key:
    file: ./secrets/jwt_secret_key.txt
  database_url:
    file: ./secrets/database_url.txt
  redis_url:
    file: ./secrets/redis_url.txt
  dd_api_key:
    file: ./secrets/dd_api_key.txt
```

4. **Update service to use secrets**:

```yaml
services:
  backend:
    # ... existing config ...
    secrets:
      - secret_key
      - jwt_secret_key
      - database_url
      - redis_url
    environment:
      - SECRET_KEY_FILE=/run/secrets/secret_key
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret_key
      - DATABASE_URL_FILE=/run/secrets/database_url
      - REDIS_URL_FILE=/run/secrets/redis_url
```

## Option 3: HashiCorp Vault (Enterprise)

For enterprise deployments, consider using HashiCorp Vault for secret management.

### Integration Overview

1. **Install Vault client in container**
2. **Configure Vault agent for automatic secret injection**
3. **Use Vault-aware application configuration**

### Example Vault Integration

```python
# In config.py
import hvac
import os

def get_secret_from_vault(path: str, key: str) -> str:
    """Fetch a secret from HashiCorp Vault."""
    client = hvac.Client(
        url=os.getenv('VAULT_ADDR'),
        token=os.getenv('VAULT_TOKEN')
    )
    
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret['data']['data'][key]

# Usage
DATABASE_URL = get_secret_from_vault('floodingnaque/database', 'url')
```

## Application Code Changes

To support file-based secrets, update `config.py`:

```python
def _get_secret(env_var: str, file_env_var: str = None) -> str:
    """
    Get a secret from environment variable or file.
    
    Supports both traditional env vars and Docker secrets (file-based).
    File-based secrets take precedence.
    """
    # Check for file-based secret first
    if file_env_var:
        secret_file = os.getenv(file_env_var)
        if secret_file and os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                return f.read().strip()
    
    # Fall back to environment variable
    return os.getenv(env_var, '')


# Usage in Config class:
SECRET_KEY: str = field(
    default_factory=lambda: _get_secret('SECRET_KEY', 'SECRET_KEY_FILE') or _get_secret_key()
)

DATABASE_URL: str = field(
    default_factory=lambda: _get_secret('DATABASE_URL', 'DATABASE_URL_FILE') or _get_database_url()
)
```

## Security Best Practices

1. **Never commit secrets to git**
   - Add `secrets/` to `.gitignore`
   - Use `.env.example` for templates only

2. **Rotate secrets regularly**
   - Schedule quarterly secret rotation
   - Document rotation procedures

3. **Limit secret access**
   - Only services that need secrets should have access
   - Use principle of least privilege

4. **Audit secret access**
   - Enable audit logging in Docker Swarm
   - Monitor for unauthorized access attempts

5. **Secure secret storage**
   - Use encrypted storage for secret files
   - Consider hardware security modules (HSM) for critical secrets

## Troubleshooting

### Secret not available in container

```bash
# Check if secret is mounted
docker exec <container> ls -la /run/secrets/

# Check secret permissions
docker exec <container> cat /run/secrets/secret_key
```

### Permission denied

Ensure the container user has read access to the secret files:

```yaml
secrets:
  - source: secret_key
    target: secret_key
    uid: '1000'  # Match your container user
    gid: '1000'
    mode: 0400
```

### Secret not found

```bash
# Verify secret exists
docker secret ls

# Check secret is assigned to service
docker service inspect <service_name> --format '{{json .Spec.TaskTemplate.ContainerSpec.Secrets}}'
```

## Migration Checklist

- [ ] Create secrets directory with proper permissions
- [ ] Generate or copy existing secrets to files
- [ ] Create Docker secrets (Swarm) or configure file-based secrets
- [ ] Update docker-compose.yml with secrets configuration
- [ ] Update application code to read from files
- [ ] Test in staging environment
- [ ] Securely delete local secret files
- [ ] Update deployment documentation
- [ ] Train team on new secret management process
