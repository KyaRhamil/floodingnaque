import json
import os
from pathlib import Path

credentials_file = Path("astral-archive-482008-g2-b4f2279053c0.json")

if not credentials_file.exists():
    print(f"❌ Credentials file not found: {credentials_file}")
    exit(1)

print(f"✓ Found credentials file: {credentials_file}")

try:
    with open(credentials_file, 'r') as f:
        creds = json.load(f)
    
    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
    missing = [field for field in required_fields if field not in creds]
    
    if missing:
        print(f"❌ Missing required fields: {missing}")
        exit(1)
    
    print(f"✓ Credential type: {creds['type']}")
    print(f"✓ Project ID: {creds['project_id']}")
    print(f"✓ Client email: {creds['client_email']}")
    print(f"✓ Private key ID: {creds['private_key_id'][:20]}...")
    
    private_key = creds['private_key']
    if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
        print(f"❌ Private key format invalid (should start with BEGIN PRIVATE KEY)")
        exit(1)
    
    if not private_key.strip().endswith('-----END PRIVATE KEY-----'):
        print(f"❌ Private key format invalid (should end with END PRIVATE KEY)")
        exit(1)
    
    print(f"✓ Private key format valid")
    
    # Test Earth Engine initialization
    print("\n--- Testing Earth Engine Authentication ---")
    import ee
    
    credentials = ee.ServiceAccountCredentials(
        creds['client_email'],
        str(credentials_file)
    )
    ee.Initialize(credentials, project=creds['project_id'])
    
    print("✓ Earth Engine authentication successful!")
    
    # Test a simple query
    dataset = ee.ImageCollection('NOAA/GFS0P25').limit(1)
    info = dataset.getInfo()
    print(f"✓ Successfully queried Earth Engine (got {len(info.get('features', []))} results)")
    
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON in credentials file: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All checks passed!")