from logging.config import fileConfig
import os
import sys
import re
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import pool

from alembic import context

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models for autogenerate support
from app.models.db import Base
from app.core.config import load_env

# Load environment variables
load_env()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def _get_pg_driver():
    """Determine the best PostgreSQL driver for the current platform."""
    if sys.platform == 'win32':
        return 'pg8000'
    try:
        import psycopg2
        return 'psycopg2'
    except ImportError:
        return 'pg8000'


def _prepare_db_url(url):
    """
    Prepare database URL for the current platform.
    Handles SSL mode for pg8000 which doesn't accept sslmode in URL.
    Supports full SSL verification with CA certificate loading.
    
    Returns:
        tuple: (prepared_url, connect_args)
    """
    import logging
    logger = logging.getLogger('alembic.env')
    
    if not url or url.startswith('sqlite'):
        return url, {}
    
    pg_driver = _get_pg_driver()
    connect_args = {}
    
    # Get SSL configuration from environment
    db_ssl_mode = os.getenv('DB_SSL_MODE', '').lower()
    db_ssl_ca_cert = os.getenv('DB_SSL_CA_CERT', '')
    
    # Determine app environment for default SSL mode
    app_env = os.getenv('APP_ENV', 'development').lower()
    if not db_ssl_mode:
        # Default: verify-full for production/staging, require for development
        if app_env in ('production', 'prod', 'staging', 'stage'):
            db_ssl_mode = 'verify-full'
        else:
            db_ssl_mode = 'require'
    
    # Handle SSL mode for pg8000 (doesn't accept sslmode in URL like psycopg2)
    if pg_driver == 'pg8000':
        import ssl
        
        # Remove sslmode from URL for pg8000 if present
        if 'sslmode=' in url:
            sslmode_match = re.search(r'[?&]sslmode=([^&]*)', url)
            if sslmode_match:
                url_sslmode = sslmode_match.group(1).lower()
                # Use URL sslmode if DB_SSL_MODE not explicitly set
                if not os.getenv('DB_SSL_MODE'):
                    db_ssl_mode = url_sslmode
                # Remove sslmode parameter from URL
                url = re.sub(r'[?&]sslmode=[^&]*', '', url)
                # Clean up URL if we left a dangling ? or &
                url = url.replace('?&', '?').rstrip('?')
        
        if db_ssl_mode in ('require', 'verify-ca', 'verify-full'):
            ssl_context = ssl.create_default_context()
            
            if db_ssl_mode == 'require':
                # Encrypted connection, no certificate verification
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.info("Alembic SSL mode 'require': Encrypted connection without certificate verification")
                
            elif db_ssl_mode in ('verify-ca', 'verify-full'):
                # Full certificate verification
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                
                if db_ssl_mode == 'verify-full':
                    ssl_context.check_hostname = True
                    logger.info("Alembic SSL mode 'verify-full': Full certificate and hostname verification")
                else:
                    ssl_context.check_hostname = False
                    logger.info("Alembic SSL mode 'verify-ca': Certificate verification without hostname check")
                
                # Load CA certificate
                if db_ssl_ca_cert:
                    if os.path.isfile(db_ssl_ca_cert):
                        ssl_context.load_verify_locations(db_ssl_ca_cert)
                        logger.info(f"Alembic loaded CA certificate from: {db_ssl_ca_cert}")
                    else:
                        error_msg = (
                            f"CRITICAL: SSL certificate file not found: {db_ssl_ca_cert}. "
                            f"SSL mode '{db_ssl_mode}' requires a valid CA certificate. "
                            "Set DB_SSL_CA_CERT to the path of your CA certificate file."
                        )
                        logger.error(error_msg)
                        # Fail fast in production/staging
                        if app_env in ('production', 'prod', 'staging', 'stage'):
                            raise ValueError(error_msg)
                        else:
                            # Fall back to require mode in development
                            logger.warning("Alembic falling back to SSL mode 'require' for development")
                            ssl_context.check_hostname = False
                            ssl_context.verify_mode = ssl.CERT_NONE
                            db_ssl_mode = 'require'
                else:
                    error_msg = (
                        f"CRITICAL: DB_SSL_CA_CERT not set but SSL mode is '{db_ssl_mode}'. "
                        "Certificate verification modes require DB_SSL_CA_CERT to be set."
                    )
                    logger.error(error_msg)
                    # Fail fast in production/staging
                    if app_env in ('production', 'prod', 'staging', 'stage'):
                        raise ValueError(error_msg)
                    else:
                        # Fall back to require mode in development
                        logger.warning("Alembic falling back to SSL mode 'require' for development")
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE
                        db_ssl_mode = 'require'
            
            connect_args['ssl_context'] = ssl_context
            logger.info(f"Alembic configured SSL context for pg8000 (sslmode={db_ssl_mode})")
    
    # Add driver to URL
    if url.startswith('postgres://'):
        url = url.replace('postgres://', f'postgresql+{pg_driver}://', 1)
    elif url.startswith('postgresql://') and '+' not in url.split('://')[0]:
        url = url.replace('postgresql://', f'postgresql+{pg_driver}://', 1)
    
    return url, connect_args


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Get database URL from environment variable
    url = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
    url, _ = _prepare_db_url(url)
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Get database URL from environment variable
    db_url = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
    db_url, connect_args = _prepare_db_url(db_url)
    
    # Create engine with proper SSL handling
    connectable = create_engine(
        db_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
