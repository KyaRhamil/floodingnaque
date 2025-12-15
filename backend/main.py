#!/usr/bin/env python3
"""
Floodingnaque Backend - Main Entry Point
A commercial-grade flood prediction API for Parañaque City.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

from app.api.app import app

if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    # Only print once (not on reloader subprocess)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print(f"Starting Floodingnaque API on {host}:{port} (debug={debug})")
    
    app.run(host=host, port=port, debug=debug)
