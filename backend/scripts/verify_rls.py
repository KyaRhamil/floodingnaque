hyyyyyyyyyyyyyyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh#!/usr/bin/env python3
"""Verify RLS status on Supabase tables."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import text
from app.models.db import engine

print("=" * 60)
print("RLS VERIFICATION")
print("=" * 60)

with engine.connect() as conn:
    print("\n--- RLS Status ---")
    r = conn.execute(text("""
        SELECT tablename, rowsecurity 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename IN ('weather_data', 'predictions', 'alert_history', 'model_registry')
    """))
    for row in r:
        status = "Enabled" if row[1] else "Disabled"
        print(f"  {row[0]}: {status}")
    
    print("\n--- Policies ---")
    r2 = conn.execute(text("""
        SELECT tablename, policyname, cmd, roles 
        FROM pg_policies 
        WHERE schemaname = 'public' 
        ORDER BY tablename, policyname
    """))
    for row in r2:
        print(f"  {row[0]}: {row[1]} ({row[2]}) -> {row[3]}")

print("\n" + "=" * 60)
