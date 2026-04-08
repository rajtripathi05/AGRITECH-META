"""
server/app.py – FastAPI server entry point for multi-mode deployment.
Re-exports the app from the root app module.
"""
import sys, os

# Ensure root directory is on path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: F401
