"""
server/app.py – Entry point for multi-mode deployment.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
