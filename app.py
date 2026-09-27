"""Sigmo V2 — deployment entrypoint.

Runs the production FastAPI application (``backend.main:app``) under
uvicorn, binding to 0.0.0.0 on the port the platform provides
(7860 by default; PORT overrides locally).
"""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )
