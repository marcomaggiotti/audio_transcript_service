"""Entrypoint: `python main.py` or `uvicorn main:app`."""

import logging

from app.config import get_settings
from app.main import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    # One process only: the job queue and job state are in-process.
    # Trust X-Forwarded-* from the platform proxy (e.g. Render terminates TLS in front of the container).
    uvicorn.run(app, host=settings.host, port=settings.port, proxy_headers=True, forwarded_allow_ips="*")
