"""
chronos_app.py — launcher for the Chronos Web UI from a source checkout.

Installed users run ``chronos-ui`` (or ``chronos webui``); both start the same server.

    python chronos_app.py [--port N] [--host H] [--workdir DIR] [--runs-dir DIR]
    python chronos_app.py --legacy [--port N] [--share]     # the old Gradio console

The default is the FastAPI + single-page UI (chronos/webui). It listens on loopback without a login;
any other host needs --allow-remote and a key in CHRONOS_UI_KEY or CHRONOS_API_KEY.
"""
import os
import sys

# Ensure the Project_Chronos root is on sys.path when run directly.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from chronos.webui.server import main

if __name__ == "__main__":
    raise SystemExit(main())
