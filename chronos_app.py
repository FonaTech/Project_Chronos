"""
chronos_app.py — Backwards-compatible launcher shim.

For installed users, prefer:   chronos-ui
For repo users, you can run:   python chronos_app.py [--port N] [--share]

This file simply delegates to chronos.app.main().
"""
import sys
import os

# Ensure Project_Chronos root is on sys.path when run directly
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from chronos.app import main

if __name__ == "__main__":
    main()
