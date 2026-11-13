"""Import the extension (``demos/translation``) and the kernel (repo root) from source."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_DEMO))
for path in (_REPO, _DEMO):
    if path not in sys.path:
        sys.path.insert(0, path)
