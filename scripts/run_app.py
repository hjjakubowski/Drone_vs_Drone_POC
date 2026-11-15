from __future__ import annotations

import sys
from pathlib import Path

# Also support `python /path/to/project/scripts/run_app.py` from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.hud_desktop.main import main


if __name__ == "__main__":
    raise SystemExit(main())
