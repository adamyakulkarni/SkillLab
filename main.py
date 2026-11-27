"""Entry point for Team Infinite Loop's Bengaluru Air Quality Tkinter Application."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
import tkinter as tk

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_app import AirQualityApp


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    AirQualityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
