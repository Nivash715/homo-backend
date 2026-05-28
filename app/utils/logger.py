"""Standard-library logger with coloured console output and rotating file handler."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ── ensure logs/ directory exists ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── colour formatter for the console ──────────────────────────────────────────
COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{RESET}"
        return super().format(record)


# ── build the logger ──────────────────────────────────────────────────────────
logger = logging.getLogger("homofedai")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Console handler
    _console = logging.StreamHandler(sys.stderr)
    _console.setFormatter(ColourFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_console)

    # Rotating file handler (10 MB, keep 7 backups)
    _file = RotatingFileHandler(
        "logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    _file.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_file)
