"""Record completed demo inferences in a persistent Space volume."""

from datetime import datetime, timedelta
from functools import lru_cache
import logging
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4
from zoneinfo import ZoneInfo


USAGE_DIR = Path(os.environ.get("RSN_USAGE_DIR", "/mnt/rsn-usage"))
PACIFIC = ZoneInfo("America/Los_Angeles")
LOCK = Lock()
LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _read_counts(day, cache_window):
    first = (day - timedelta(days=29)).isoformat()
    last = day.isoformat()
    counts_by_day = {
        folder.name: sum(marker.suffix == ".run" for marker in folder.iterdir())
        for folder in USAGE_DIR.iterdir()
        if first <= folder.name <= last and folder.is_dir()
    }
    return counts_by_day.get(last, 0), sum(counts_by_day.values())


def usage_counts():
    """Return today's count and the count across the last 30 Pacific dates."""
    with LOCK:
        try:
            day = datetime.now(PACIFIC).date()
            # Use a new cache key every 30 seconds so other workers' writes become visible.
            return _read_counts(day, int(monotonic() // 30))
        except OSError:
            return None, None


def record_success():
    """Write one anonymous marker after an image or video has been estimated."""
    with LOCK:
        try:
            if not USAGE_DIR.is_dir():
                return
            folder = USAGE_DIR / datetime.now(PACIFIC).date().isoformat()
            folder.mkdir(exist_ok=True)
            # Unique objects avoid lost counts when two workers complete concurrently.
            marker = folder / f"{uuid4().hex}.run"
            marker.write_text("1\n", encoding="ascii")
        except OSError as error:
            LOG.warning("Could not persist a successful pose estimate (%s)", type(error).__name__)
        finally:
            _read_counts.cache_clear()
