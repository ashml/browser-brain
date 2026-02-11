from __future__ import annotations

import datetime as dt
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from .models import HistoryRecord


CHROME_DEFAULT = Path("~/.config/google-chrome/Default/History").expanduser()
FIREFOX_DEFAULT_ROOT = Path("~/.mozilla/firefox").expanduser()


def _copy_sqlite_for_safe_reading(path: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="browser_brain_"))
    copied = tmp_dir / path.name
    shutil.copy2(path, copied)
    return copied


def _chrome_time_to_iso(chrome_ts: int) -> str:
    # microseconds since 1601-01-01
    epoch_start = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc)
    return (epoch_start + dt.timedelta(microseconds=chrome_ts)).isoformat()


def _firefox_time_to_iso(firefox_ts: int) -> str:
    # microseconds since Unix epoch
    return dt.datetime.fromtimestamp(firefox_ts / 1_000_000, tz=dt.timezone.utc).isoformat()


def discover_firefox_places() -> Optional[Path]:
    if not FIREFOX_DEFAULT_ROOT.exists():
        return None
    candidates = sorted(FIREFOX_DEFAULT_ROOT.glob("*.default*/places.sqlite"))
    return candidates[0] if candidates else None


def load_chrome_history(path: Optional[Path] = None, limit: Optional[int] = None) -> Iterator[HistoryRecord]:
    src = path or CHROME_DEFAULT
    if not src.exists():
        return iter(())

    safe_copy = _copy_sqlite_for_safe_reading(src)
    conn = sqlite3.connect(safe_copy)
    conn.row_factory = sqlite3.Row
    try:
        q = (
            "SELECT url, title, last_visit_time FROM urls "
            "WHERE url IS NOT NULL ORDER BY last_visit_time DESC"
        )
        if limit:
            q += f" LIMIT {int(limit)}"

        rows = conn.execute(q)
        for row in rows:
            title = (row["title"] or "").strip()
            url = (row["url"] or "").strip()
            snippet = f"{title} {url}".strip()
            yield HistoryRecord(
                url=url,
                title=title,
                snippet=snippet,
                visited_at=_chrome_time_to_iso(int(row["last_visit_time"] or 0)),
                browser="chrome",
            )
    finally:
        conn.close()
        shutil.rmtree(safe_copy.parent, ignore_errors=True)


def load_firefox_history(path: Optional[Path] = None, limit: Optional[int] = None) -> Iterator[HistoryRecord]:
    src = path or discover_firefox_places()
    if src is None or not src.exists():
        return iter(())

    safe_copy = _copy_sqlite_for_safe_reading(src)
    conn = sqlite3.connect(safe_copy)
    conn.row_factory = sqlite3.Row
    try:
        q = (
            "SELECT url, title, last_visit_date FROM moz_places "
            "WHERE url IS NOT NULL ORDER BY last_visit_date DESC"
        )
        if limit:
            q += f" LIMIT {int(limit)}"

        rows = conn.execute(q)
        for row in rows:
            title = (row["title"] or "").strip()
            url = (row["url"] or "").strip()
            snippet = f"{title} {url}".strip()
            yield HistoryRecord(
                url=url,
                title=title,
                snippet=snippet,
                visited_at=_firefox_time_to_iso(int(row["last_visit_date"] or 0)),
                browser="firefox",
            )
    finally:
        conn.close()
        shutil.rmtree(safe_copy.parent, ignore_errors=True)


def load_history(browser: str, path: Optional[Path] = None, limit: Optional[int] = None) -> list[HistoryRecord]:
    browser = browser.lower().strip()
    if browser == "chrome":
        return list(load_chrome_history(path=path, limit=limit))
    if browser == "firefox":
        return list(load_firefox_history(path=path, limit=limit))
    raise ValueError(f"Unsupported browser: {browser}")
