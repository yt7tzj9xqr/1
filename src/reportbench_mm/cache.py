from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable


class JsonCache:
    """Content-addressed SQLite cache shared safely by resumed runs."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        db = self._connect()
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, namespace TEXT NOT NULL, value TEXT NOT NULL, "
                "created_at REAL NOT NULL)"
            )
            db.commit()
        finally:
            db.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def key(namespace: str, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{namespace}\n{raw}".encode("utf-8")).hexdigest()

    def get(self, namespace: str, payload: Any) -> Any | None:
        key = self.key(namespace, payload)
        db = self._connect()
        try:
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        finally:
            db.close()
        return json.loads(row[0]) if row else None

    def put(self, namespace: str, payload: Any, value: Any) -> Any:
        key = self.key(namespace, payload)
        encoded = json.dumps(value, ensure_ascii=False)
        with self._lock:
            db = self._connect()
            try:
                db.execute(
                    "INSERT OR REPLACE INTO cache(key,namespace,value,created_at) VALUES(?,?,?,?)",
                    (key, namespace, encoded, time.time()),
                )
                db.commit()
            finally:
                db.close()
        return value

    def get_or_create(self, namespace: str, payload: Any, factory: Callable[[], Any]) -> Any:
        cached = self.get(namespace, payload)
        if cached is not None:
            return cached
        return self.put(namespace, payload, factory())
