"""SQLite storage for confirmed incidents."""
import sqlite3
from datetime import datetime
from pathlib import Path


class IncidentDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, mode TEXT,
                frame_idx INTEGER, video_time_s REAL, probability REAL,
                evidence_path TEXT, dispatch_status TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, source, mode, frame_idx, video_time_s, probability, evidence_path, dispatch_status):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO incidents (created_at, source, mode, frame_idx, video_time_s, probability,"
                " evidence_path, dispatch_status) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), source, mode, frame_idx,
                 round(video_time_s, 2), round(probability, 4), str(evidence_path), dispatch_status))
            return cur.lastrowid

    def list(self, limit=100):
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))]

    def clear(self):
        with self._conn() as c:
            c.execute("DELETE FROM incidents")
