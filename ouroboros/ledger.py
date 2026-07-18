"""Ledger: 연구의 기준 상태(source of truth).

에이전트 대화가 아니라 이 SQLite 파일이 연구의 공식 상태다.
- state:      상태기계 현재 상태
- events:     모든 상태 전이 기록 (감사 추적)
- gates:      인간 승인 기록
- runs:       실험 실행 기록 (seed 단위)
- claims:     주장 원장 (2축 격자: 기계론 E / 대응 H)
- artifacts:  산출물 경로 + sha256
"""
from __future__ import annotations
import sqlite3, json, time, hashlib, os

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
  ts REAL, actor TEXT, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS gates (
  gate TEXT, experiment_id TEXT, approved_by TEXT, ts REAL, note TEXT);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, experiment_id TEXT, phase TEXT, seed INTEGER,
  status TEXT, result_path TEXT, sha256 TEXT, started REAL, finished REAL);
CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY, experiment_id TEXT, text TEXT,
  e_level INTEGER, h_level INTEGER, status TEXT, ts REAL, evidence TEXT);
CREATE TABLE IF NOT EXISTS artifacts (
  path TEXT, sha256 TEXT, experiment_id TEXT, kind TEXT, ts REAL);
"""

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

class Ledger:
    def __init__(self, path: str = "ouroboros.db"):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- state machine persistence --
    def get_state(self, default="IDLE") -> str:
        row = self.db.execute("SELECT value FROM state WHERE key='fsm'").fetchone()
        return row[0] if row else default

    def set_state(self, s: str, actor="system", note=""):
        prev = self.get_state()
        self.db.execute("INSERT OR REPLACE INTO state VALUES('fsm',?)", (s,))
        self.event(actor, "transition", {"from": prev, "to": s, "note": note})
        self.db.commit()

    def event(self, actor: str, kind: str, detail: dict):
        self.db.execute("INSERT INTO events VALUES(?,?,?,?)",
                        (time.time(), actor, kind, json.dumps(detail, ensure_ascii=False)))
        self.db.commit()

    # -- gates --
    def approve_gate(self, gate: str, experiment_id: str, approver: str, note=""):
        self.db.execute("INSERT INTO gates VALUES(?,?,?,?,?)",
                        (gate, experiment_id, approver, time.time(), note))
        self.event(approver, "gate_approved", {"gate": gate, "experiment": experiment_id})
        self.db.commit()

    def gate_approved(self, gate: str, experiment_id: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM gates WHERE gate=? AND experiment_id=?",
            (gate, experiment_id)).fetchone()
        return row is not None

    # -- runs --
    def record_run(self, run_id, experiment_id, phase, seed, status,
                   result_path=None, started=None, finished=None):
        sha = sha256_file(result_path) if result_path and os.path.exists(result_path) else None
        self.db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?)",
                        (run_id, experiment_id, phase, seed, status,
                         result_path, sha, started, finished))
        self.db.commit()

    def runs_for(self, experiment_id, phase=None):
        q = "SELECT * FROM runs WHERE experiment_id=?"
        args = [experiment_id]
        if phase:
            q += " AND phase=?"; args.append(phase)
        return self.db.execute(q, args).fetchall()

    # -- claims --
    def upsert_claim(self, claim_id, experiment_id, text, e_level, h_level,
                     status, evidence: dict):
        self.db.execute("INSERT OR REPLACE INTO claims VALUES(?,?,?,?,?,?,?,?)",
                        (claim_id, experiment_id, text, e_level, h_level,
                         status, time.time(), json.dumps(evidence, ensure_ascii=False)))
        self.event("governor", "claim_update",
                   {"claim": claim_id, "E": e_level, "H": h_level, "status": status})
        self.db.commit()

    def claims(self):
        return self.db.execute("SELECT * FROM claims").fetchall()

    def register_artifact(self, path, experiment_id, kind):
        self.db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?)",
                        (path, sha256_file(path), experiment_id, kind, time.time()))
        self.db.commit()
