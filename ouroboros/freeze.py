"""Protocol Freezer.

파일럿 종료 후, 확증실험 시작 전에 호출된다.
동결 대상: 실험 코드 + 분석 코드 + spec + 에이전트 프롬프트(해석에 영향).
동결 후 verify()가 실패하면 확증실험 결과는 무효 처리한다.
"""
from __future__ import annotations
import os, json, time, subprocess, sys
from .ledger import sha256_file

FREEZE_FILE = "protocol_freeze.json"


def _walk_hashes(root: str) -> dict:
    out = {}
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn == FREEZE_FILE or fn.endswith((".pyc", ".db")):
                continue
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def freeze(experiment_dir: str, spec: dict, prompt_dir: str | None = None) -> str:
    record = {
        "experiment_id": spec["experiment_id"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "file_hashes": _walk_hashes(experiment_dir),
        "prompt_hashes": _walk_hashes(prompt_dir) if prompt_dir else {},
        "python": sys.version,
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True).stdout.splitlines(),
        "seeds": spec["seeds"],
        "pilot_seeds": spec.get("pilot_seeds"),
        "confirmatory_seeds": spec.get("confirmatory_seeds"),
        "stats": spec["stats"],
        "metrics": spec["metrics"],
        "kill_criteria": spec["kill_criteria"],
        "claim_ceiling": spec["claim_ceiling"],
        "pilot_excluded": True,  # 파일럿 데이터는 확증 분석에서 제외
    }
    path = os.path.join(experiment_dir, FREEZE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def verify(experiment_dir: str) -> list[str]:
    """동결 이후 변경된 파일 목록 반환. 비어 있으면 무결."""
    path = os.path.join(experiment_dir, FREEZE_FILE)
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    current = _walk_hashes(experiment_dir)
    changed = []
    for rel, h in record["file_hashes"].items():
        # 확증 결과물(results/)은 동결 이후 생성되므로 제외
        if rel.startswith("results"):
            continue
        if current.get(rel) != h:
            changed.append(rel)
    return changed
