"""LLM 에이전트 계층.

원칙:
  - 에이전트는 prompts/ 아래 버전관리되는 프롬프트로 정의된다
    (프롬프트도 protocol freeze에 해시로 포함).
  - 에이전트 출력은 반드시 JSON 스키마를 따르고,
    'blocking_issues' / 'non_blocking_flags' 만 낼 수 있다.
    증거수준 승격 권한은 없다 (governor.py 참조).
  - ANTHROPIC_API_KEY가 없으면 stub 모드로 동작해
    파이프라인 자체는 항상 테스트 가능하다.
"""
from __future__ import annotations
import os, json, urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("OURO_MODEL", "claude-sonnet-4-6")
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPT_DIR, f"{name}.md"), encoding="utf-8") as f:
        return f.read()


def call_llm(system: str, user: str, max_tokens=2000) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return json.dumps({
            "stub": True,
            "blocking_issues": [],
            "non_blocking_flags": ["stub 모드: ANTHROPIC_API_KEY 미설정, 실제 검토 미수행"],
        }, ensure_ascii=False)
    body = json.dumps({
        "model": MODEL, "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data["content"])


def run_agent(name: str, payload: dict) -> dict:
    """에이전트 실행 → JSON 파싱. 파싱 실패 시 blocking 처리(안전 기본값)."""
    system = load_prompt(name)
    raw = call_llm(system, json.dumps(payload, ensure_ascii=False, indent=2))
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {"blocking_issues": [f"{name} 출력 파싱 실패 — 수동 검토 필요"],
               "non_blocking_flags": [], "raw": raw[:2000]}
    out.setdefault("blocking_issues", [])
    out.setdefault("non_blocking_flags", [])
    return out
