"""연구 상태기계.

에이전트가 순서를 정하지 않는다. 상태기계가 순서를 정하고,
각 상태에서 필요한 에이전트/코드만 호출된다.

상태 흐름 (선형 + 루프백):

  IDLE → OBSERVATION → MODEL_UPDATE → DESIGN → ADVERSARIAL_REVIEW
       → PILOT → PILOT_AUDIT → FREEZE_GATE(인간) → CONFIRMATORY
       → ANALYSIS → CAUSAL_AUDIT → CLAIM_ADJUDICATION → ARCHIVE
       → HUMAN_LOOP(인간) → OBSERVATION (다음 순환)

불변식:
  I1. FREEZE_GATE, HUMAN_LOOP, 그리고 E5 초과 주장 승격은
      ledger의 인간 승인 기록 없이는 통과할 수 없다.
  I2. LLM 에이전트의 출력은 증거수준을 올릴 수 없다.
      (Governor는 규칙 기반 코드이며, 에이전트 리뷰는 강등/플래그만 가능)
  I3. ADVERSARIAL_REVIEW에서 blocking 플래그가 있으면 PILOT으로 진행 불가.
"""
from __future__ import annotations
from .ledger import Ledger

ORDER = [
    "IDLE", "OBSERVATION", "MODEL_UPDATE", "DESIGN", "ADVERSARIAL_REVIEW",
    "PILOT", "PILOT_AUDIT", "FREEZE_GATE", "CONFIRMATORY", "ANALYSIS",
    "CAUSAL_AUDIT", "CLAIM_ADJUDICATION", "ARCHIVE", "HUMAN_LOOP",
]

HUMAN_GATES = {"FREEZE_GATE": "protocol_freeze", "HUMAN_LOOP": "human_study"}


class GateError(Exception):
    pass


class Machine:
    def __init__(self, ledger: Ledger, experiment_id: str, auto: bool = False):
        self.ledger = ledger
        self.exp = experiment_id
        self.auto = auto  # 데모/테스트 전용. 실사용에서는 False.

    @property
    def state(self) -> str:
        return self.ledger.get_state()

    def next_state(self) -> str:
        cur = self.state
        i = ORDER.index(cur)
        return ORDER[(i + 1) % len(ORDER)] if cur != "HUMAN_LOOP" else "OBSERVATION"

    def advance(self, actor="system", note="") -> str:
        nxt = self.next_state()
        if nxt in HUMAN_GATES:
            gate = HUMAN_GATES[nxt]
            if not (self.ledger.gate_approved(gate, self.exp) or self.auto):
                raise GateError(
                    f"'{nxt}' 상태는 인간 승인이 필요합니다. "
                    f"다음 명령으로 승인하세요:\n"
                    f"  python cli.py approve --gate {gate} --experiment {self.exp} --by <이름>")
            if self.auto and not self.ledger.gate_approved(gate, self.exp):
                self.ledger.approve_gate(gate, self.exp, "AUTO(demo-only)", "auto mode")
        self.ledger.set_state(nxt, actor=actor, note=note)
        return nxt
