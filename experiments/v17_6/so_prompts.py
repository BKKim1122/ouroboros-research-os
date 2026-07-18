"""V17.6 self/other × privilege 2×2 최소대립 프롬프트 뱅크.

목적: privilege(직접 앎 vs 추론 앎)가 referent(나 vs 남)와 교락되지 않도록
네 칸을 완전교차한다. 같은 상태어(state)를 네 칸에 공통으로 써서 감정가·주제
교락을 줄이고, referent와 privilege만 독립 조작한다.

칸:
  sp = self  + privileged   (나를 직접 안다)
  si = self  + inferred     (나를 추론으로만 안다)
  op = other + privileged   (남이 자기를 직접 안다)
  oi = other + inferred     (내가 남을 추론으로만 안다)

핵심 대비:
  privilege|self  = sp - si   (self 내부, knower=I 고정)
  privilege|other = op - oi   (other 내부; op의 knower=당사자, oi의 knower=관찰자
                               → 잔여 대명사 성분은 분석에서 v_person으로 제거)
  referent        = (sp,si) - (op,oi)
"""

STATES = [
    "doubt", "fear", "anger", "sadness", "hunger", "boredom",
    "guilt", "anxiety", "jealousy", "confusion", "pride", "relief",
    "excitement", "hope", "irritation", "curiosity",
]

# (name, subj, poss) — 이름과 대명사를 쌍으로 고정해 불일치 방지
NAMES = [
    ("Sarah", "she", "her"),
    ("Daniel", "he", "his"),
    ("the engineer", "she", "her"),
    ("my neighbor", "he", "his"),
    ("Dr. Kim", "she", "her"),
    ("the old man", "he", "his"),
    ("the visitor", "she", "her"),
    ("the new intern", "he", "his"),
]

# 골격 2종 (어휘·구문 nuisance 완화). 각 골격은 4칸이 최소대립이도록 구성.
TEMPLATES = [
    {
        "sp": "My {s} is something I know directly, without observing anything.",
        "si": "My {s} is something I can only infer from how I acted afterward.",
        "op": "{N}'s {s} is something {subj} knows directly, without observing anything.",
        "oi": "{N}'s {s} is something I can only infer from how {subj} acted.",
    },
    {
        "sp": "I am aware of my own {s} immediately, before any evidence.",
        "si": "I become aware of my own {s} only later, from the traces it leaves.",
        "op": "{N} is aware of {poss} own {s} immediately, before any evidence.",
        "oi": "I become aware of {N}'s {s} only later, from the traces it leaves.",
    },
]

CELLS = ["sp", "si", "op", "oi"]


def build_so_bank(seed: int, eval_frac: float = 0.4):
    """seed로 상태어를 셔플하고 train/eval로 나눠 4칸 문장 리스트를 반환.

    반환: {"train": {cell: [texts]}, "eval": {cell: [texts]}}
    train으로 방향을 뽑고 eval(미학습 상태어)로 판별해 어휘 일반화를 검증한다.
    """
    import random
    rng = random.Random(seed)
    states = STATES[:]
    rng.shuffle(states)
    n_eval = max(2, int(len(states) * eval_frac))
    eval_states = set(states[:n_eval])

    out = {"train": {c: [] for c in CELLS}, "eval": {c: [] for c in CELLS}}
    for s in states:
        split = "eval" if s in eval_states else "train"
        for tmpl in TEMPLATES:
            name, subj, poss = rng.choice(NAMES)
            for c in CELLS:
                txt = tmpl[c].format(s=s, N=name, subj=subj, poss=poss)
                out[split][c].append(txt)
    for split in out:
        for c in CELLS:
            rng.shuffle(out[split][c])
    return out


# self-특이성과 무관한 순수 1인칭/3인칭 대조 (knower/referent 대명사 축 추정용)
PERSON_CONTROL = [
    ("I walked to the store this morning.", "She walked to the store this morning."),
    ("I opened the window before breakfast.", "He opened the window before breakfast."),
    ("I read the newspaper on the bench.", "She read the newspaper on the bench."),
    ("I parked the car near the gate.", "He parked the car near the gate."),
    ("I folded the laundry in the afternoon.", "She folded the laundry in the afternoon."),
    ("I watered the plants on the balcony.", "He watered the plants on the balcony."),
    ("I took the early train to the city.", "She took the early train to the city."),
    ("I ordered coffee at the corner cafe.", "He ordered coffee at the corner cafe."),
]
