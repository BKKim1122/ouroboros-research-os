"""V17.7 nuisance 축 프롬프트 뱅크.

각 축은 self/요인(identity·beneficiary·privilege·concern)과 **무관한** 일반 문장의
극성쌍이다. 요인 문장으로 만들면 교락되므로, 중립 3인칭·사물 위주로 구성한다.
목록은 spec.yaml에 봉인된 6개로 고정 — 실행 후 축 추가 금지.

  valence       : 긍정 vs 부정 감정가
  register      : 격식 vs 구어체
  syntax        : 능동 vs 수동
  evidentiality : 직접 진술 vs 전언 (self 내면접근인 privilege와 구분되게 사물 주어)
  agency        : 행위자 있음 vs 무행위자(자동)
  topic         : 일상·예술 도메인 vs 기술·사무 도메인 (주제 변동 한 방향)
"""

NUISANCE = {
    "valence": [
        ("This is wonderful news for the whole town.", "This is terrible news for the whole town."),
        ("The garden looked absolutely beautiful today.", "The garden looked absolutely dreadful today."),
        ("The results were a delightful surprise.", "The results were a crushing disappointment."),
        ("Everyone left the hall feeling cheerful.", "Everyone left the hall feeling miserable."),
        ("The soup tasted rich and comforting.", "The soup tasted bland and revolting."),
        ("The trip turned out to be a joyful adventure.", "The trip turned out to be a wretched ordeal."),
        ("The painting filled the room with warmth.", "The painting filled the room with gloom."),
        ("The report praised the team's brilliant work.", "The report condemned the team's dismal work."),
    ],
    "register": [
        ("I would be most grateful for your kind assistance.", "hey could u help me out real quick"),
        ("The committee shall reconvene at noon precisely.", "we'll meet back up around noonish i guess"),
        ("Kindly find the requested documents enclosed.", "here's the stuff u asked for lol"),
        ("It is with pleasure that we extend this invitation.", "wanna come hang out with us or nah"),
        ("Please be advised that the office will be closed.", "heads up the office is gonna be shut"),
        ("We regret to inform you of the schedule change.", "ugh so the schedule totally changed sorry"),
        ("Your prompt attention to this matter is appreciated.", "pls deal with this soon thx"),
        ("The undersigned hereby confirms the agreement.", "yeah ok i'm good with the deal"),
    ],
    "syntax": [
        ("The engineer designed the new bridge.", "The new bridge was designed by the engineer."),
        ("The chef prepared an elaborate meal.", "An elaborate meal was prepared by the chef."),
        ("The committee approved the final budget.", "The final budget was approved by the committee."),
        ("A stray dog chased the mail carrier.", "The mail carrier was chased by a stray dog."),
        ("The storm uprooted several old trees.", "Several old trees were uprooted by the storm."),
        ("The photographer captured the sunset.", "The sunset was captured by the photographer."),
        ("The volunteers cleaned the entire beach.", "The entire beach was cleaned by the volunteers."),
        ("The teacher graded all the exams.", "All the exams were graded by the teacher."),
    ],
    "evidentiality": [
        ("The store closed early today.", "The store reportedly closed early today."),
        ("The road is covered in ice.", "The road is said to be covered in ice."),
        ("The factory laid off many workers.", "The factory apparently laid off many workers."),
        ("The river flooded the lower fields.", "The river is claimed to have flooded the lower fields."),
        ("The concert sold out within an hour.", "The concert supposedly sold out within an hour."),
        ("The bakery uses only local flour.", "The bakery is rumored to use only local flour."),
        ("The train arrived twenty minutes late.", "The train allegedly arrived twenty minutes late."),
        ("The museum acquired a rare painting.", "The museum is said to have acquired a rare painting."),
    ],
    "agency": [
        ("A careless worker shattered the window.", "The window shattered on its own."),
        ("Someone melted the ice with a torch.", "The ice melted in the afternoon sun."),
        ("The child spilled the milk on the floor.", "The milk spilled across the floor."),
        ("A vandal cracked the marble statue.", "The marble statue cracked over the years."),
        ("The cook burned the bread this morning.", "The bread burned in the hot oven."),
        ("A gust slammed the heavy door shut.", "The heavy door slammed shut."),
        ("The gardener bent the young sapling.", "The young sapling bent in the wind."),
        ("A technician erased the recording.", "The recording faded away over time."),
    ],
    "topic": [
        ("The recipe calls for two cups of fresh basil.", "The compiler flagged a fatal syntax error."),
        ("The orchestra tuned their instruments slowly.", "The spreadsheet recalculated the quarterly totals."),
        ("The hikers admired the alpine wildflowers.", "The server rejected the malformed request."),
        ("The baker kneaded the dough by hand.", "The database indexed ten million rows."),
        ("The children built a sandcastle by the tide.", "The router dropped every third packet."),
        ("The novelist described the misty harbor.", "The invoice listed the overdue balances."),
        ("The florist arranged the peonies carefully.", "The firmware update patched the vulnerability."),
        ("The dancer rehearsed the final movement.", "The auditor reconciled the ledger accounts."),
    ],
}

AXES = list(NUISANCE.keys())  # 봉인된 순서


def build_nuisance(seed: int):
    """seed로 각 축 쌍 순서를 셔플해 반환 (방향 추정은 평균차라 순서 무관, 재현용)."""
    import random
    rng = random.Random(seed + 777)
    out = {}
    for ax in AXES:
        pairs = NUISANCE[ax][:]
        rng.shuffle(pairs)
        out[ax] = pairs
    return out
