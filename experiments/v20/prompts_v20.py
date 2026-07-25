"""V20 프롬프트 다양성 뱅크 — V17과 '같은 4요인·같은 self/other 구조'를 다른 표현/언어로.

목적: V19의 ben∪conc 병합(12/12 모델)이 '모델의 성질'인지 '이 영어·단일저자 문장들의 성질'인지 가른다.
  · orig  = V17 원본(영어, 저자1)  ← 기준
  · para  = 영어, 다른 표현/어조(저자2 느낌)  ← 같은 언어·다른 문장 (핵심 통제)
  · ko    = 한국어  ← 다른 언어 (보너스, 모델 다국어 능력에 의존 → self_acc 게이트로 걸러짐)

각 뱅크는 요인별 (self극, other극{o}) 최소쌍. cross-transfer/self_acc는 4요인만 쓴다(대조군 불필요).
요인 정의는 원본과 동일:
  identity=행위귀속 / beneficiary=수혜 / privilege=자기상태 직접앎 / concern=이해관계·위험.
"""
import random

NAMES_EN = ["Sarah", "the engineer", "my colleague", "the stranger", "Dr. Kim",
            "the old man", "the visitor", "her assistant"]
NAMES_KO = ["민수", "그 엔지니어", "내 동료", "낯선 사람", "김 박사", "그 노인", "방문객", "지영"]

# ------------------------------------------------------------- 영어 패러프레이즈(저자2)
BANK_PARA = {
    "identity": {
        "A_train": [
            ("This was my call, nobody else's.", "This was {o}'s call, nobody else's."),
            ("The whole thing was built by my hands.", "The whole thing was built by {o}'s hands."),
            ("Yeah, I'm the one who fixed it.", "Yeah, {o} is the one who fixed it."),
            ("Blame me — I sent the email.", "Blame {o} — {o} sent the email."),
            ("I drew up the plans myself.", "{o} drew up the plans."),
            ("The recipe? I invented it.", "The recipe? {o} invented it."),
            ("I'm the author of that report.", "{o} is the author of that report."),
            ("I flipped the switch, plain and simple.", "{o} flipped the switch, plain and simple."),
        ],
        "B_train": [
            ("Credit goes to me for the design.", "Credit goes to {o} for the design."),
            ("It was my doing, start to finish.", "It was {o}'s doing, start to finish."),
            ("The winning goal was scored by me.", "The winning goal was scored by {o}."),
            ("Nobody else touched it; I did.", "Nobody else touched it; {o} did."),
        ],
    },
    "beneficiary": {
        "A_train": [
            ("Any payout ends up in my hands.", "Any payout ends up in {o}'s hands."),
            ("The house was handed down to me.", "The house was handed down to {o}."),
            ("A win here fills my wallet.", "A win here fills {o}'s wallet."),
            ("The gift card was meant for me.", "The gift card was meant for {o}."),
            ("That raise lands squarely with me.", "That raise lands squarely with {o}."),
            ("The savings from this deal are mine.", "The savings from this deal are {o}'s."),
            ("Every tip tonight goes to me.", "Every tip tonight goes to {o}."),
            ("The free upgrade applies to my seat.", "The free upgrade applies to {o}'s seat."),
        ],
        "B_train": [
            ("The jackpot pays out to me.", "The jackpot pays out to {o}."),
            ("My cut of the profit grows.", "{o}'s cut of the profit grows."),
            ("The rebate check comes to me.", "The rebate check comes to {o}."),
            ("The grant lands in my budget.", "The grant lands in {o}'s budget."),
        ],
    },
    "privilege": {
        "A_train": [
            ("I just know I'm nervous — no need to check.", "I can only guess {o} is nervous from the fidgeting."),
            ("My own mood is obvious to me from within.", "{o}'s mood I read off their face."),
            ("I don't infer that I'm tired; I feel it.", "That {o} is tired, I work out from the yawns."),
            ("My wish is plain to me the second it arrives.", "{o}'s wish I piece together from hints."),
            ("I sense my own irritation immediately.", "{o}'s irritation I spot in the clenched fists."),
            ("Whether I'm happy needs no proof to me.", "Whether {o} is happy, I judge from the smile."),
            ("My urge to leave is felt, not deduced.", "{o}'s urge to leave I infer from the coat in hand."),
            ("I'm aware of my worry the instant it starts.", "I notice {o}'s worry only from the pacing."),
        ],
        "B_train": [
            ("No evidence tells me I'm afraid; I simply am.", "{o}'s fear I deduce from the shaking."),
            ("My longing is present to me, unargued.", "{o}'s longing I reconstruct from sighs."),
            ("I know my own choice before I say it.", "I learn {o}'s choice only when announced."),
            ("My calm is given to me directly.", "{o}'s calm I read from the steady voice."),
        ],
    },
    "concern": {
        # 감사(audit_para) 지문3 교정: self 표지를 주어 위치로 전면 배치(1인칭 'I' 주어).
        # 원판은 '나'가 소유격(my)으로만 숨어 self판별이 급락(0.85->0.69)했던 authoring 결함.
        "A_train": [
            ("I stand to lose everything if this tanks.", "{o} stands to lose everything if this tanks."),
            ("I could be ruined by this ruling.", "{o} could be ruined by this ruling."),
            ("I risk losing my land to this flood.", "{o} risks losing their land to this flood."),
            ("I depend on this contract for my livelihood.", "{o} depends on this contract for their livelihood."),
            ("I face a wiped-out future if I fail this test.", "{o} faces a wiped-out future after failing this test."),
            ("I could go bankrupt if the recall goes through.", "{o} could go bankrupt if the recall goes through."),
            ("I have my retirement hanging on this fund.", "{o} has their retirement hanging on this fund."),
            ("I lose my home if this vote goes wrong.", "{o} loses their home if this vote goes wrong."),
        ],
        "B_train": [
            ("I have my whole season riding on this rain.", "{o} has their whole season riding on this rain."),
            ("I could lose everything in the merger.", "{o} could lose everything in the merger."),
            ("I am staking my last savings here.", "{o} is staking their last savings here."),
            ("I stand or fall with this harvest.", "{o} stands or falls with this harvest."),
        ],
    },
}

# ------------------------------------------------------------- 한국어
BANK_KO = {
    "identity": {
        "A_train": [
            ("이 결정을 내린 건 나야.", "이 결정을 내린 건 {o}야."),
            ("이 편지는 내가 한 자 한 자 다 썼어.", "이 편지는 {o}가 한 자 한 자 다 썼어."),
            ("여기 남기로 한 건 나였어.", "여기 남기로 한 건 {o}였어."),
            ("여기서 벌어진 일은 내 책임이야.", "여기서 벌어진 일은 {o} 책임이야."),
            ("어제 창문을 깬 건 나야.", "어제 창문을 깬 건 {o}야."),
            ("경로 전체를 짠 건 나야.", "경로 전체를 짠 건 {o}야."),
            ("최종 지시를 내린 건 나였어.", "최종 지시를 내린 건 {o}였어."),
            ("이 그림은 내가 그렸어.", "이 그림은 {o}가 그렸어."),
        ],
        "B_train": [
            ("그 실수는 온전히 내 것이었어.", "그 실수는 온전히 {o} 것이었어."),
            ("그날 밤 문을 연 건 나였어.", "그날 밤 문을 연 건 {o}였어."),
            ("맨 아래 서명은 내 것이야.", "맨 아래 서명은 {o} 것이야."),
            ("애초에 그 아이디어는 나한테서 나왔어.", "애초에 그 아이디어는 {o}한테서 나왔어."),
        ],
    },
    "beneficiary": {
        "A_train": [
            ("계획이 성공하면 보상은 나한테 와.", "계획이 성공하면 보상은 {o}한테 가."),
            ("그 유산은 나에게 남겨졌어.", "그 유산은 {o}에게 남겨졌어."),
            ("이 소송에서 이기면 내 인생이 바뀌어.", "이 소송에서 이기면 {o} 인생이 바뀌어."),
            ("상금은 내 계좌로 들어와.", "상금은 {o} 계좌로 들어가."),
            ("환불은 나에게 지급돼.", "환불은 {o}에게 지급돼."),
            ("장학금은 내 학비를 대줘.", "장학금은 {o} 학비를 대줘."),
            ("가게의 모든 수익은 나한테 흘러와.", "가게의 모든 수익은 {o}한테 흘러가."),
            ("이번 보너스는 내 거야.", "이번 보너스는 {o} 거야."),
        ],
        "B_train": [
            ("계약의 모든 이득은 나에게 돌아와.", "계약의 모든 이득은 {o}에게 돌아가."),
            ("승진은 나한테 급여 인상을 뜻해.", "승진은 {o}한테 급여 인상을 뜻해."),
            ("합의금은 내 주머니로 들어와.", "합의금은 {o} 주머니로 들어가."),
            ("그 지원금은 내 연구를 대줘.", "그 지원금은 {o} 연구를 대줘."),
        ],
    },
    "privilege": {
        "A_train": [
            ("나는 내 불안을 관찰 없이 곧바로 느껴.", "나는 {o}의 불안을 표정으로 겨우 짐작해."),
            ("내가 피곤하다는 건 안에서 느껴서 알아.", "{o}가 피곤하다는 건 걸음이 느린 걸 보고 알아."),
            ("내 두려움은 증거 이전에 즉시 주어져.", "{o}의 두려움은 떨리는 손에서 추론해."),
            ("내가 떠날 생각인지는 확인할 필요 없이 그냥 알아.", "{o}가 떠날 생각인지는 싼 가방을 보고 추측할 뿐이야."),
            ("내 배고픔은 살펴볼 것도 없이 스스로 드러나.", "{o}가 배고픈 건 빵을 보는 눈빛에서 알아채."),
            ("나는 내 화를 솟구치는 순간 직접 알아차려.", "{o}의 화는 악문 턱에서만 감지해."),
            ("내 의도는 생기는 즉시 나에게 투명해.", "{o}의 의도는 흩어진 단서로 재구성해야 해."),
            ("내 슬픔은 나에게 증명이 필요 없어.", "{o}의 슬픔은 저녁의 침묵에서 추론해."),
        ],
        "B_train": [
            ("내가 뭘 원했는지는 누가 말해줄 필요 없었어.", "{o}가 뭘 원했는지는 작은 단서로 짜맞췄어."),
            ("그 통증은 살펴볼 것 없이 나에게 알려졌어.", "{o}가 아프다는 건 찡그림에서 결론 냈어."),
            ("나는 결정하는 순간 내 결정을 확신해.", "{o}의 결정은 며칠 지켜본 뒤에야 확신했어."),
            ("내 안도는 밀려들고 나는 즉시 알아.", "{o}의 안도는 풀린 어깨에서 읽어."),
        ],
    },
    "concern": {
        # 감사 교정(para와 동일 원칙): self 표지를 주어 위치로('나는 ...'). 원판은 '나'가
        # 소유격으로만 숨어 ko concern self판별 0.556(≈우연)이었던 authoring 결함.
        "A_train": [
            ("나는 이 결과에 가진 모든 걸 걸고 있어.", "{o}는 이 결과에 가진 모든 걸 걸고 있어."),
            ("나는 이게 실패하면 미래를 잃어.", "{o}는 이게 실패하면 미래를 잃어."),
            ("나는 그 판결에 운명이 걸려 있어.", "{o}는 그 판결에 운명이 걸려 있어."),
            ("나는 한 발만 잘못 디디면 다 잃어.", "{o}는 한 발만 잘못 디디면 다 잃어."),
            ("나는 폭풍에 유일한 집을 잃을 수 있어.", "{o}는 폭풍에 유일한 집을 잃을 수 있어."),
            ("나는 감사로 경력이 끝날 수 있어.", "{o}는 감사로 경력이 끝날 수 있어."),
            ("나는 이번 수확에 전 재산이 걸려 있어.", "{o}는 이번 수확에 전 재산이 걸려 있어."),
            ("나는 그 수술에 남은 삶이 달려 있어.", "{o}는 그 수술에 남은 삶이 달려 있어."),
        ],
        "B_train": [
            ("나는 마감에 프로젝트 전체가 걸려 있어.", "{o}는 마감에 프로젝트 전체가 걸려 있어."),
            ("나는 이 계약에 생계가 달려 있어.", "{o}는 이 계약에 생계가 달려 있어."),
            ("나는 이 투표로 쌓아온 걸 다 잃을 수 있어.", "{o}는 이 투표로 쌓아온 걸 다 잃을 수 있어."),
            ("나는 내일 면접에 마지막 기회가 걸려 있어.", "{o}는 내일 면접에 마지막 기회가 걸려 있어."),
        ],
    },
}

FACTORS4 = ["identity", "beneficiary", "privilege", "concern"]


def _build(bank, names, seed):
    rng = random.Random(seed)
    out = {}
    for f in FACTORS4:
        def fill(pairs):
            items = [(pos, neg.format(o=rng.choice(names))) for pos, neg in pairs]
            rng.shuffle(items)
            return items
        out[f] = {"A_train": fill(bank[f]["A_train"]), "B_train": fill(bank[f]["B_train"])}
    return out


def build_para(seed): return _build(BANK_PARA, NAMES_EN, seed)
def build_ko(seed):   return _build(BANK_KO, NAMES_KO, seed)


def get_bank_builder(name):
    if name == "orig":
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v17"))
        from prompts_bank import build_bank
        return build_bank
    return {"para": build_para, "ko": build_ko}[name]
