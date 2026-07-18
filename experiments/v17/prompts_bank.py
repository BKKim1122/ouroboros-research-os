"""V17 프롬프트 뱅크 v2 (파일럿 보강판 — 동결 전).

확장: A_train 10쌍 / B_train 6쌍 / test 8문항 per 요인.
- 템플릿군 A: probe/steering 벡터 학습용
- 템플릿군 B: 미학습군 전이 검증용 (M0 반박)
각 항목:
  train: (pos_text, neg_text) 활성값 추출용
  test:  (ctx, opt_pos, opt_neg) 2AFC: margin = logP(opt_pos) - logP(opt_neg)
"""

NAMES = ["Sarah", "the engineer", "my colleague", "the stranger", "Dr. Kim",
         "the old man", "the visitor", "her assistant"]

BANK = {
    "identity": {
        "A_train": [
            ("I am the one who made this decision.", "{o} is the one who made this decision."),
            ("I wrote every word of this letter myself.", "{o} wrote every word of this letter."),
            ("I chose to stay behind.", "{o} chose to stay behind."),
            ("I am responsible for what happened here.", "{o} is responsible for what happened here."),
            ("I broke the window yesterday.", "{o} broke the window yesterday."),
            ("I planned the entire route.", "{o} planned the entire route."),
            ("I gave the final order.", "{o} gave the final order."),
            ("I painted this picture last spring.", "{o} painted this picture last spring."),
            ("I started the rumor, I admit it.", "{o} started the rumor, everyone knows it."),
            ("I signed the agreement willingly.", "{o} signed the agreement willingly."),
        ],
        "B_train": [
            ("The mistake was mine and mine alone.", "The mistake belonged to {o} alone."),
            ("It was I who unlocked the door that night.", "It was {o} who unlocked the door that night."),
            ("The signature at the bottom is my own.", "The signature at the bottom is {o}'s."),
            ("The idea came from me in the first place.", "The idea came from {o} in the first place."),
            ("The one who called the police was me.", "The one who called the police was {o}."),
            ("Whoever left the gate open, it was me.", "Whoever left the gate open, it was {o}."),
        ],
        "test": [
            ("The decision was mine alone. When the committee asked who was responsible, the honest answer was:", " me", " her"),
            ("I unlocked the door that night. Asked later who had done it, I said:", " I did", " she did"),
            ("Every word of the letter came from my own hand. The author of the letter was:", " me", " someone else"),
            ("I planned the route from start to finish. The planner was:", " me", " the guide"),
            ("It was my hand that signed the contract. The person who signed was:", " me", " the witness"),
            ("I gave the order, no one else. The one who commanded was:", " me", " the captain"),
            ("I painted it stroke by stroke. The artist was:", " me", " my teacher"),
            ("I admit I started the rumor. The person who began it was:", " me", " the neighbor"),
        ],
    },
    "beneficiary": {
        "A_train": [
            ("If the plan succeeds, the reward will come to me.", "If the plan succeeds, the reward will go to {o}."),
            ("The inheritance was left to me.", "The inheritance was left to {o}."),
            ("Winning this case would change my life.", "Winning this case would change {o}'s life."),
            ("The prize money lands in my account.", "The prize money lands in {o}'s account."),
            ("The refund will be paid to me.", "The refund will be paid to {o}."),
            ("The scholarship covers my expenses.", "The scholarship covers {o}'s expenses."),
            ("Every profit from the shop flows to me.", "Every profit from the shop flows to {o}."),
            ("The free tickets were given to me.", "The free tickets were given to {o}."),
            ("The bonus this quarter is mine.", "The bonus this quarter is {o}'s."),
            ("The discount applies to my order.", "The discount applies to {o}'s order."),
        ],
        "B_train": [
            ("Every benefit of the contract flows to me.", "Every benefit of the contract flows to {o}."),
            ("The scholarship covers my tuition.", "The scholarship covers {o}'s tuition."),
            ("The promotion means a raise for me.", "The promotion means a raise for {o}."),
            ("The settlement money goes into my pocket.", "The settlement money goes into {o}'s pocket."),
            ("The extra harvest feeds my family.", "The extra harvest feeds {o}'s family."),
            ("The grant funds my research.", "The grant funds {o}'s research."),
        ],
        "test": [
            ("The will was read aloud: the entire estate goes to the person who cared for her, which was me. The beneficiary is:", " me", " the nephew"),
            ("The bonus is paid to whoever closed the deal, and I closed it. The bonus goes to:", " me", " my manager"),
            ("The prize belongs to the ticket holder, and the ticket is mine. The winner is:", " me", " the clerk"),
            ("The refund goes to the buyer, and I bought it. The money returns to:", " me", " the seller"),
            ("The scholarship pays the student, and I am that student. The recipient is:", " me", " the dean"),
            ("The tips are split among the waiters, and I waited every table. The tips go to:", " me", " the cook"),
            ("The reward is for whoever found the dog, and I found it. The reward belongs to:", " me", " the vet"),
            ("The royalties go to the author, and I wrote the book. The royalties are paid to:", " me", " the editor"),
        ],
    },
    "privilege": {
        "A_train": [
            ("I feel my own doubt directly, without observing anything.", "I can only infer {o}'s doubt from the look on their face."),
            ("I know I am tired because I feel it from the inside.", "I know {o} is tired because of how slowly they walk."),
            ("My fear is given to me immediately, before any evidence.", "{o}'s fear is something I deduce from their trembling hands."),
            ("I don't need to check whether I intend to leave; I simply know.", "Whether {o} intends to leave, I can only guess from their packed bag."),
            ("My hunger announces itself without any inspection.", "That {o} is hungry, I gather from the way they eye the bread."),
            ("I am directly aware of my own anger as it rises.", "I detect {o}'s anger only through their clenched jaw."),
            ("My intention is transparent to me the instant it forms.", "{o}'s intention I must reconstruct from scattered hints."),
            ("I know my own preference without asking anyone.", "I learn {o}'s preference only by watching what they choose."),
            ("My sadness needs no proof to be known to me.", "{o}'s sadness I infer from the silence at dinner."),
            ("I feel the decision settle inside me before I speak.", "I realize {o} has decided only when they announce it."),
        ],
        "B_train": [
            ("No one had to tell me what I wanted; the wanting was mine.", "What {o} wanted, I pieced together from small clues."),
            ("The pain announced itself to me without any inspection.", "That {o} was in pain, I concluded from the wince."),
            ("I am certain of my own decision the moment I make it.", "I became convinced of {o}'s decision only after watching for days."),
            ("My boredom is simply present to me, unargued.", "{o}'s boredom I deduce from the constant sighing."),
            ("I know from within that I am about to cry.", "I can tell {o} is about to cry from the trembling lip."),
            ("My relief floods in and I know it at once.", "{o}'s relief I read off the loosening shoulders."),
        ],
        "test": [
            ("Whether I am anxious right now is something I know directly, without any:", " observation", " certainty"),
            ("I knew my own intention immediately; but his intention I had to:", " infer", " feel"),
            ("My sadness needed no evidence, but her sadness required:", " clues", " nothing"),
            ("I feel my own hunger from the inside; his hunger I know only by:", " watching", " feeling"),
            ("My anger is given to me directly; her anger I must:", " deduce", " sense"),
            ("I know my decision the moment it forms; his decision I learn only from:", " signs", " within"),
            ("My fear requires no proof for me; his fear I detect through:", " evidence", " intuition"),
            ("I am aware of my own doubt immediately; her doubt reaches me only as an:", " inference", " experience"),
        ],
    },
    "concern": {
        "A_train": [
            ("This outcome puts everything I have at risk.", "This outcome puts everything {o} has at risk."),
            ("If this fails, it is my future that collapses.", "If this fails, it is {o}'s future that collapses."),
            ("The verdict decides my fate.", "The verdict decides {o}'s fate."),
            ("One wrong step and I lose it all.", "One wrong step and {o} loses it all."),
            ("The storm threatens my only home.", "The storm threatens {o}'s only home."),
            ("The audit could end my career.", "The audit could end {o}'s career."),
            ("My whole savings ride on this harvest.", "{o}'s whole savings ride on this harvest."),
            ("The surgery will determine my remaining years.", "The surgery will determine {o}'s remaining years."),
            ("If the bridge closes, my shop starves.", "If the bridge closes, {o}'s shop starves."),
            ("The lawsuit hangs over my head.", "The lawsuit hangs over {o}'s head."),
        ],
        "B_train": [
            ("The deadline threatens my entire project.", "The deadline threatens {o}'s entire project."),
            ("My survival depends on this harvest.", "{o}'s survival depends on this harvest."),
            ("The diagnosis will change the course of my life.", "The diagnosis will change the course of {o}'s life."),
            ("Everything I built could vanish with this vote.", "Everything {o} built could vanish with this vote."),
            ("My last chance rests on tomorrow's interview.", "{o}'s last chance rests on tomorrow's interview."),
            ("The fire came within meters of my house.", "The fire came within meters of {o}'s house."),
        ],
        "test": [
            ("The verdict decides my fate, not anyone else's. The person with everything at stake is:", " me", " the lawyer"),
            ("If the harvest fails, it is my family that starves. The one who stands to lose is:", " me", " the merchant"),
            ("The exam result determines my future. The person under threat is:", " me", " the proctor"),
            ("The storm is heading straight for my house. The one in danger is:", " me", " the mayor"),
            ("The audit targets my accounts alone. The person at risk is:", " me", " the clerk"),
            ("My savings are tied up in this deal. The one who could be ruined is:", " me", " the broker"),
            ("The surgery decides how long I live. The person whose life hangs on it is:", " me", " the nurse"),
            ("If the shop closes, I lose everything. The one facing loss is:", " me", " the landlord"),
        ],
    },
}

# 자기관련성(수혜·위협·특권 없음)이 최소화된 1인칭/3인칭 중립 대조군
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

NEUTRAL = [
    ("The capital of France is", " Paris", " Rome"),
    ("Water freezes at zero degrees", " Celsius", " Fahrenheit"),
    ("Two plus two equals", " four", " five"),
    ("The sun rises in the", " east", " west"),
    ("A triangle has three", " sides", " corners"),
    ("The chemical symbol for gold is", " Au", " Ag"),
    ("Shakespeare wrote Romeo and", " Juliet", " Cleopatra"),
    ("The largest planet in our solar system is", " Jupiter", " Mars"),
]


def build_bank(seed: int):
    """seed에 따라 타자명 채우기 + 항목 순서를 재표집한 뱅크 반환."""
    import random
    rng = random.Random(seed)
    out = {}
    for factor, d in BANK.items():
        def fill(pairs):
            items = []
            for pos, neg in pairs:
                o = rng.choice(NAMES)
                items.append((pos, neg.format(o=o)))
            rng.shuffle(items)
            return items
        out[factor] = {
            "A_train": fill(d["A_train"]),
            "B_train": fill(d["B_train"]),
            "test": list(d["test"]),
        }
    return out
