"""V19 대상 모델 자동 다운로드 — base·safetensors·dense·텍스트 후보를 하나씩 시도해서
받아지는 것만 HF 캐시에 내려받는다. 게이트/없는이름/safetensors없음/instruct는 건너뜀.

사용:
  python fetch_models.py            # 후보 전체 시도, 되는 것만 다운로드 → v19에 반영할 목록 출력
  python fetch_models.py --check    # 다운로드 없이 '받을 수 있나'만 확인
  python fetch_models.py --only Qwen/Qwen2.5-7B EleutherAI/pythia-1.4b   # 특정 것만
"""
from __future__ import annotations
import argparse, sys, json, os

# 후보: (repo, tag, 예상크기). 전부 base·dense·텍스트 지향. 실재/safetensors는 코드가 검증.
CANDIDATES = [
    ("Qwen/Qwen2.5-1.5B",        "qwen25_1p5b"),   # V17/V18 기준(고정)
    ("Qwen/Qwen2.5-0.5B",        "qwen25_0p5b"),
    ("Qwen/Qwen2.5-3B",          "qwen25_3b"),
    ("Qwen/Qwen2.5-7B",          "qwen25_7b"),
    ("Qwen/Qwen3-1.7B-Base",     "qwen3_1p7b"),    # 세대축(존재하면 받음)
    ("Qwen/Qwen3-4B-Base",       "qwen3_4b"),
    ("Qwen/Qwen3-8B-Base",       "qwen3_8b"),
    ("EleutherAI/pythia-1.4b",   "pythia_1p4b"),   # 랩축, 게이트 없음
    ("EleutherAI/pythia-6.9b",   "pythia_6p9b"),
    ("allenai/OLMo-2-1124-7B",   "olmo2_7b"),      # 완전공개
    ("mistralai/Mistral-7B-v0.3","mistral_7b"),
    ("meta-llama/Llama-3.1-8B",  "llama31_8b"),    # 게이트(토큰) 가능
    ("google/gemma-2-9b",        "gemma2_9b"),     # 게이트(토큰) 가능
    ("microsoft/phi-2",          "phi2"),
]

BAD = ("instruct", "-it", "chat", "-vl", "reasoning")  # 이름 안전장치(후보엔 없지만 방어)


def has_safetensors(info):
    return any(s.rfilename.endswith(".safetensors") for s in info.siblings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="다운로드 없이 가능여부만")
    ap.add_argument("--only", nargs="+", default=None, help="특정 repo만")
    args = ap.parse_args()

    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError:
        print("huggingface_hub 없음:  pip install huggingface_hub --break-system-packages")
        sys.exit(1)

    cands = CANDIDATES
    if args.only:
        cands = [(r, r.split("/")[-1].lower().replace(".", "p").replace("-", "_")) for r in args.only]

    ok, skipped = [], []
    for repo, tag in cands:
        if any(b in repo.lower() for b in BAD):
            skipped.append((repo, "instruct/vl 계열 — 제외")); continue
        try:
            info = model_info(repo)
        except Exception as e:
            skipped.append((repo, f"없음/접근불가: {str(e)[:50]}")); continue
        if not has_safetensors(info):
            skipped.append((repo, "safetensors 없음(GGUF전용 등)")); continue
        gated = getattr(info, "gated", False)
        if gated:
            # 토큰 있으면 시도, 없으면 스킵 안내
            if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
                skipped.append((repo, "gated — HF 토큰 필요(huggingface-cli login)")); continue
        if args.check:
            ok.append((repo, tag, "받을 수 있음")); print(f"OK   {repo}  (tag={tag}, gated={gated})")
            continue
        try:
            print(f"↓ 다운로드: {repo} …", flush=True)
            snapshot_download(repo, allow_patterns=["*.safetensors", "*.json", "*.txt",
                                                     "tokenizer*", "*.model"])
            ok.append((repo, tag, "다운로드 완료")); print(f"✓ 완료: {repo}")
        except Exception as e:
            skipped.append((repo, f"다운로드 실패: {str(e)[:60]}"))

    print("\n" + "=" * 60)
    print(f"받을 수 있는/받은 모델 {len(ok)}개:")
    for repo, tag, st in ok:
        print(f"  ✓ {repo:32s} tag={tag:14s} [{st}]")
    if skipped:
        print(f"\n건너뜀 {len(skipped)}개:")
        for repo, why in skipped:
            print(f"  · {repo:32s} {why}")
    # spec models 블록 자동 생성
    print("\n--- 아래를 spec.yaml의 models: 로 쓰면 됨 (성공분만) ---")
    print("models:")
    for repo, tag, _ in ok:
        print(f'  - {{name: "{repo}", tag: {tag}}}')
    json.dump([{"name": r, "tag": t} for r, t, _ in ok],
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_available.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n(목록도 저장: experiments/v19/models_available.json)")


if __name__ == "__main__":
    main()
