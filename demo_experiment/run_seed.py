"""데모용 toy 실험. 실제 V16에서는 이 파일을 실제 학습/개입 코드로 교체."""
import json, random, sys, os

seed = int(sys.argv[1])
rng = random.Random(seed)
factors = ["identity", "beneficiary", "privilege", "concern"]
effects = {a: {b: (rng.gauss(0.62, 0.04) if a == b else abs(rng.gauss(0.08, 0.03)))
               for b in factors} for a in factors}
result = {
    "seed": seed, "factors": factors, "effects": effects,
    "controls": {
        "random_direction": abs(rng.gauss(0.03, 0.01)),
        "shuffled_label": abs(rng.gauss(0.02, 0.01)),
        "neutral_task_damage": abs(rng.gauss(0.01, 0.005)),
    },
}
os.makedirs("results", exist_ok=True)
out = f"results/seed_{seed}.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(out)
