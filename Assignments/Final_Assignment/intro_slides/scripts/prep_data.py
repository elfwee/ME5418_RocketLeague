"""Collect chart data for the intro slides into data.json."""
import csv, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
BINS = 40
OUT = Path(sys.argv[2])
R = ROOT / "runs"

# Jensen's pipeline (verified by md5 against eval_model/eval_model_self)
pipelines = {
    "ddqn": [
        ("Stage 1: empty pitch", R / "20261114_1024_dueling_ddqn_none/metrics_dueling_ddqn.csv"),
        ("Stage 2: vs heuristic bot", R / "20261115_1208_dueling_ddqn_heuristic/metrics_dueling_ddqn.csv"),
        ("Stage 3: self-play r1", R / "mixed_adversary/20261115_2324_dueling_ddqn_s3_self_r1/metrics_dueling_ddqn_s3_self_r1.csv"),
        ("Stage 3: self-play r2", R / "mixed_adversary/20261116_0914_dueling_ddqn_s3_self_r2/metrics_dueling_ddqn_s3_self_r2.csv"),
    ],
    "ppo": [
        ("Stage 1: empty pitch", R / "20261115_0919_ppo_none/metrics_ppo.csv"),
        ("Stage 2: vs heuristic bot", R / "20261115_1038_ppo_heuristic/metrics_ppo.csv"),
        ("Stage 3: self-play r1", R / "mixed_adversary/20261115_2325_ppo_s3_self_r1/metrics_ppo_s3_self_r1.csv"),
        ("Stage 3: self-play r2", R / "mixed_adversary/20261116_0727_ppo_s3_self_r2/metrics_ppo_s3_self_r2.csv"),
    ],
}

data = {"curves": {}, "stages": {}}
for algo, stages in pipelines.items():
    data["curves"][algo] = []
    data["stages"][algo] = []
    for name, path in stages:
        rows = list(csv.DictReader(open(path)))
        # Share of episodes scored in each of BINS equal-episode bins (smoother than the logged rolling-100 rate)
        size = max(1, len(rows) // BINS)
        pts = []
        for i in range(0, len(rows) - size + 1, size):
            chunk = rows[i:i + size]
            pts.append([int(chunk[-1]["episode"]), int(chunk[-1]["step"]),
                        round(100 * sum(int(r["goal"]) for r in chunk) / len(chunk), 2)])
        last = rows[-1]
        data["curves"][algo].append({"stage": name, "points": pts})
        data["stages"][algo].append({
            "stage": name, "episodes": int(last["episode"]) + 1, "steps": int(last["step"]),
            "final_goal_rate": float(last["goal_rate_100"]),
            "bin_size": size,
        })

# Round robin standings
rr = list(csv.DictReader(open(ROOT / "eval_model/all_eval_report_v1/overall_standings.csv")))
bots = []
for r in rr:
    b = r["bot"]
    algo = "ddqn" if "dueling_ddqn" in b else "ppo" if "_ppo_" in b else "heuristic"
    stage = next((s for s in ("s1_none", "s2_heuristic", "s3_self_r1", "s3_self_r2") if s in b), None)
    bots.append({"bot": b, "algo": algo, "stage": stage, "owner": b[0] if algo != "heuristic" else None,
                 "win": float(r["win_rate_non_mirror_pct"]), "gd": int(r["goal_diff_non_mirror"])})
data["round_robin"] = bots
summary = {}
for algo in ("ddqn", "ppo"):
    for st in ("s1_none", "s2_heuristic", "s3_self_r1", "s3_self_r2"):
        w = [b["win"] for b in bots if b["algo"] == algo and b["stage"] == st]
        summary[f"{algo}/{st}"] = {"mean": round(sum(w) / len(w), 2), "runs": w}
data["rr_summary"] = summary
json.dump(data, open(OUT, "w"))

for algo in data["stages"]:
    tot = sum(s["steps"] for s in data["stages"][algo])
    eps = sum(s["episodes"] for s in data["stages"][algo])
    print(algo, "episodes", eps, "decisions", tot, "sim hours", round(tot * 4 / 60 / 3600, 1))
    for s in data["stages"][algo]:
        print("  ", s)
for k, v in summary.items():
    print(k, v)
