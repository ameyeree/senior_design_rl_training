import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


SAC_EVAL_LOG = os.path.join("logs_dobot_sac_vel", "eval_log.csv")
NN_EVAL_LOG = os.path.join("logs_nn_ik", "eval_log.csv")


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV log not found at {path}")

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _final_distances_from_sac(rows: List[Dict[str, str]]) -> np.ndarray:
    final_by_ep: Dict[int, float] = {}
    last_step_by_ep: Dict[int, int] = {}

    for r in rows:
        ep = int(r["episode"])
        step = int(r["step"])
        dist = float(r["distance"])

        if (ep not in last_step_by_ep) or (step > last_step_by_ep[ep]):
            last_step_by_ep[ep] = step
            final_by_ep[ep] = dist

    return np.array(list(final_by_ep.values()), dtype=np.float32)


def _success_rate_sac(rows: List[Dict[str, str]], threshold: float) -> float:
    # Use explicit success column if present; fallback to threshold on final distance
    if "success" in rows[0]:
        vals = [int(r["success"]) for r in rows if r.get("success") is not None]
        if vals:
            return float(np.mean(vals))

    final_dists = _final_distances_from_sac(rows)
    if final_dists.size == 0:
        return 0.0
    return float(np.mean(final_dists < threshold))


def _final_distances_from_nn(rows: List[Dict[str, str]]) -> np.ndarray:
    return np.array([float(r["distance"]) for r in rows], dtype=np.float32)


def _success_rate_nn(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([int(r["success"]) for r in rows]))


def _summary(distances: np.ndarray) -> Dict[str, float]:
    if distances.size == 0:
        return {"min": np.nan, "mean": np.nan, "median": np.nan, "max": np.nan}
    return {
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def compare(
    sac_path: str,
    nn_path: str,
    out_dir: str,
    success_threshold: float,
) -> None:
    sac_rows = _load_csv(sac_path)
    nn_rows = _load_csv(nn_path)

    sac_final = _final_distances_from_sac(sac_rows)
    nn_final = _final_distances_from_nn(nn_rows)

    sac_success = _success_rate_sac(sac_rows, success_threshold)
    nn_success = _success_rate_nn(nn_rows)

    os.makedirs(out_dir, exist_ok=True)

    # 1) Final distance histogram (overlay)
    plt.figure(figsize=(7, 4))
    plt.hist(sac_final, bins=20, alpha=0.6, label="SAC")
    plt.hist(nn_final, bins=20, alpha=0.6, label="NN IK")
    plt.xlabel("Final Distance (m)")
    plt.ylabel("Count")
    plt.title("Eval: Final Distance Histogram (SAC vs NN IK)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_final_distance_hist_compare.png"), dpi=200)
    plt.close()

    # 2) Success rate comparison
    plt.figure(figsize=(4, 4))
    plt.bar(["SAC", "NN IK"], [sac_success, nn_success], color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title("Eval: Success Rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_success_rate_compare.png"), dpi=200)
    plt.close()

    # 3) Boxplot of final distances
    plt.figure(figsize=(5, 4))
    plt.boxplot([sac_final, nn_final], labels=["SAC", "NN IK"], showfliers=False)
    plt.ylabel("Final Distance (m)")
    plt.title("Eval: Final Distance Boxplot")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_final_distance_boxplot.png"), dpi=200)
    plt.close()

    # 4) Summary CSV
    summary_path = os.path.join(out_dir, "eval_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "episodes", "success_rate", "min", "mean", "median", "max"])
        sac_stats = _summary(sac_final)
        nn_stats = _summary(nn_final)
        writer.writerow(
            [
                "SAC",
                int(len(sac_final)),
                sac_success,
                sac_stats["min"],
                sac_stats["mean"],
                sac_stats["median"],
                sac_stats["max"],
            ]
        )
        writer.writerow(
            [
                "NN IK",
                int(len(nn_final)),
                nn_success,
                nn_stats["min"],
                nn_stats["mean"],
                nn_stats["median"],
                nn_stats["max"],
            ]
        )

    print(f"Comparison plots written to {out_dir}")
    print(f"Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAC vs NN IK evaluation logs.")
    parser.add_argument("--sac-log", type=str, default=SAC_EVAL_LOG)
    parser.add_argument("--nn-log", type=str, default=NN_EVAL_LOG)
    parser.add_argument("--out", type=str, default=os.path.join("logs_compare"))
    parser.add_argument("--success-threshold", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare(
        sac_path=args.sac_log,
        nn_path=args.nn_log,
        out_dir=args.out,
        success_threshold=args.success_threshold,
    )


if __name__ == "__main__":
    main()
