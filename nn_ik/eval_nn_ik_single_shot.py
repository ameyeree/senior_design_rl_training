import argparse
import csv
import os
import numpy as np
import torch
import pybullet as p
import pybullet_data

from config import (
    URDF_PATH,
    CONTROL_JOINT_INDICES,
    EE_LINK_INDEX,
    THETA_LOW,
    THETA_HIGH,
    PCTL_LOW,
    PCTL_HIGH,
    reachable_points_path,
    concept_to_urdf,
)
from train_nn_ik import IKMLP, _standardize, _destandardize


def _apply_q(robot_id: int, q: np.ndarray) -> None:
    for i, j in enumerate(CONTROL_JOINT_INDICES):
        p.resetJointState(robot_id, j, float(q[i]))


def _ee_pos(robot_id: int) -> np.ndarray:
    ls = p.getLinkState(robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
    return np.array(ls[0], dtype=np.float32)


def _load_reachable_points() -> np.ndarray:
    pts = np.load(reachable_points_path()).astype(np.float32)
    lo = np.percentile(pts, PCTL_LOW, axis=0)
    hi = np.percentile(pts, PCTL_HIGH, axis=0)
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[mask]


def evaluate(
    model_path: str,
    scaler_path: str,
    log_path: str,
    num_episodes: int,
    seed: int,
    success_threshold: float,
    gui: bool,
) -> None:
    rng = np.random.default_rng(seed)

    scaler = np.load(scaler_path)
    pos_mean = scaler["pos_mean"]
    pos_std = scaler["pos_std"]
    theta_mean = scaler["theta_mean"]
    theta_std = scaler["theta_std"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IKMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    targets = _load_reachable_points()

    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "target_x",
                "target_y",
                "target_z",
                "pred_theta0",
                "pred_theta1",
                "pred_theta2",
                "pred_ee_x",
                "pred_ee_y",
                "pred_ee_z",
                "distance",
                "success",
            ]
        )

        successes = 0
        distances = []

        for ep in range(1, num_episodes + 1):
            target = targets[rng.integers(0, len(targets))]

            x = _standardize(target[None, :], pos_mean, pos_std)
            x_t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                pred_n = model(x_t).cpu().numpy()[0]

            pred_theta = _destandardize(pred_n, theta_mean, theta_std)
            pred_theta = np.clip(pred_theta, THETA_LOW, THETA_HIGH)

            q = concept_to_urdf(pred_theta)
            _apply_q(robot_id, q)
            p.stepSimulation()
            pred_pos = _ee_pos(robot_id)

            dist = float(np.linalg.norm(pred_pos - target))
            success = dist < success_threshold
            successes += int(success)
            distances.append(dist)

            writer.writerow(
                [
                    ep,
                    target[0],
                    target[1],
                    target[2],
                    pred_theta[0],
                    pred_theta[1],
                    pred_theta[2],
                    pred_pos[0],
                    pred_pos[1],
                    pred_pos[2],
                    dist,
                    int(success),
                ]
            )

            if ep % 10 == 0:
                print(f"Episode {ep}/{num_episodes} | dist={dist:.4f} | success={success}")

    p.disconnect(client)

    if distances:
        print("=== NN IK Evaluation Summary ===")
        print(f"Episodes: {num_episodes}")
        print(f"Successes: {successes}")
        print(
            f"Final distance (min/mean/max): "
            f"{np.min(distances):.4f} / {np.mean(distances):.4f} / {np.max(distances):.4f}"
        )
    print(f"Eval log saved to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NN IK (single-shot) on random targets.")
    parser.add_argument("--model", type=str, default=os.path.join("nn_ik", "models", "ik_mlp.pt"))
    parser.add_argument("--scaler", type=str, default=os.path.join("nn_ik", "models", "ik_scaler.npz"))
    parser.add_argument("--log", type=str, default=os.path.join("logs_nn_ik", "eval_log.csv"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        scaler_path=args.scaler,
        log_path=args.log,
        num_episodes=args.episodes,
        seed=args.seed,
        success_threshold=args.success_threshold,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
