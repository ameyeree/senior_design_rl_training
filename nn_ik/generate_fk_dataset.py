import argparse
import os
import csv
import numpy as np
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
    concept_to_urdf,
)


def _apply_q(robot_id: int, q: np.ndarray) -> None:
    for i, j in enumerate(CONTROL_JOINT_INDICES):
        p.resetJointState(robot_id, j, float(q[i]))


def _ee_pos(robot_id: int) -> np.ndarray:
    ls = p.getLinkState(robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
    return np.array(ls[0], dtype=np.float32)


def generate_dataset(
    num_samples: int,
    seed: int,
    gui: bool,
    out_path: str,
    csv_path: str | None,
    log_every: int,
) -> None:
    rng = np.random.default_rng(seed)

    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

    thetas = np.zeros((num_samples, 3), dtype=np.float32)
    positions = np.zeros((num_samples, 3), dtype=np.float32)

    for k in range(num_samples):
        theta = rng.uniform(THETA_LOW, THETA_HIGH).astype(np.float32)
        q = concept_to_urdf(theta)
        _apply_q(robot_id, q)
        p.stepSimulation()
        pos = _ee_pos(robot_id)

        thetas[k] = theta
        positions[k] = pos

        if log_every > 0 and (k + 1) % log_every == 0:
            print(f"Generated {k + 1}/{num_samples} samples")

    # Trim outliers to match environment sampling
    lo = np.percentile(positions, PCTL_LOW, axis=0)
    hi = np.percentile(positions, PCTL_HIGH, axis=0)
    mask = np.all((positions >= lo) & (positions <= hi), axis=1)

    positions = positions[mask]
    thetas = thetas[mask]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, positions=positions, thetas=thetas)
    print(f"Saved dataset: {out_path}")
    print(f"Kept {len(positions)} / {num_samples} samples after trimming")

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "theta0", "theta1", "theta2"])
            for pos, th in zip(positions, thetas):
                writer.writerow([pos[0], pos[1], pos[2], th[0], th[1], th[2]])
        print(f"Saved CSV dataset: {csv_path}")

    p.disconnect(client)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FK dataset in PyBullet.")
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", type=str, default=os.path.join("nn_ik", "data", "fk_dataset.npz"))
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--log-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv else None
    generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        gui=args.gui,
        out_path=args.out,
        csv_path=csv_path,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
