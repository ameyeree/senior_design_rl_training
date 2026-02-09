import argparse
import os
import numpy as np
import torch
import pybullet as p
import pybullet_data

from config import (
    URDF_PATH,
    CONTROL_JOINT_INDICES,
    EE_LINK_INDEX,
    concept_to_urdf,
)
from train_nn_ik import IKMLP, _standardize, _destandardize


def _apply_q(robot_id: int, q: np.ndarray) -> None:
    for i, j in enumerate(CONTROL_JOINT_INDICES):
        p.resetJointState(robot_id, j, float(q[i]))


def _ee_pos(robot_id: int) -> np.ndarray:
    ls = p.getLinkState(robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
    return np.array(ls[0], dtype=np.float32)


def evaluate(model_path: str, scaler_path: str, data_path: str, num_samples: int, gui: bool) -> None:
    data = np.load(data_path)
    positions = data["positions"].astype(np.float32)
    thetas = data["thetas"].astype(np.float32)

    if num_samples > 0 and num_samples < len(positions):
        idx = np.random.choice(len(positions), size=num_samples, replace=False)
        positions = positions[idx]
        thetas = thetas[idx]

    scaler = np.load(scaler_path)
    pos_mean = scaler["pos_mean"]
    pos_std = scaler["pos_std"]
    theta_mean = scaler["theta_mean"]
    theta_std = scaler["theta_std"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IKMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = _standardize(positions, pos_mean, pos_std)
    x_t = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred_n = model(x_t).cpu().numpy()

    pred_theta = _destandardize(pred_n, theta_mean, theta_std)

    joint_err = np.linalg.norm(pred_theta - thetas, axis=1)
    print(f"Joint error | mean={joint_err.mean():.6f} | median={np.median(joint_err):.6f}")

    # FK error using PyBullet
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

    pred_pos = np.zeros_like(positions)
    for i, th in enumerate(pred_theta):
        q = concept_to_urdf(th)
        _apply_q(robot_id, q)
        p.stepSimulation()
        pred_pos[i] = _ee_pos(robot_id)

    p.disconnect(client)

    pos_err = np.linalg.norm(pred_pos - positions, axis=1)
    print(f"Position error | mean={pos_err.mean():.6f} m | median={np.median(pos_err):.6f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NN IK model.")
    parser.add_argument("--model", type=str, default=os.path.join("nn_ik", "models", "ik_mlp.pt"))
    parser.add_argument("--scaler", type=str, default=os.path.join("nn_ik", "models", "ik_scaler.npz"))
    parser.add_argument("--data", type=str, default=os.path.join("nn_ik", "data", "fk_dataset.npz"))
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        scaler_path=args.scaler,
        data_path=args.data,
        num_samples=args.num_samples,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
