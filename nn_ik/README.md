# Neural-Network IK (Dobot)

This folder generates forward-kinematics (FK) data in PyBullet and trains a neural network to predict conceptual joint angles from end-effector position.

## Why this matches the SAC setup
- Uses the same Dobot URDF as the velocity-control environment.
- Uses the same conceptual joint limits and the same EE link.
- Samples joint angles within the restricted workspace (percentile-filtered positions).

## Quick start

1) Generate dataset (headless, faster):

- Default output: nn_ik/data/fk_dataset.npz

2) Train the IK model:

- Default output: nn_ik/models/ik_mlp.pt

3) Evaluate model (reports joint and position error):

## Files
- generate_fk_dataset.py: FK data generator in PyBullet.
- train_nn_ik.py: MLP training with normalization and CSV logging.
- eval_nn_ik.py: quick evaluation of joint and position errors.
- eval_nn_ik_single_shot.py: single-shot evaluator that logs final distance per episode.
- compare_sac_vs_nn_ik.py: comparison plots and summary CSV (SAC vs NN IK).
- config.py: shared robot config and kinematic mapping.

## Notes
- Input is position-only (x, y, z), which matches the SAC environment’s position-based reward.
- If you later need orientation or multi-solution disambiguation, see `train_nn_ik.py` for extension notes.
