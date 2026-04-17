#!/usr/bin/env python

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility


@dataclass
class KeyState:
    pressed: set[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a local LeRobot dataset by controlling one SO-100 follower with keyboard."
    )
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id (calibration id).")
    parser.add_argument("--fps", type=float, default=30.0, help="Control + record loop rate.")
    parser.add_argument("--step_deg", type=float, default=2.0, help="Body joint step in degrees per tick.")
    parser.add_argument("--gripper_step", type=float, default=5.0, help="Gripper step in [0..100] per tick.")
    parser.add_argument("--task", default="manual keyboard teleop", help="Task string saved in dataset frames.")
    parser.add_argument(
        "--repo_id",
        default="local/so100_follower_keyboard",
        help="Dataset repo_id metadata (can be local namespace).",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./outputs/so100_follower_keyboard_dataset",
        help="Directory where dataset files are written.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Stop automatically after this many saved episodes (0 = unlimited).",
    )
    parser.add_argument(
        "--calib_margin_ratio",
        type=float,
        default=0.0,
        help="Safety margin inside calibrated body-joint limits in [0.0, 0.5).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new episodes to an existing dataset at --dataset_dir (same --repo_id as when it was created).",
    )
    args = parser.parse_args()

    if args.calib_margin_ratio < 0 or args.calib_margin_ratio >= 0.5:
        raise ValueError("--calib_margin_ratio must be in [0.0, 0.5).")

    try:
        from pynput import keyboard  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pynput is required for keyboard control.") from e

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    key_state = KeyState(pressed=set())
    saved_episodes = 0
    should_exit = False

    def on_press(key):
        nonlocal should_exit
        try:
            if key == keyboard.Key.esc:
                should_exit = True
                return False
            if hasattr(key, "char") and key.char:
                key_state.pressed.add(key.char.lower())
        except Exception:
            return

    def on_release(key):
        try:
            if hasattr(key, "char") and key.char:
                key_state.pressed.discard(key.char.lower())
        except Exception:
            return

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect()

    # Build dataset features from robot IO specs (state/action only, no cameras).
    ds_features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
    )

    prior_eps = 0
    if args.resume:
        info_path = dataset_dir / "meta" / "info.json"
        if not dataset_dir.is_dir() or not info_path.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing LeRobot dataset directory with meta/info.json: {dataset_dir}"
            )
        dataset = LeRobotDataset(args.repo_id, root=dataset_dir, batch_encoding_size=1)
        sanity_check_dataset_robot_compatibility(dataset, robot, int(args.fps), ds_features)
        prior_eps = dataset.meta.total_episodes
    else:
        if dataset_dir.exists():
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_dir}\n"
                "Use a new --dataset_dir path, or pass --resume to append episodes."
            )
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=int(args.fps),
            features=ds_features,
            root=dataset_dir,
            robot_type=robot.name,
            use_videos=False,
        )

    obs = robot.get_observation()
    targets = {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}

    # Convert calibration raw ranges into degree bounds for safer keyboard control.
    deg_bounds: dict[str, tuple[float, float]] = {}
    for motor_name, cal in robot.calibration.items():
        if motor_name == "gripper":
            continue
        model = robot.bus.motors[motor_name].model
        max_res = robot.bus.model_resolution_table[model] - 1
        mid = (cal.range_min + cal.range_max) / 2
        deg_min = (cal.range_min - mid) * 360 / max_res
        deg_max = (cal.range_max - mid) * 360 / max_res
        lo, hi = (min(deg_min, deg_max), max(deg_min, deg_max))
        if args.calib_margin_ratio > 0:
            span = hi - lo
            lo = lo + span * args.calib_margin_ratio
            hi = hi - span * args.calib_margin_ratio
        deg_bounds[f"{motor_name}.pos"] = (float(lo), float(hi))

    dt = 1.0 / float(args.fps)
    step = float(args.step_deg)
    gstep = float(args.gripper_step)
    prev_pressed: set[str] = set()

    help_text = """
SO-100 keyboard record mode (joint space)

Motion keys (hold):
  shoulder_pan   : a / d
  shoulder_lift  : w / s
  elbow_flex     : q / e
  wrist_flex     : i / k
  wrist_roll     : j / l
  gripper        : u(close) / o(open)

Episode controls:
  z : save current episode
  x : discard current episode
  c : save current episode then exit
  ESC : exit without saving current episode

Run again on the same folder:
  add --resume (keep the same --repo_id as the first run)
"""
    print(help_text)
    print(f"Recording dataset into: {dataset_dir}")
    if args.resume:
        print(f"Resuming: dataset already has {prior_eps} episode(s); new saves will append as episode index {prior_eps} and above.")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        while listener.is_alive() and not should_exit:
            t0 = time.perf_counter()
            p = key_state.pressed

            def bump(name: str, delta: float):
                k = f"{name}.pos"
                if k in targets:
                    targets[k] = float(targets[k] + delta)

            bump("shoulder_pan", (+step if "d" in p else 0.0) + (-step if "a" in p else 0.0))
            bump("shoulder_lift", (+step if "w" in p else 0.0) + (-step if "s" in p else 0.0))
            bump("elbow_flex", (+step if "e" in p else 0.0) + (-step if "q" in p else 0.0))
            bump("wrist_flex", (+step if "i" in p else 0.0) + (-step if "k" in p else 0.0))
            bump("wrist_roll", (+step if "l" in p else 0.0) + (-step if "j" in p else 0.0))
            if "o" in p:
                bump("gripper", +gstep)
            if "u" in p:
                bump("gripper", -gstep)

            if "gripper.pos" in targets:
                targets["gripper.pos"] = float(max(0.0, min(100.0, targets["gripper.pos"])))
            for k, (lo, hi) in deg_bounds.items():
                if k in targets:
                    targets[k] = float(max(lo, min(hi, targets[k])))

            obs = robot.get_observation()
            robot.send_action(targets)

            frame = {
                **build_dataset_frame(dataset.features, obs, OBS_STR),
                **build_dataset_frame(dataset.features, targets, ACTION),
                "task": args.task,
            }
            dataset.add_frame(frame)

            # Edge-triggered controls (once per keypress).
            newly_pressed = p - prev_pressed
            prev_pressed = set(p)

            if "x" in newly_pressed:
                dataset.clear_episode_buffer(delete_images=False)
                print("Discarded current episode buffer.")

            if "z" in newly_pressed or "c" in newly_pressed:
                if dataset.episode_buffer and int(dataset.episode_buffer["size"]) > 0:
                    dataset.save_episode()
                    saved_episodes += 1
                    print(f"Saved episode #{saved_episodes}.")
                else:
                    print("Episode buffer is empty. Nothing to save.")

            if "c" in newly_pressed:
                break

            if args.max_episodes > 0 and saved_episodes >= args.max_episodes:
                print(f"Reached --max_episodes={args.max_episodes}. Exiting.")
                break

            sleep_s = dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            if listener.is_alive():
                listener.stop()
        except Exception:
            pass
        dataset.finalize()
        robot.disconnect()

    print("\nDone.")
    print(f"Saved episodes: {saved_episodes}")
    print("Replay example (episode indices are 0, 1, ... for each saved episode):")
    print(
        "python -m lerobot.scripts.lerobot_replay "
        f"--robot.type=so100_follower --robot.port={args.port} --robot.id={args.id} "
        f"--robot.use_degrees=true "
        f"--dataset.repo_id={args.repo_id} --dataset.root=\"{dataset_dir}\" --dataset.episode=0"
    )


if __name__ == "__main__":
    main()
