#!/usr/bin/env python

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


BODY_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]


@dataclass
class TrialResult:
    test_name: str
    trial: int
    values: dict[str, Any]


def parse_weights(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(x.strip()) for x in raw.split(",")]


def parse_pose(raw: str) -> dict[str, float]:
    values = [float(x.strip()) for x in raw.split(",")]
    if len(values) != 6:
        raise ValueError("Pose must have 6 comma-separated values: pan,lift,elbow,wflex,wroll,gripper")
    keys = BODY_JOINTS + ["gripper.pos"]
    return {k: v for k, v in zip(keys, values, strict=True)}


def get_joint_vector(obs: dict[str, Any], keys: list[str]) -> list[float]:
    return [float(obs[k]) for k in keys]


def mean_std_per_joint(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(samples)
    dims = len(samples[0])
    means = []
    stds = []
    for j in range(dims):
        vals = [row[j] for row in samples]
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / max(1, n - 1)
        means.append(mu)
        stds.append(var ** 0.5)
    return means, stds


def run_payload_test(
    robot: SO100Follower,
    weights_kg: list[float],
    poses: list[dict[str, float]],
    settle_s: float,
    trial_rows: list[TrialResult],
) -> dict[str, Any]:
    if not weights_kg:
        return {"executed": False, "reason": "no weights provided"}

    max_passed = 0.0
    all_records: list[dict[str, Any]] = []
    for w in weights_kg:
        input(f"\nAttach {w:.2f} kg to end-effector, then press ENTER to continue...")
        passed_this_weight = True
        for idx, pose in enumerate(poses, start=1):
            sent = robot.send_action(pose)
            time.sleep(settle_s)
            obs = robot.get_observation()
            pos = get_joint_vector(obs, BODY_JOINTS)
            currents = robot.bus.sync_read("Present_Current")  # torque proxy
            peak_current = max(abs(float(v)) for v in currents.values())
            rec = {
                "weight_kg": w,
                "pose_idx": idx,
                "sent_action": sent,
                "observed_body_pos": {k: float(obs[k]) for k in BODY_JOINTS},
                "present_current_raw": {k: int(v) for k, v in currents.items()},
                "peak_abs_current_raw": peak_current,
            }
            all_records.append(rec)
            trial_rows.append(
                TrialResult(
                    "payload_current",
                    len(all_records),
                    {
                        "weight_kg": w,
                        "pose_idx": idx,
                        "peak_abs_current_raw": peak_current,
                        "shoulder_pan_pos": pos[0],
                        "shoulder_lift_pos": pos[1],
                        "elbow_flex_pos": pos[2],
                        "wrist_flex_pos": pos[3],
                        "wrist_roll_pos": pos[4],
                    },
                )
            )
            print(
                f"[payload] weight={w:.2f}kg pose#{idx} peak_current_raw={peak_current:.1f} pos={['%.2f' % p for p in pos]}"
            )
        if passed_this_weight:
            max_passed = w

    return {
        "executed": True,
        "max_payload_tested_kg": max_passed,
        "records": all_records,
        "note": "Current is in raw motor units; convert using motor datasheet if needed.",
    }


def run_repeatability_test(
    robot: SO100Follower,
    target_pose: dict[str, float],
    repeats: int,
    settle_s: float,
    trial_rows: list[TrialResult],
) -> dict[str, Any]:
    samples: list[list[float]] = []
    for i in range(repeats):
        robot.send_action(target_pose)
        time.sleep(settle_s)
        obs = robot.get_observation()
        vals = get_joint_vector(obs, BODY_JOINTS)
        samples.append(vals)
        trial_rows.append(
            TrialResult(
                "repeatability",
                i + 1,
                {
                    "shoulder_pan_pos": vals[0],
                    "shoulder_lift_pos": vals[1],
                    "elbow_flex_pos": vals[2],
                    "wrist_flex_pos": vals[3],
                    "wrist_roll_pos": vals[4],
                },
            )
        )

    means, stds = mean_std_per_joint(samples)
    return {
        "executed": True,
        "repeats": repeats,
        "target_pose": target_pose,
        "mean_body_pos": {k: v for k, v in zip(BODY_JOINTS, means, strict=True)},
        "std_body_pos": {k: v for k, v in zip(BODY_JOINTS, stds, strict=True)},
        "samples": samples,
    }


def run_speed_test(
    robot: SO100Follower,
    move_delta_deg: float,
    move_window_s: float,
    sample_dt_s: float,
    trial_rows: list[TrialResult],
) -> dict[str, Any]:
    results = {}
    for joint in BODY_JOINTS:
        obs0 = robot.get_observation()
        start = float(obs0[joint])
        target = {k: float(obs0[k]) for k in BODY_JOINTS + ["gripper.pos"]}
        target[joint] = start + move_delta_deg

        t0 = time.perf_counter()
        robot.send_action(target)
        times = []
        values = []
        while True:
            now = time.perf_counter()
            if now - t0 > move_window_s:
                break
            obs = robot.get_observation()
            times.append(now - t0)
            values.append(float(obs[joint]))
            time.sleep(sample_dt_s)

        peak_vel = 0.0
        for i in range(1, len(values)):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            vel = abs((values[i] - values[i - 1]) / dt)
            peak_vel = max(peak_vel, vel)

        results[joint] = {
            "start_pos": start,
            "target_pos": target[joint],
            "peak_velocity_deg_per_s": peak_vel,
            "n_samples": len(values),
        }
        trial_rows.append(
            TrialResult(
                "speed_peak_velocity",
                len(results),
                {"joint": joint, "peak_velocity_deg_per_s": peak_vel, "n_samples": len(values)},
            )
        )
        print(f"[speed] {joint}: peak_velocity_deg_per_s={peak_vel:.2f} ({len(values)} samples)")

    return {
        "executed": True,
        "move_delta_deg": move_delta_deg,
        "move_window_s": move_window_s,
        "sample_dt_s": sample_dt_s,
        "joint_results": results,
        "note": "Estimated from finite differences of observed positions.",
    }


def save_rows_csv(path: Path, rows: list[TrialResult]) -> None:
    flat_rows = []
    for r in rows:
        base = {"test_name": r.test_name, "trial": r.trial}
        base.update(r.values)
        flat_rows.append(base)
    all_keys = sorted({k for row in flat_rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-100 hardware spec test runner (payload, precision, speed).")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id for calibration lookup.")
    parser.add_argument("--no_calibrate", action="store_true", help="Skip calibration on connect.")
    parser.add_argument("--weights_kg", default="0.5,1.0,1.5,2.0", help="Payload test weights list.")
    parser.add_argument("--payload_settle_s", type=float, default=1.0)
    parser.add_argument("--repeatability_repeats", type=int, default=10)
    parser.add_argument("--repeatability_settle_s", type=float, default=0.5)
    parser.add_argument(
        "--repeatability_target",
        default="0,30,45,15,0,40",
        help="Pose: pan,lift,elbow,wflex,wroll,gripper (degrees and gripper 0-100).",
    )
    parser.add_argument("--speed_delta_deg", type=float, default=20.0, help="Step per joint for speed estimation.")
    parser.add_argument("--speed_window_s", type=float, default=1.0)
    parser.add_argument("--speed_sample_dt_s", type=float, default=0.05)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"so100_specs_{stamp}.json"
    csv_path = output_dir / f"so100_specs_trials_{stamp}.csv"

    payload_poses = [
        parse_pose("0,45,90,0,0,40"),
        parse_pose("30,30,45,0,0,40"),
        parse_pose("-30,30,45,0,0,40"),
    ]
    weights = parse_weights(args.weights_kg)
    repeat_pose = parse_pose(args.repeatability_target)

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    trial_rows: list[TrialResult] = []
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "robot_id": args.id,
        "port": args.port,
        "config": vars(args),
    }

    robot.connect(calibrate=not args.no_calibrate)
    try:
        result["payload_test"] = run_payload_test(
            robot=robot,
            weights_kg=weights,
            poses=payload_poses,
            settle_s=args.payload_settle_s,
            trial_rows=trial_rows,
        )
        result["repeatability_test"] = run_repeatability_test(
            robot=robot,
            target_pose=repeat_pose,
            repeats=args.repeatability_repeats,
            settle_s=args.repeatability_settle_s,
            trial_rows=trial_rows,
        )
        result["speed_test"] = run_speed_test(
            robot=robot,
            move_delta_deg=args.speed_delta_deg,
            move_window_s=args.speed_window_s,
            sample_dt_s=args.speed_sample_dt_s,
            trial_rows=trial_rows,
        )
    finally:
        robot.disconnect()

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    save_rows_csv(csv_path, trial_rows)

    print("\nDone.")
    print(f"- JSON summary: {json_path}")
    print(f"- Trial CSV:    {csv_path}")


if __name__ == "__main__":
    main()
