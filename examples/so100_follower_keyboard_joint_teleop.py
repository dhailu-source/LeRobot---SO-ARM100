#!/usr/bin/env python

import argparse
import time
from dataclasses import dataclass

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


@dataclass
class KeyState:
    pressed: set[str]


def _deg_to_norm(deg: float) -> float:
    # SO100Follower defaults to MotorNormMode.RANGE_M100_100 for body joints unless use_degrees=True.
    # For a first working teleop, we run in use_degrees=True so increments are intuitive.
    return deg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard joint teleop for a single SO-100 follower arm (no IK/URDF needed)."
    )
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id (used for calibration file).")
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    parser.add_argument(
        "--step_deg",
        type=float,
        default=2.0,
        help="Joint increment per tick, in degrees (use_degrees=True).",
    )
    parser.add_argument(
        "--gripper_step",
        type=float,
        default=5.0,
        help="Gripper increment per tick, in [0..100].",
    )
    parser.add_argument(
        "--no_calib_clamp",
        action="store_true",
        help="Disable clamping joint targets to the calibration min/max ranges.",
    )
    parser.add_argument(
        "--calib_margin_ratio",
        type=float,
        default=0.0,
        help=(
            "Tighter safety margin inside calibrated limits for body joints. "
            "Example: 0.1 keeps 80% of the calibrated range (shrinks by 10% on each side)."
        ),
    )
    args = parser.parse_args()

    try:
        from pynput import keyboard  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pynput is required for keyboard teleop. Install dependencies and try again."
        ) from e

    key_state = KeyState(pressed=set())

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
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

    help_text = """
SO-100 follower keyboard teleop (joint space)

Quit:
  - ESC

Joint controls (hold keys):
  shoulder_pan   : a / d
  shoulder_lift  : w / s
  elbow_flex     : q / e
  wrist_flex     : i / k
  wrist_roll     : j / l

Gripper (hold keys):
  open / close   : o / u
"""
    print(help_text)

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect()

    # Seed targets from current state
    obs = robot.get_observation()
    targets = {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}

    dt = 1.0 / float(args.fps)
    step = _deg_to_norm(float(args.step_deg))
    gstep = float(args.gripper_step)

    # In DEGREES norm mode, we convert the motor calibration raw range into a degree range
    # so we can clamp the keyboard targets to safe min/max.
    #
    # This prevents commanding angles outside the calibrated range of motion.
    deg_bounds: dict[str, tuple[float, float]] = {}
    if not args.no_calib_clamp:
        margin_ratio = float(args.calib_margin_ratio)
        if margin_ratio < 0 or margin_ratio >= 0.5:
            raise ValueError("--calib_margin_ratio must be in [0.0, 0.5).")

        for motor_name, cal in robot.calibration.items():
            if motor_name == "gripper":
                continue  # gripper is RANGE_0_100 and already clamped below
            model = robot.bus.motors[motor_name].model
            max_res = robot.bus.model_resolution_table[model] - 1
            mid = (cal.range_min + cal.range_max) / 2
            deg_min = (cal.range_min - mid) * 360 / max_res
            deg_max = (cal.range_max - mid) * 360 / max_res
            lo, hi = (min(deg_min, deg_max), max(deg_min, deg_max))
            if margin_ratio > 0:
                span = hi - lo
                lo = lo + span * margin_ratio
                hi = hi - span * margin_ratio
            deg_bounds[f"{motor_name}.pos"] = (float(lo), float(hi))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        while listener.is_alive():
            t0 = time.perf_counter()

            # Update targets based on currently pressed keys
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

            # Clamp gripper to [0, 100] (body joints are left unclamped; the motor calibration enforces bounds)
            if "gripper.pos" in targets:
                targets["gripper.pos"] = float(max(0.0, min(100.0, targets["gripper.pos"])))

            # Clamp body joints to calibration degree bounds
            if deg_bounds:
                for k, (lo, hi) in deg_bounds.items():
                    if k in targets:
                        targets[k] = float(max(lo, min(hi, targets[k])))

            robot.send_action(targets)

            sleep_s = dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            if listener.is_alive():
                listener.stop()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()

