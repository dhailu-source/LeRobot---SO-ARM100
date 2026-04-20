#!/usr/bin/env python

import argparse
import time

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Control one SO-100 follower arm with a gamepad.")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id (calibration id).")
    parser.add_argument("--fps", type=float, default=50.0, help="Control loop frequency.")
    parser.add_argument("--max_joint_speed_deg_s", type=float, default=90.0, help="Max body-joint speed.")
    parser.add_argument("--gripper_speed_unit_s", type=float, default=80.0, help="Max gripper speed in [0..100]/s.")
    parser.add_argument("--deadzone", type=float, default=0.12, help="Joystick deadzone.")
    parser.add_argument(
        "--calib_margin_ratio",
        type=float,
        default=0.03,
        help="Safety margin inside calibrated body-joint limits in [0.0, 0.5).",
    )
    args = parser.parse_args()

    if args.calib_margin_ratio < 0 or args.calib_margin_ratio >= 0.5:
        raise ValueError("--calib_margin_ratio must be in [0.0, 0.5).")

    try:
        import pygame
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pygame is required for gamepad control. Install with `pip install -e \".[gamepad]\"`.") from e

    dt = 1.0 / float(args.fps)

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect()

    obs = robot.get_observation()
    targets = {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}

    # Convert calibration raw ranges into degree bounds for safer teleop.
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

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Connect your controller and try again.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Using gamepad: {joystick.get_name()}")

    print(
        """
SO-100 gamepad teleop (joint space):
  Left stick:  shoulder_pan (X), shoulder_lift (Y)
  Right stick: wrist_flex (X), elbow_flex (Y)
  L1 / R1:     wrist_roll - / +
  L2 / R2:     gripper close / open
  Square button:  exit
"""
    )

    def axis_raw(index: int) -> float:
        if index >= joystick.get_numaxes():
            return 0.0
        return float(joystick.get_axis(index))

    # Measure stick center offsets to prevent slow drift when sticks are released.
    center = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    sample_count = 20
    for _ in range(sample_count):
        pygame.event.pump()
        for axis_idx in center:
            center[axis_idx] += axis_raw(axis_idx)
        time.sleep(0.005)
    for axis_idx in center:
        center[axis_idx] /= sample_count

    def axis_value(index: int, deadzone: float) -> float:
        v = axis_raw(index) - center.get(index, 0.0)
        v = clamp(v, -1.0, 1.0)
        return 0.0 if abs(v) < deadzone else float(v)

    def button_pressed(candidates: list[int]) -> bool:
        for idx in candidates:
            if idx < joystick.get_numbuttons() and joystick.get_button(idx):
                return True
        return False

    def trigger_value(axis_candidates: list[int], button_candidates: list[int]) -> float:
        # Most drivers report triggers in [-1, 1], where -1 is released and +1 is pressed.
        for axis_idx in axis_candidates:
            if axis_idx < joystick.get_numaxes():
                raw = float(joystick.get_axis(axis_idx))
                return (raw + 1.0) * 0.5
        # Some drivers expose triggers as digital buttons.
        if button_pressed(button_candidates):
            return 1.0
        return 0.0

    joint_speed = float(args.max_joint_speed_deg_s)
    gripper_speed = float(args.gripper_speed_unit_s)

    try:
        while True:
            t0 = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # Circle/B exits. Mappings differ by driver, so check both common indices.
            if button_pressed([1, 2]):
                break

            # Keep right stick on axis 3 for elbow to avoid trigger/axis overlap on some drivers.
            lx = axis_value(0, args.deadzone)
            ly = axis_value(1, args.deadzone)
            rx = axis_value(2, args.deadzone)
            ry = axis_value(3, args.deadzone)

            # Map sticks to joints.
            targets["shoulder_pan.pos"] = targets.get("shoulder_pan.pos", 0.0) + lx * joint_speed * dt
            targets["shoulder_lift.pos"] = targets.get("shoulder_lift.pos", 0.0) - ly * joint_speed * dt
            targets["wrist_flex.pos"] = targets.get("wrist_flex.pos", 0.0) + rx * joint_speed * dt
            targets["elbow_flex.pos"] = targets.get("elbow_flex.pos", 0.0) - ry * joint_speed * dt

            roll_dir = 0.0
            # Include alternate shoulder-button indices seen on some DS drivers.
            if button_pressed([4, 6, 9]):  # L1/LB
                roll_dir -= 1.0
            if button_pressed([5, 7, 10]):  # R1/RB
                roll_dir += 1.0
            targets["wrist_roll.pos"] = targets.get("wrist_roll.pos", 0.0) + roll_dir * joint_speed * dt

            # Trigger-based gripper control.
            # Common DualSense trigger axes on pygame: L2=4, R2=5.
            l2 = trigger_value([4], [6])
            r2 = trigger_value([5], [7])
            targets["gripper.pos"] = targets.get("gripper.pos", 0.0) + (r2 - l2) * gripper_speed * dt

            # Enforce bounds.
            targets["gripper.pos"] = clamp(targets.get("gripper.pos", 0.0), 0.0, 100.0)
            for key, (lo, hi) in deg_bounds.items():
                if key in targets:
                    targets[key] = clamp(targets[key], lo, hi)

            robot.send_action(targets)

            sleep_s = dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            if joystick.get_init():
                joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()
