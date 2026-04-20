#!/usr/bin/env python
"""SO-100 gamepad teleop tuned for Raspberry Pi + Linux pygame + PS5 DualSense / DS4.

Axis layout differs from typical Windows/SDL mappings: L2/R2 are often axes 2–3 and the
right stick is axes 4–5. L2/R2 digital buttons are usually 6/7; those must not be treated
as L1/R1 for wrist_roll.

For a Windows laptop, use so100_follower_gamepad_teleop.py instead.
"""

import argparse
import time

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

# pygame axis indices on Linux for DualSense / many DS4 hid drivers
_RIGHT_STICK_X_AXIS = 4
_RIGHT_STICK_Y_AXIS = 5
_L2_TRIGGER_AXIS = 2
_R2_TRIGGER_AXIS = 3
_L2_TRIGGER_BUTTONS = [6]
_R2_TRIGGER_BUTTONS = [7]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control one SO-100 follower with a gamepad (Raspberry Pi / Linux PS controller layout)."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0).")
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
    print(f"Using gamepad: {joystick.get_name()} (Raspberry Pi / Linux PS layout)")

    print(
        """
SO-100 gamepad teleop (joint space):
  Left stick:  shoulder_pan (X), shoulder_lift (Y)
  Right stick: wrist_flex (X), elbow_flex (Y)
  L1 / R1:     wrist_roll - / +
  L2 / R2:     gripper close / open
  Circle/B or Square/Cross:  exit (driver-dependent)
"""
    )

    def axis_raw(index: int) -> float:
        if index >= joystick.get_numaxes():
            return 0.0
        return float(joystick.get_axis(index))

    center_axes = {0, 1, _RIGHT_STICK_X_AXIS, _RIGHT_STICK_Y_AXIS}
    center = {i: 0.0 for i in center_axes}
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
        for axis_idx in axis_candidates:
            if axis_idx < joystick.get_numaxes():
                raw = float(joystick.get_axis(axis_idx))
                return (raw + 1.0) * 0.5
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

            if button_pressed([1, 2]):
                break

            lx = axis_value(0, args.deadzone)
            ly = axis_value(1, args.deadzone)
            rx = axis_value(_RIGHT_STICK_X_AXIS, args.deadzone)
            ry = axis_value(_RIGHT_STICK_Y_AXIS, args.deadzone)

            targets["shoulder_pan.pos"] = targets.get("shoulder_pan.pos", 0.0) + lx * joint_speed * dt
            targets["shoulder_lift.pos"] = targets.get("shoulder_lift.pos", 0.0) - ly * joint_speed * dt
            targets["wrist_flex.pos"] = targets.get("wrist_flex.pos", 0.0) + rx * joint_speed * dt
            targets["elbow_flex.pos"] = targets.get("elbow_flex.pos", 0.0) - ry * joint_speed * dt

            roll_dir = 0.0
            if button_pressed([4]):
                roll_dir -= 1.0
            if button_pressed([5]):
                roll_dir += 1.0
            targets["wrist_roll.pos"] = targets.get("wrist_roll.pos", 0.0) + roll_dir * joint_speed * dt

            l2 = trigger_value([_L2_TRIGGER_AXIS], _L2_TRIGGER_BUTTONS)
            r2 = trigger_value([_R2_TRIGGER_AXIS], _R2_TRIGGER_BUTTONS)
            targets["gripper.pos"] = targets.get("gripper.pos", 0.0) + (r2 - l2) * gripper_speed * dt

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
