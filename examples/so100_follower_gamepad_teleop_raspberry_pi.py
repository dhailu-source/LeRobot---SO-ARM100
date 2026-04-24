#!/usr/bin/env python
"""SO-100 gamepad teleop for Raspberry Pi + Linux pygame + PS5 DualSense / DS4.

Controls (see in-game help): left stick shoulders; right stick Y elbow; D-pad Y wrist;
L1/R1 wrist roll; L2/R2 gripper; Square exits. Axis pairs for sticks/triggers are
auto-detected (USB vs Bluetooth driver layouts).

For a Windows laptop, use so100_follower_gamepad_teleop.py instead.
"""

from __future__ import annotations

import argparse
import statistics
import time

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

_L2_TRIGGER_BUTTONS = [6]
_R2_TRIGGER_BUTTONS = [7]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def detect_axis_layout(pump, axis_raw, num_axes: int) -> tuple[str, int, int, int, int]:
    """Return (layout_name, rs_x, rs_y, l2_axis, r2_axis).

    sticks_on_45: L2/R2 on 2,3 (rest ~ -1), right stick on 4,5 (rest ~ 0).
    sticks_on_23: right stick on 2,3, L2/R2 on 4,5.
    """
    if num_axes < 6:
        return "sticks_on_45_fallback", 4, 5, 2, 3

    samples: list[tuple[float, float, float, float]] = []
    for _ in range(45):
        pump()
        samples.append(
            (
                axis_raw(2),
                axis_raw(3),
                axis_raw(4),
                axis_raw(5),
            )
        )
        time.sleep(0.004)

    m2 = statistics.fmean(s[0] for s in samples)
    m3 = statistics.fmean(s[1] for s in samples)
    m4 = statistics.fmean(s[2] for s in samples)
    m5 = statistics.fmean(s[3] for s in samples)

    def trigger_like(v: float) -> bool:
        return v < -0.35

    def stick_like(v: float) -> bool:
        return abs(v) < 0.45

    pair_23_triggers = trigger_like(m2) and trigger_like(m3) and stick_like(m4) and stick_like(m5)
    pair_45_triggers = stick_like(m2) and stick_like(m3) and trigger_like(m4) and trigger_like(m5)

    if pair_23_triggers and not pair_45_triggers:
        return "sticks_on_45", 4, 5, 2, 3
    if pair_45_triggers and not pair_23_triggers:
        return "sticks_on_23", 2, 3, 4, 5
    # Ambiguous: pick whichever pair is more "trigger-like" (more negative average).
    avg23 = (m2 + m3) * 0.5
    avg45 = (m4 + m5) * 0.5
    if avg23 < avg45:
        return "sticks_on_45_heuristic", 4, 5, 2, 3
    return "sticks_on_23_heuristic", 2, 3, 4, 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control one SO-100 follower with a gamepad (Raspberry Pi / Linux PS controller)."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id (calibration id).")
    parser.add_argument("--fps", type=float, default=50.0, help="Control loop frequency.")
    parser.add_argument("--max_joint_speed_deg_s", type=float, default=90.0, help="Max body-joint speed.")
    parser.add_argument("--gripper_speed_unit_s", type=float, default=80.0, help="Max gripper speed in [0..100]/s.")
    parser.add_argument("--deadzone", type=float, default=0.12, help="Joystick deadzone.")
    parser.add_argument(
        "--gripper_deadzone",
        type=float,
        default=0.14,
        help="Ignore analog trigger difference smaller than this (reduces drift / false open).",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "sticks_on_23", "sticks_on_45"),
        default="auto",
        help="Axis map: triggers on 4,5 vs 2,3. Use auto unless detection is wrong.",
    )
    parser.add_argument(
        "--exit_button",
        type=int,
        default=2,
        help="Button index for Square (exit). Try 2 or 3 if exit does not respond.",
    )
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
    n_axes = joystick.get_numaxes()
    if joystick.get_numhats() < 1:
        raise RuntimeError(
            "This layout needs a D-pad (joystick hat). Pygame reports 0 hats on this device. "
            "Try another USB port/cable or check the controller in `jstest` / `evtest`."
        )
    wrist_hat_index = 0

    def axis_raw(index: int) -> float:
        if index >= n_axes:
            return 0.0
        return float(joystick.get_axis(index))

    def pump() -> None:
        pygame.event.pump()

    for _ in range(5):
        pump()
        time.sleep(0.01)

    if args.layout == "auto":
        layout_name, rs_x, rs_y, l2_ax, r2_ax = detect_axis_layout(pump, axis_raw, n_axes)
    elif args.layout == "sticks_on_23":
        layout_name, rs_x, rs_y, l2_ax, r2_ax = "sticks_on_23_manual", 2, 3, 4, 5
    else:
        layout_name, rs_x, rs_y, l2_ax, r2_ax = "sticks_on_45_manual", 4, 5, 2, 3

    print(f"Using gamepad: {joystick.get_name()} | layout={layout_name} | axes RS=({rs_x},{rs_y}) L2R2=({l2_ax},{r2_ax})")
    print("Keep fingers off L2/R2 and the right stick for one second while axes are detected.")

    # Resting values for trigger axes (for signed vs unsigned scaling).
    l2_rest_samples: list[float] = []
    r2_rest_samples: list[float] = []
    for _ in range(25):
        pump()
        l2_rest_samples.append(axis_raw(l2_ax))
        r2_rest_samples.append(axis_raw(r2_ax))
        time.sleep(0.004)
    l2_rest = statistics.fmean(l2_rest_samples)
    r2_rest = statistics.fmean(r2_rest_samples)

    def trigger_activation(axis_idx: int, rest: float) -> float:
        raw = axis_raw(axis_idx)
        # Signed triggers (DualSense): rest ~ -1, full press ~ +1
        if rest < -0.25:
            return (clamp(raw, -1.0, 1.0) + 1.0) * 0.5
        # Unsigned 0..1: remove idle bias so a noisy rest value does not count as "open".
        span = max(0.2, 1.0 - clamp(rest, 0.0, 0.95))
        return clamp((raw - rest) / span, 0.0, 1.0)

    center_axes = {0, 1, rs_x, rs_y}
    center = {i: 0.0 for i in center_axes}
    sample_count = 20
    for _ in range(sample_count):
        pump()
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

    def read_l2_r2() -> tuple[float, float]:
        l_act = trigger_activation(l2_ax, l2_rest)
        r_act = trigger_activation(r2_ax, r2_rest)
        if button_pressed(_L2_TRIGGER_BUTTONS):
            l_act = max(l_act, 1.0)
        if button_pressed(_R2_TRIGGER_BUTTONS):
            r_act = max(r_act, 1.0)
        return l_act, r_act

    print(
        """
SO-100 gamepad teleop (joint space) — Raspberry Pi layout:
  Left stick:   shoulder_pan (X), shoulder_lift (Y)
  Right stick Y: elbow_flex
  D-pad up/down: wrist_flex
  L1 / R1:      wrist_roll - / +
  L2 / R2:      gripper close / open
  Square:       exit  (if wrong button exits, try: --exit_button 3)
"""
    )

    joint_speed = float(args.max_joint_speed_deg_s)
    gripper_speed = float(args.gripper_speed_unit_s)

    try:
        while True:
            t0 = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if button_pressed([args.exit_button]):
                break

            lx = axis_value(0, args.deadzone)
            ly = axis_value(1, args.deadzone)
            ry = axis_value(rs_y, args.deadzone)
            _hx, hy = joystick.get_hat(wrist_hat_index)

            targets["shoulder_pan.pos"] = targets.get("shoulder_pan.pos", 0.0) + lx * joint_speed * dt
            targets["shoulder_lift.pos"] = targets.get("shoulder_lift.pos", 0.0) - ly * joint_speed * dt
            targets["wrist_flex.pos"] = targets.get("wrist_flex.pos", 0.0) + float(hy) * joint_speed * dt
            targets["elbow_flex.pos"] = targets.get("elbow_flex.pos", 0.0) - ry * joint_speed * dt

            roll_dir = 0.0
            if button_pressed([4]):
                roll_dir -= 1.0
            if button_pressed([5]):
                roll_dir += 1.0
            targets["wrist_roll.pos"] = targets.get("wrist_roll.pos", 0.0) + roll_dir * joint_speed * dt

            l2, r2 = read_l2_r2()
            grip_delta = r2 - l2
            if abs(grip_delta) < float(args.gripper_deadzone):
                grip_delta = 0.0
            targets["gripper.pos"] = targets.get("gripper.pos", 0.0) + grip_delta * gripper_speed * dt

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
