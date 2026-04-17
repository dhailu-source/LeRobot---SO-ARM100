#!/usr/bin/env python

import argparse
import time

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick connection check for an SO-100 follower arm.")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3).")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot id (used for calibration file).")
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Do not auto-calibrate on connect (use existing calibration file only).",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of observation reads.")
    args = parser.parse_args()

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id))
    robot.connect(calibrate=not args.no_calibrate)
    try:
        for i in range(args.n):
            obs = robot.get_observation()
            keys = [k for k in obs.keys() if isinstance(k, str) and k.endswith(".pos")]
            first = {k: float(obs[k]) for k in sorted(keys)[:3]}
            print(f"[{i+1}/{args.n}] ok, sample joints: {first}")
            time.sleep(0.1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

