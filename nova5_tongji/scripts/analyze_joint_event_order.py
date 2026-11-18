#!/usr/bin/env python3
"""Compare observed changes in joint motion on their respective host clocks."""
import argparse
import json


def accelerations(rows, clock, position):
    samples = []
    for row in rows:
        t = row[clock]
        if samples and t == samples[-1][0]:
            continue
        if samples and t < samples[-1][0]:
            raise ValueError(f"Nonmonotonic {clock}")
        samples.append((t, row[position]))
    events = []
    for (a, p), (b, q), (c, r) in zip(samples, samples[1:], samples[2:]):
        accel = [2 * ((z-y)/(c-b) - (y-x)/(b-a)) / (c-a)
                 for x, y, z in zip(p, q, r)]
        events.append((a, b, c, accel))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--start", type=float, default=15)
    parser.add_argument("--end", type=float, default=16)
    args = parser.parse_args()
    with open(args.log) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(row["command_mode"] != "servo_j" for row in rows):
        parser.error("Expected a nonempty ServoJ log")
    origin = rows[0]["time"]
    data = []
    for clock, position in [("time", "command_value"), ("feedback_time", "q_actual")]:
        series = accelerations(rows, clock, position)
        data.append([e for e in series if args.start <= e[1]-origin <= args.end])
    if not all(data):
        parser.error("No acceleration samples in window")
    print("Per-axis maximum absolute acceleration, signed; times relative to first row")
    for axis in range(len(rows[0]["q_actual"])):
        print("J" + str(axis+1), end=" ")
        for name, series in zip(("cmd", "fb"), data):
            a,b,c,values = max(series, key=lambda e: abs(e[3][axis]))
            print(f"{name}={values[axis]:+.2f}@{b-origin:.4f} [{a-origin:.4f},{c-origin:.4f}]", end=" ")
        print()


if __name__ == "__main__":
    main()
