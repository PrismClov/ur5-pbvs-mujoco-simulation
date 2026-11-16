"""Offline ideal Cartesian plant, not Nova5 dynamics. CSV for P/FF comparison."""
import argparse
import csv
import numpy as np
from .core import Config, Controller, TargetTracker, exp_so3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='simulation.csv')
    parser.add_argument('--no-ff', action='store_true')
    parser.add_argument('--no-prediction', action='store_true')
    args = parser.parse_args()
    def fk(q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        T[:3, :3] = exp_so3(q[3:])
        return T
    cfg = Config(np.ones(6), np.ones(6)*2, -np.ones(6)*10, np.ones(6)*10,
                 np.ones(6), np.ones(6)*2, np.ones(6), .01, .15, .1, .2, .005, .04,
                 feedforward=not args.no_ff, prediction=not args.no_prediction)
    tr = TargetTracker(.15, 1., 2.)
    tr.update(0, -.02, np.eye(4), np.eye(4), np.eye(4))
    tr.update(1, 0., np.eye(4), np.eye(4), np.eye(4))
    q, qd = np.zeros(6), np.zeros(6)
    ctrl = Controller(cfg, tr, np.eye(4), np.eye(4), fk, lambda q: np.eye(6),
                      lambda q: True, q, qd, 0.)
    errors = []
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'target_x', 'actual_x', 'error_m', 'request_rad_s'])
        for k in range(1, 501):
            t = k*.02
            target = .05*np.sin(t)
            if k % 2 == 0:
                T = np.eye(4)
                T[0, 3] = target
                tr.update(k, t, fk(q), np.eye(4), np.linalg.inv(fk(q))@T)
            out = ctrl.tick(t, q, qd, t)
            qd = (out['q_plan']-q)/.02
            q = out['q_plan']
            errors.append(target-q[0])
            writer.writerow([t, target, q[0], errors[-1], out['qdot_req'][0]])
    print(f'Ideal Cartesian simulation RMS: {np.sqrt(np.mean(np.square(errors))):.6f} m')


if __name__ == '__main__':
    main()
