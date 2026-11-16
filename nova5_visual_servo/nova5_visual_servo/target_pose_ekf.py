"""12D left-error ESKF: [dp, dv, dtheta_base, domega_base].

Constant base-frame linear/angular velocity; continuous white acceleration.
No SciPy dependency. Angular Q is integrated using 8-point Gauss quadrature.
"""
import numpy as np


def left_jacobian(phi):
    from .core import skew
    a = np.linalg.norm(phi)
    K = skew(phi)
    if a < 1e-6:
        return np.eye(3)+.5*K+K@K/6
    return np.eye(3)+(1-np.cos(a))/a**2*K+(a-np.sin(a))/a**3*K@K


def transition(w, dt):
    from .core import exp_so3
    F = np.eye(12)
    F[:3, 3:6] = np.eye(3)*dt
    F[6:9, 6:9] = exp_so3(w*dt)
    F[6:9, 9:12] = left_jacobian(w*dt)*dt
    return F


def process_noise(w, dt, linear_psd, angular_psd):
    Q = np.zeros((12, 12))
    Q[:3, :3] = np.eye(3)*linear_psd*dt**3/3
    Q[:3, 3:6] = Q[3:6, :3] = np.eye(3)*linear_psd*dt**2/2
    Q[3:6, 3:6] = np.eye(3)*linear_psd*dt
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for x, weight in zip(nodes, weights):
        s = (x+1)*dt/2
        G = np.vstack((left_jacobian(w*s)*s, np.eye(3)))
        Q[6:, 6:] += angular_psd*weight*dt/2*(G@G.T)
    return Q


class TargetTracker:
    def __init__(self, max_gap, max_speed, max_omega,
                 position_std_m=.002, orientation_std_rad=.00872664626,
                 linear_accel_psd_m2_s3=.02, angular_accel_psd_rad2_s3=.20,
                 initial_velocity_std=.5, initial_omega_std=1.,
                 innovation_gate=22.46, max_rejections=3,
                 max_position_sigma=.05, max_orientation_sigma=.2):
        values = [max_gap, max_speed, max_omega, position_std_m, orientation_std_rad,
                  linear_accel_psd_m2_s3, angular_accel_psd_rad2_s3,
                  initial_velocity_std, initial_omega_std, innovation_gate,
                  max_position_sigma, max_orientation_sigma]
        if not np.isfinite(values).all() or min(values) <= 0 or not isinstance(max_rejections, int) or max_rejections < 1:
            raise ValueError('invalid ESKF configuration')
        self.max_gap, self.max_speed, self.max_omega = max_gap, max_speed, max_omega
        self.linear_psd, self.angular_psd = linear_accel_psd_m2_s3, angular_accel_psd_rad2_s3
        self.Rm = np.diag([position_std_m**2]*3+[orientation_std_rad**2]*3)
        self.P = np.diag([position_std_m**2]*3+[initial_velocity_std**2]*3+
                         [orientation_std_rad**2]*3+[initial_omega_std**2]*3)
        self.gate, self.max_rejections = innovation_gate, max_rejections
        self.max_position_sigma, self.max_orientation_sigma = max_position_sigma, max_orientation_sigma
        self.t, self.seq, self.last_seen_time = None, -1, -np.inf
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.T = np.eye(4)
        self.ready = False
        self.rejections = 0
        self.nis = None
        self.reason = 'uninitialized'

    def _propagate(self, dt):
        from .core import exp_so3
        T = self.T.copy()
        T[:3, 3] += self.v*dt
        T[:3, :3] = exp_so3(self.w*dt)@T[:3, :3]
        F = transition(self.w, dt)
        P = F@self.P@F.T+process_noise(self.w, dt, self.linear_psd, self.angular_psd)
        return T, (P+P.T)/2

    def update(self, seq, exposure_time, T_BE_at_exposure, T_EC, T_CQ):
        from .core import pose, log_so3, exp_so3, Fault
        if not np.isfinite(exposure_time):
            raise Fault('invalid exposure time')
        if seq <= self.seq or exposure_time <= self.last_seen_time:
            return False
        Z = pose(T_BE_at_exposure)@pose(T_EC)@pose(T_CQ)
        self.seq, self.last_seen_time = seq, exposure_time
        if self.t is None:
            self.T, self.t, self.reason = Z, exposure_time, 'initializing'
            return True
        dt = exposure_time-self.t
        if dt > self.max_gap:
            raise Fault('accepted observation gap exceeded; stop and reinitialize ESKF')
        T, P = self._propagate(dt)
        residual = np.r_[Z[:3, 3]-T[:3, 3], log_so3(Z[:3, :3]@T[:3, :3].T)]
        H = np.zeros((6, 12)); H[:3, :3] = np.eye(3); H[3:, 6:9] = np.eye(3)
        S = H@P@H.T+self.Rm
        self.nis = float(residual@np.linalg.solve(S, residual))
        if self.nis > self.gate:
            self.rejections += 1
            self.reason = 'innovation rejected'
            return False  # accepted timestamp/state/covariance remain unchanged
        K = np.linalg.solve(S, H@P).T
        dx = K@residual
        v, w = self.v+dx[3:6], self.w+dx[9:12]
        if np.linalg.norm(v) > self.max_speed or np.linalg.norm(w) > self.max_omega:
            raise Fault('estimated target speed outside device configuration')
        T[:3, 3] += dx[:3]
        T[:3, :3] = exp_so3(dx[6:9])@T[:3, :3]
        A = np.eye(12)-K@H
        P = A@P@A.T+K@self.Rm@K.T  # Joseph update
        G = np.eye(12); G[6:9, 6:9] = left_jacobian(dx[6:9])
        P = G@P@G.T  # reset left attitude error after injection
        self.T, self.P = T, (P+P.T)/2
        self.v, self.w, self.t = v, w, exposure_time
        self.ready, self.rejections, self.reason = True, 0, 'ok'
        return True

    def predict_state(self, t, max_horizon):
        from .core import Fault
        if self.t is None or not 0 <= t-self.t <= max_horizon:
            raise Fault('target prediction horizon invalid')
        T, P = self._propagate(t-self.t)
        ps = float(np.sqrt(max(0., np.linalg.eigvalsh(P[:3, :3])[-1])))
        rs = float(np.sqrt(max(0., np.linalg.eigvalsh(P[6:9, 6:9])[-1])))
        usable = (self.ready and self.rejections < self.max_rejections and
                  ps <= self.max_position_sigma and rs <= self.max_orientation_sigma)
        return dict(T=T, covariance=P, velocity=self.v.copy(), angular_velocity=self.w.copy(),
                    position_sigma_max_m=ps, orientation_sigma_max_rad=rs,
                    horizon_s=t-self.t, usable=usable,
                    reason='ok' if usable else 'warmup, rejection count or uncertainty limit')

    def predict(self, t, max_horizon):
        from .core import Fault
        result = self.predict_state(t, max_horizon)
        if not result['usable']:
            raise Fault(result['reason'])
        return result['T']
