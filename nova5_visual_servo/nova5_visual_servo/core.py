"""Offline PBVS core. Units: metres, radians, seconds. No robot connection.
T_AB maps coordinates in B into A. Jacobian: base-frame [p_dot_E, omega_E].
All timestamps must be mapped to the same monotonic clock.
"""
from dataclasses import dataclass
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_so3(v):
    a = np.linalg.norm(v)
    K = skew(v)
    if a < 1e-8:
        return np.eye(3) + K + .5 * K @ K
    return np.eye(3) + np.sin(a)/a*K + (1-np.cos(a))/a**2*K@K


def log_so3(R):
    a = np.arccos(np.clip((np.trace(R)-1)/2, -1., 1.))
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if a < 1e-8:
        return .5*v
    if np.pi-a < 1e-5:
        _, U = np.linalg.eigh((R+R.T)/2)
        axis = U[:, -1]
        if axis @ v < 0:
            axis = -axis
        return a*axis
    return a/(2*np.sin(a))*v


def pose(T):
    T = np.asarray(T, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)):
        raise ValueError('invalid rigid transform')
    return T.copy()


def vector(x, n):
    x = np.asarray(x, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError('invalid vector')
    return x


class Fault(RuntimeError):
    """Caller must invoke the separately validated hardware stop path."""


from .target_pose_ekf import TargetTracker


@dataclass
class Config:
    # No hardware defaults: limits must come from a verified device configuration.
    v_max: np.ndarray
    a_max: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    follow_max: np.ndarray
    kp: np.ndarray                  # six Cartesian proportional gains
    task_scale: np.ndarray          # six positive unit-balancing weights for DLS
    damping: float
    max_visual_age: float
    max_feedback_age: float
    max_prediction: float
    dt_min: float
    dt_max: float
    response_scale: float = 1.
    ff_gain: float = 1.
    execution_lead: float = 0.      # identified extra execution lead, NOT 255 ms
    prediction: bool = True
    feedforward: bool = True

    def __post_init__(self):
        for name in ('v_max', 'a_max', 'q_min', 'q_max', 'follow_max', 'kp', 'task_scale'):
            setattr(self, name, vector(getattr(self, name), 6).copy())
        for name in ('v_max', 'a_max', 'follow_max', 'kp', 'task_scale'):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(name)
        scalars = [self.damping, self.max_visual_age, self.max_feedback_age,
                   self.max_prediction, self.dt_min, self.dt_max,
                   self.response_scale, self.ff_gain, self.execution_lead]
        if (not np.isfinite(scalars).all() or min(scalars[:6]) <= 0
                or self.dt_min >= self.dt_max or np.any(self.q_min >= self.q_max)
                or self.response_scale <= 0 or not 0 <= self.ff_gain <= 1
                or self.execution_lead < 0):
            raise ValueError('invalid configuration')


class Controller:
    """Single-owner control core; never sends commands itself.

    fk(q)->T_BE; jac(q)->6x6 BASE GEOMETRIC Jacobian at end-effector origin.
    workspace_ok(q)->bool must check verified workspace/collision constraints.
    predict_robot(q,qd,sample_time,execution_time)->q_pred is optional.
    Default short-horizon model is q + qd*h, not a plant-delay inverse.
    Any fault latches; recreate ONLY after validated stop and resynchronization.
    """
    def __init__(self, cfg, tracker, T_QC_des, T_EC, fk, jac, workspace_ok,
                 q_initial, qd_initial, start_time, predict_robot=None):
        self.c, self.tracker = cfg, tracker
        self.T_QE_des = pose(T_QC_des)@np.linalg.inv(pose(T_EC))
        self.fk, self.jac, self.workspace_ok = fk, jac, workspace_ok
        self.robot_model = predict_robot
        self.q_plan = vector(q_initial, 6).copy()
        self.v = vector(qd_initial, 6).copy()
        if np.any(abs(self.v) > cfg.v_max) or not np.isfinite(start_time):
            raise ValueError('invalid initial state')
        self.last_time = start_time
        self.faulted = False

    def tick(self, now, q_actual, qd_actual, feedback_sample_time):
        if self.faulted:
            raise Fault('fault latched; stop and explicitly reinitialize')
        try:
            return self._tick(now, q_actual, qd_actual, feedback_sample_time)
        except Exception:
            self.faulted = True
            raise

    def _tick(self, now, q_actual, qd_actual, feedback_sample_time):
        c, tr = self.c, self.tracker
        q, qd = vector(q_actual, 6), vector(qd_actual, 6)
        dt = now-self.last_time
        if not c.dt_min <= dt <= c.dt_max:
            raise Fault('control cycle outside bounds; do not catch up')
        if not 0 <= now-feedback_sample_time <= c.max_feedback_age:
            raise Fault('stale feedback or invalid source timestamp')
        if tr.t is None or not 0 <= now-tr.t <= c.max_visual_age:
            raise Fault('stale vision')
        if np.any(abs(self.q_plan-q) > c.follow_max):
            raise Fault('command-feedback divergence')
        te = now+c.execution_lead if c.prediction else now
        if c.prediction:
            T_BQ = tr.predict(te, c.max_prediction)
            h = te-feedback_sample_time
            if not 0 <= h <= c.max_prediction:
                raise Fault('robot prediction horizon exceeded')
            qp = (q+qd*h if self.robot_model is None else
                  vector(self.robot_model(q, qd, feedback_sample_time, te), 6))
        else:
            T_BQ, qp = tr.predict(tr.t, c.max_prediction), q.copy()  # no extrapolation; retain ESKF validity checks
        Td = T_BQ@self.T_QE_des
        Ta = pose(self.fk(qp))
        error = np.r_[Td[:3, 3]-Ta[:3, 3], log_so3(Td[:3, :3]@Ta[:3, :3].T)]
        p_term = c.response_scale*c.kp*error
        # Velocity at the desired END-EFFECTOR origin, including rotating offset.
        offset = T_BQ[:3, :3]@self.T_QE_des[:3, 3]
        ff = np.r_[tr.v+np.cross(tr.w, offset), tr.w]
        ff *= c.ff_gain if c.feedforward and tr.ready else 0.
        J = np.asarray(self.jac(qp), dtype=float)
        if J.shape != (6, 6) or not np.isfinite(J).all():
            raise Fault('invalid Jacobian')
        WJ = c.task_scale[:, None]*J
        rhs = c.task_scale*(p_term+ff)
        req = WJ.T@np.linalg.solve(WJ@WJ.T+c.damping**2*np.eye(6), rhs)
        goal = np.clip(req, -c.v_max, c.v_max)
        vn = self.v+np.clip(goal-self.v, -c.a_max*dt, c.a_max*dt)
        qn = self.q_plan+.5*(self.v+vn)*dt
        for candidate in (q, qp, qn):
            if (not np.isfinite(candidate).all() or np.any(candidate < c.q_min)
                    or np.any(candidate > c.q_max) or not self.workspace_ok(candidate)):
                raise Fault('joint/workspace bounds')
        self.q_plan, self.v, self.last_time = qn, vn, now
        estimate = tr.predict_state(te if c.prediction else tr.t, c.max_prediction)
        return dict(ekf_position_sigma=estimate['position_sigma_max_m'],
                    ekf_orientation_sigma=estimate['orientation_sigma_max_rad'],
                    ekf_nis=tr.nis, ekf_rejections=tr.rejections,
                    q_plan=qn.copy(), v_next=vn.copy(), qdot_req=req,
                    p_term=p_term, ff=ff, error=error, execution_time=te,
                    visual_age=now-tr.t, feedback_age=now-feedback_sample_time,
                    velocity_limited=bool(np.any(abs(req) > c.v_max)),
                    acceleration_limited=bool(np.any(abs(goal-vn) > 1e-10)))
