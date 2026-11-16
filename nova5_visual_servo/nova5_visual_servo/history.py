from collections import deque
import numpy as np


class JointHistory:
    def __init__(self, max_gap=.05):
        self.samples = deque(maxlen=500)
        self.max_gap = max_gap

    def add(self, t, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(t):
            raise ValueError('invalid joint sample')
        if self.samples and t <= self.samples[-1][0]:
            raise ValueError('joint timestamp not increasing')
        self.samples.append((t, q.copy()))

    def at(self, t):
        for (ta, a), (tb, b) in zip(self.samples, list(self.samples)[1:]):
            if ta <= t <= tb and tb-ta <= self.max_gap:
                return a+(b-a)*(t-ta)/(tb-ta)
        raise ValueError('no bracketing feedback for image')

    def latest(self):
        (ta, a), (tb, b) = list(self.samples)[-2:]
        if tb-ta > self.max_gap:
            raise ValueError('feedback gap')
        return tb, b.copy(), (b-a)/(tb-ta)
