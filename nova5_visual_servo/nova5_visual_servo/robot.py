import xml.etree.ElementTree as ET
import numpy as np
from .core import exp_so3, vector


class URDFRobot:
    """Serial-chain FK and base geometric Jacobian from official URDF."""
    def __init__(self, path, base='base_link', tip='Link6'):
        root = ET.parse(path).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain = []
        while tip != base:
            j = by_child[tip]
            chain.insert(0, j)
            tip = j.find('parent').get('link')
        self.chain = []
        self.names = []
        self.lower, self.upper = [], []
        for j in chain:
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            T = np.eye(4)
            T[:3, :3] = exp_so3(np.array([0, 0, y]))@exp_so3(np.array([0, p, 0]))@exp_so3(np.array([r, 0, 0]))
            T[:3, 3] = xyz
            active = j.get('type') != 'fixed'
            axis = np.zeros(3)
            if active:
                if j.get('type') != 'revolute':
                    raise ValueError('expected bounded revolute joint')
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                axis /= np.linalg.norm(axis)
                self.names.append(j.get('name'))
                self.lower.append(float(j.find('limit').get('lower')))
                self.upper.append(float(j.find('limit').get('upper')))
            self.chain.append((T, axis, active))
        if len(self.names) != 6:
            raise ValueError('expected six joints')

    def evaluate(self, q):
        q = vector(q, 6)
        T, axes, origins, index = np.eye(4), [], [], 0
        for origin, axis, active in self.chain:
            T = T@origin
            if active:
                axes.append(T[:3, :3]@axis)
                origins.append(T[:3, 3].copy())
                R = np.eye(4)
                R[:3, :3] = exp_so3(axis*q[index])
                T = T@R
                index += 1
        J = np.column_stack([np.r_[np.cross(a, T[:3, 3]-p), a] for a, p in zip(axes, origins)])
        return T, J

    def fk(self, q):
        return self.evaluate(q)[0]

    def jac(self, q):
        return self.evaluate(q)[1]
