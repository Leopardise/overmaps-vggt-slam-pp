import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import X


class _Mat4Wrapper:
    """So map.py can keep calling .matrix() like SL4/Pose3."""
    def __init__(self, T: np.ndarray):
        self._T = np.asarray(T, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        return self._T


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def sim3_from_matrix(T: np.ndarray, eps: float = 1e-12) -> gtsam.Similarity3:
    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4)
    A = T[:3, :3]
    t = T[:3, 3]

    detA = np.linalg.det(A)
    s = np.cbrt(max(detA, eps))
    if not np.isfinite(s) or abs(s) < eps:
        s = 1.0

    R_raw = A / s
    R = _project_to_so3(R_raw)

    rot = gtsam.Rot3(R)

    # translation: some bindings accept Point3, some accept ndarray
    try:
        trans = gtsam.Point3(float(t[0]), float(t[1]), float(t[2]))
        return gtsam.Similarity3(rot, trans, float(s))
    except Exception:
        return gtsam.Similarity3(rot, np.array(t, dtype=np.float64), float(s))


def _as_xyz(tr) -> np.ndarray:
    """
    Handle both possible binding behaviors:
      - tr is gtsam.Point3-like with x()/y()/z()
      - tr is already numpy array shape (3,) or (3,1)
    """
    if hasattr(tr, "x") and callable(tr.x):
        return np.array([tr.x(), tr.y(), tr.z()], dtype=np.float64)
    arr = np.asarray(tr, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"Unexpected translation shape: {np.asarray(tr).shape}")
    return arr[:3]


def sim3_to_matrix(S) -> np.ndarray:
    """
    Convert gtsam.Similarity3 to 4x4 Sim(3) matrix.
    Bindings may return translation as ndarray.
    """
    # Some builds may return Similarity3 as a python proxy; keep robust:
    R = S.rotation().matrix()
    t = _as_xyz(S.translation())
    s = float(S.scale())

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


class PoseGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.relative_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05], dtype=np.float64)
        )
        self.anchor_noise = noiseModel.Diagonal.Sigmas(
            np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64)
        )

        self.initialized_nodes = set()
        self.num_loop_closures = 0

    def add_homography(self, key: int, global_h: np.ndarray):
        k = X(int(key))
        if k in self.initialized_nodes:
            return
        S = sim3_from_matrix(global_h)
        self.values.insert(k, S)
        self.initialized_nodes.add(k)

    def add_between_factor(self, key1: int, key2: int, relative_h: np.ndarray, noise):
        k1, k2 = X(int(key1)), X(int(key2))
        if k1 not in self.initialized_nodes or k2 not in self.initialized_nodes:
            raise ValueError(f"Both poses must exist before adding factor: {key1}, {key2}")
        rel = sim3_from_matrix(relative_h)
        self.graph.add(gtsam.BetweenFactorSimilarity3(k1, k2, rel, noise))

    def add_prior_factor(self, key: int, global_h: np.ndarray, noise):
        k = X(int(key))
        if k not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        prior = sim3_from_matrix(global_h)
        self.graph.add(gtsam.PriorFactorSimilarity3(k, prior, noise))

    def get_homography(self, node_id: int):
        """
        Return wrapper with .matrix(), so map.py can stay unchanged.
        """
        k = X(int(node_id))
        S = self.values.atSimilarity3(k)
        T = sim3_to_matrix(S)
        return _Mat4Wrapper(T)

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        opt = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        self.values = opt.optimize()

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures
