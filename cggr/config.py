"""CGGR parameters; fixed numerical protocol from Supplementary S1.7."""
from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class ModelConfig:
    clusters: int
    alpha: float = 1.0
    beta: float = 1.0
    random_state: int = 0
    device: str = "cuda"
    epochs: int = field(default=300, init=False)
    neighbors: int = field(default=30, init=False)
    weight_floor_mass: float = field(default=0.1, init=False)
    graph_inner_steps: int = field(default=5, init=False)
    penalty_initial: float = field(default=1.0, init=False)
    penalty_growth: float = field(default=1.5, init=False)
    penalty_max: float = field(default=1000000.0, init=False)
    cg_rtol: float = field(default=1e-08, init=False)
    cg_maxiter: int = field(default=300, init=False)
    projection_tolerance: float = field(default=1e-08, init=False)
    projection_maxiter: int = field(default=2000, init=False)
    rank_penalty_initial: float = field(default=15.0, init=False)
    rank_penalty_min: float = field(default=1e-08, init=False)
    rank_penalty_max: float = field(default=100000000.0, init=False)
    eigen_zero_tolerance: float = field(default=1e-06, init=False)
    eigen_solver_tolerance: float = field(default=1e-09, init=False)
    eigen_maxiter: int = field(default=10000, init=False)
    h_initial_step: float = field(default=0.001, init=False)
    h_adaptive_step: bool = field(default=True, init=False)
    armijo: float = field(default=0.0001, init=False)
    backtracking: float = field(default=0.5, init=False)
    max_backtracks: int = field(default=30, init=False)
    distance_block_rows: int = field(default=128, init=False)
    kmeans_n_init: int = field(default=20, init=False)
    gpu_eigen_maxiter: int = field(default=300, init=False)
    gpu_dense_eigen_limit: int = field(default=1024, init=False)

    def validate(self, n: int, views: int) -> None:
        if views not in (2, 3):
            raise ValueError("Expected two or three paired modalities")
        if type(self.clusters) is not int or not 2 <= self.clusters <= n // 2:
            raise ValueError("The graph rank target requires 2 <= K <= N//2")
        if self.neighbors > n - 2:
            raise ValueError("The Method uses 30 neighbors and requires at least 32 cells")
        if any(not math.isfinite(x) or x <= 0 for x in (self.alpha, self.beta)):
            raise ValueError("CGGR requires alpha > 0 and beta > 0")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        if not isinstance(self.device, str) or not (
                self.device in ("cpu", "cuda") or
                (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")

    def to_dict(self):
        return asdict(self)
