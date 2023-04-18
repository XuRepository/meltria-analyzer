from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Config:
    """Dataclass with app parameters"""

    # Epochs
    epochs: int = 1000  # 1000 in paper

    # Batch size (note: should be divisible by sample size, otherwise throw an error)
    # batch_size: int = 50
    sample_to_batch_size_factor: int = 5

    # Learning rate (baseline rate = 1e-3)
    lr: float = 1e-3

    x_dims: int = 1
    z_dims: int = 1
    optimizer: str = "Adam"
    graph_threshold: float = 0.3  # original  0.3
    tau_A: float = 0.0
    lambda_A: float = 0.0
    c_A: int = 1  # penalty parameter
    c_A_ul: float = 1e20  # c_A upper litmit (default 1e20)
    use_A_connect_loss: int = 1  # default 0
    use_A_positiver_loss: int = 1  # default 0
    cuda: bool = False
    # no_cuda = True
    seed: int = 42
    encoder_hidden: int = 64
    decoder_hidden: int = 64
    temp: float = 0.5
    k_max_iter: float = 5  # default 1e2
    encoder: str = "mlp"
    decoder: str = "mlp"
    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0
    h_tol: float = 1e-8
    lr_decay: int = 200
    sche_gamma: float = 1.0
    prior: bool = False
    factor: bool = True

    # fit parameters
    eta: int = 10
    gamma: float = 0.25  # 0.25 in original paper

    def to_prefixed_dict(self, prefix: str = "daggnn") -> dict[str, Any]:
        return {f"{prefix}_{k}": v for k, v in asdict(self).items()}

    @classmethod
    def from_prefixed_dict(cls, prefix: str = "daggnn", **params: dict[str, Any]):
        return cls(**{k.replace(f"{prefix}_", ""): v for k, v in params.items() if k.startswith(prefix)})
