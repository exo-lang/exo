# fmt: off

from __future__ import annotations

from .Sm90a_gemm_pre_config import config, Sm90aGemmConfig
from .make_Sm90a_gemm import make_Sm90a_gemm

cases = []

gemm_m1n1 = make_Sm90a_gemm(config, 1, 1)
gemm_m2n1 = make_Sm90a_gemm(config, 2, 1)
gemm_m1n2 = make_Sm90a_gemm(config, 1, 2)
gemm_m2n2 = make_Sm90a_gemm(config, 2, 2)

for p in (gemm_m1n1, gemm_m2n1, gemm_m1n2, gemm_m2n2):
    cases.append({
        "algorithm": "gemm",
        "proc": p.name(),
        "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
        "K_split_max": 0x7fffffff if config.enable_split_k else 1,
        "A_major": "row", "B_major": "col", "C_major": "col",
        "A_type": "f32", "B_type": "f32", "C_type": "f32",
    })
