# fmt: off

from dataclasses import dataclass

# Usage:
#
# from template.Sm90a_gemm_pre_config import config
# config.smem_M = ...
# config.smem_N = ...
#
# # Generates kernels based on above config and places it in this module.
# from template.Sm90a_gen_gemm import *
#
# # Autogenerate json file
# import json
# json.dump(cases, open(__file__ + ".json", "w"))

@dataclass(slots=True)
class Sm90aGemmConfig:
    smem_M: int = None
    smem_N: int = None
    RING: int = 4
    tma_to_gmem: bool = None
    enable_split_k: bool = None

    def make_proc_name(self, ncta_M: int, ncta_N: int) -> str:
        suffix = "rmemC" if not self.tma_to_gmem else "splitK" if self.enable_split_k else "tmaC"
        return f"xgemm_Sm90a_r{self.RING}_m{ncta_M}n{ncta_N}_m{self.smem_M}n{self.smem_N}_{suffix}"

config = Sm90aGemmConfig()
