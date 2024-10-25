class LaneUnit(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<exo.spork.lane_unit.LaneUnit {self.name}>"

    def __str__(self):
        return self.name

    def __bool__(self):
        return self.name != "_"


hole = LaneUnit("_")
cpu_threads = LaneUnit("cpu_threads")
cuda_clusters = LaneUnit("cuda_clusters")
cuda_blocks = LaneUnit("cuda_blocks")
cuda_warpgroups = LaneUnit("cuda_warpgroups")
cuda_warps = LaneUnit("cuda_warps")
cuda_threads = LaneUnit("cuda_threads")

lane_unit_dict = {
    unit.name: unit
    for unit in [
        hole,
        cpu_threads,
        cuda_clusters,
        cuda_blocks,
        cuda_warpgroups,
        cuda_warps,
        cuda_threads,
    ]
}
