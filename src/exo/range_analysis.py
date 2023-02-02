from .LoopIR import LoopIR


class AffineIndexRangeAnalysis:
    @staticmethod
    def merge_add(lhs_range, rhs_range):
        return (lhs_range[0] + rhs_range[0], lhs_range[1] + rhs_range[1])

    @staticmethod
    def merge_sub(lhs_range, rhs_range):
        return (lhs_range[0] - rhs_range[1], lhs_range[1] - rhs_range[0])

    @staticmethod
    def merge_mul(lhs_range, rhs_range):
        if lhs_range[0] < 0 or rhs_range[0] < 0:
            return None

        a = [i * j for i in lhs_range for j in rhs_range]
        return (min(a), max(a))

    @staticmethod
    def merge_div(lhs_range, rhs_range):
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        if lhs_range[0] < 0:
            return None

        d = rhs_range[0]
        return (lhs_range[0] // d, lhs_range[1] // d)

    @staticmethod
    def merge_mod(lhs_range, rhs_range):
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        m = rhs_range[0]
        if lhs_range[0] // m == lhs_range[1] // m:
            return (lhs_range[0] % m, lhs_range[1] % m)

        # We can be a bit smarter here when lhs_range[1] - lhs_range[0] < m
        # but we would need to keep track of two ranges for the expression

        return (0, m)

    def __init__(self, e, env) -> None:
        self._env = env
        self._result = self._analyze_range(e)

    def result(self):
        return self._result

    def _analyze_range(self, e):
        assert isinstance(e, LoopIR.expr)

        if not e.type.is_indexable():
            return None

        if isinstance(e, LoopIR.Read):
            return self._env[e.name]
        elif isinstance(e, LoopIR.Const):
            return (e.val, e.val)
        elif isinstance(e, LoopIR.USub):
            e_range = self._analyze_range(e.arg)
            if e_range is None:
                return None
            return (-e_range[1], -e_range[0])
        elif isinstance(e, LoopIR.BinOp):
            lhs_range = self._analyze_range(e.lhs)
            rhs_range = self._analyze_range(e.rhs)
            if lhs_range is None or rhs_range is None:
                return None
            merge_binop = {
                "+": AffineIndexRangeAnalysis.merge_add,
                "-": AffineIndexRangeAnalysis.merge_sub,
                "*": AffineIndexRangeAnalysis.merge_mul,
                "/": AffineIndexRangeAnalysis.merge_div,
                "%": AffineIndexRangeAnalysis.merge_mod,
            }
            return merge_binop[e.op](lhs_range, rhs_range)
        else:
            return None
