from __future__ import annotations

from .Sm90_fwd import *
from .Sm90_smem import *

from exo.API import *
from exo.platforms.cuda import *

from .Sm90_internal_util import *


__all__ = []  # Will be appended to


# TODO support more than float
store_d_util = """template <bool ColumnMajor, int32_t RegIndex, typename Window, typename Reg>
EXO_CUDA_INLINE void exo_Sm90_store_d_reg(
        Window dst, Reg value,
        int32_t m_offset, int32_t M_end, int32_t N_end)
{
    const uint32_t tid = threadIdx.x % 128u;
    const int32_t r_base = int32_t((tid / 32) * 16 + (tid % 32) / 4);
    const int32_t c_base = int32_t(tid % 4) * 2;
    const int32_t r = m_offset + r_base + ((RegIndex % 4) / 2) * 8;
    const int32_t c = c_base + (RegIndex / 4) * 8 + (RegIndex % 2);
    auto dst_ptr = reinterpret_cast<Reg*>(
            &dst.data[c * dst.strides[!ColumnMajor] + r * dst.strides[ColumnMajor]]);
    if (int32_t(r) < M_end && int32_t(c) < N_end) {
        *dst_ptr = value;
    }
}
"""

matrix_descriptor_util = """\
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename SwizzledElement>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(
    SwizzledElement* ptr, uint32_t mn_stride_elements, uint32_t mn_offset = 0)
{
    uint64_t mn_stride_bytes = mn_stride_elements * sizeof(SwizzledElement);
    return exo_matrix_descriptor_encode(exo_smemU32(ptr) + mn_offset * mn_stride_bytes)
           | exo_matrix_descriptor_encode(16) << 16u
           | exo_matrix_descriptor_encode(8 * mn_stride_bytes) << 32u
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
           | uint64_t(1) << 46  // Bit-field 46-48, Fixed constant value of 0b001 on Blackwell (?!!?)
#endif
           | uint64_t(SwizzledElement::get_swizzle_bits()) << 62;
}
"""


@dataclass(slots=True)
class WgmmaHelper:
    # wgmma.mma_async.sync.aligned.m{M}n{N}k{get_K()}.{ptx_dtype}.{ptx_atype}.{ptx.btype}
    M: int
    N: int
    ptx_dtype: str
    ptx_atype: str
    ptx_btype: str

    def __post_init__(self):
        M = self.M
        N = self.N
        if M % 64 != 0 or M <= 0:
            raise ValueError("Require M to be a positive multiple of 64")
        if N % 8 != 0 or N < 8 or N > 256:
            raise ValueError("Require N to be a multiple of 8 in [8, 256]")

    def ptx_instr_name(self):
        # fmt: off
        K = self.get_K()
        return f"wgmma.mma_async.sync.aligned.m64n{self.N}k{K}.{self.ptx_dtype}.{self.ptx_atype}.{self.ptx_btype}"
        # fmt: on

    def rmem_d_struct_name(self):
        # Qualify with exo_CudaUtil:: in usage in generated Exo function
        return f"exo_Sm90_RmemD_m{self.M}n{self.N}_{self.ptx_dtype}"

    def rmem_a_struct_name(self):
        # Qualify with exo_CudaUtil:: in usage in generated Exo function
        return f"exo_Sm90_RmemA_m{self.M}n{self.N}_{self.ptx_atype}"

    def dreg_ctype(self):
        # TODO
        assert self.ptx_dtype == "f32"
        return "float"

    def areg_ctype(self):
        # TODO
        assert self.ptx_dtype == "tf32"
        return "unsigned"

    def get_K(self):
        assert self.ptx_atype == "tf32", "TODO"
        assert self.ptx_btype == "tf32", "TODO"
        return 8  # TODO

    def dreg_names(self, m=None, n=None):
        result = []
        assert self.ptx_dtype == "f32", "TODO"
        n_stride = 8  # TODO

        m_lo = 0 if m is None else m
        m_hi = self.M if m is None else m + 64
        n_lo = 0 if n is None else n
        n_hi = self.N if n is None else n + n_stride

        for m_ in range(m_lo, m_hi, 64):
            for n_ in range(n_lo, n_hi, n_stride):
                result.append(f"m{m_}n{n_}r0")
                result.append(f"m{m_}n{n_}r1")
                result.append(f"m{m_}n{n_}r2")
                result.append(f"m{m_}n{n_}r3")

        return result

    def areg_names(self, m=None):
        result = []
        m_lo = 0 if m is None else m
        m_hi = self.M if m is None else m + 64

        assert self.ptx_atype == "tf32", "TODO"
        k_divisor = 2  # TODO

        for m_ in range(m_lo, m_hi, 64):
            for r_ in range(0, self.get_K() // k_divisor):
                result.append(f"m{m_}r{r_}")

        return result

    def rmem_d_struct_def(self):
        sname = self.rmem_d_struct_name()
        return f"""struct {sname} {{
    {self.dreg_ctype()} {", ".join(self.dreg_names())};
    int scale_d;
}};"""

    def rmem_a_struct_def(self):
        sname = self.rmem_a_struct_name()
        return """struct {sname} {{
    {self.areg_ctype()} {", ".join(self.areg_names())};
}};"""

    def cu_utils_ss(self):
        return [
            store_d_util,
            matrix_descriptor_util,
            self.wgmma_ss_function_def(),
        ]

    def cu_utils_rs(self):
        return [
            store_d_util,
            matrix_descriptor_util,
            self.wgmma_rs_function_def(),
        ]

    # fmt: off

    def wgmma_ss_function_name(self):
        return f"exo_Sm90_mma_async_ss_m{self.M}n{self.N}_{self.ptx_dtype}_{self.ptx_atype}_{self.ptx_btype}"

    def wgmma_rs_function_name(self):
        return f"exo_Sm90_mma_async_rs_m{self.M}n{self.N}_{self.ptx_dtype}_{self.ptx_atype}_{self.ptx_btype}"

    def wgmma_ss_function_def(self):
        lines = []
        fname = self.wgmma_ss_function_name()
        params = []
        d_reftype = f"{self.dreg_ctype()}&"

        for m in range(0, self.M, 64):
            params.append(f"uint64_t a_descriptor_m{m}")
        params.append("uint64_t b_descriptor")

        for rname in self.dreg_names():
            params.append(f"{d_reftype} {rname}")

        params.append("int scale_d")

        c = "f"
        assert self.ptx_dtype == "f32", "TODO"

        instr = self.ptx_instr_name()
        lines.append(fr'EXO_CUDA_INLINE void {fname}({", ".join(params)})')
        lines.append(r"{")

        for m in range(0, self.M, 64):
            dreg_names = self.dreg_names(m=m)
            dreg_count = len(dreg_names)
            lines.append(r'  asm volatile("{\n"')
            lines.append(r'  ".reg .pred p;\n"')
            lines.append(rf'  "setp.ne.b32 p, %{dreg_count + 2}, 0;\n"')
            lines.append(rf'  "{instr} "');
            d_vec_template = "{" + ", ".join(f"%{n}" for n in range(dreg_count)) + "}"
            lines.append(rf'  "{d_vec_template}, "')
            lines.append(rf'  "%{dreg_count}, %{dreg_count+1}, p, 1, 1;\n"')
            lines.append(r'  "}"')
            d_vec_args = ", ".join(f'"+{c}"({rname})' for rname in dreg_names)
            lines.append(rf'  : {d_vec_args}')
            lines.append(rf'  : "l"(a_descriptor_m{m}), "l"(b_descriptor), "r"(scale_d)')
            lines.append(r"  );")

        lines.append(r"}")
        return "\n".join(lines)


# fmt: on


class Sm90_RmemMatrixA:
    # TODO implement this

    qual_tl_dict = cuda_rmem_qual_tl_dict | {wgmma_async_instr: wgmma_async_rmem_a_qual}


@memwin_template
def Sm90_RmemMatrixD(M, N):
    helper = WgmmaHelper(M, N, "f32", None, None)

    @window_indexer(RmemIndexer)
    class Sm90_RmemMatrixD(CudaBasicDeviceVisible):
        @classmethod
        def global_(cls):
            return helper.rmem_d_struct_def()

        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: [wgmma_async_rmem_d_qual, wgmma_zero_qual]
        }

        @classmethod
        def device_permission(cls, device, instr_tl):
            return cls.device_allocated_impl(device, instr_tl)

        @classmethod
        def native_unit(cls):
            return cuda_warpgroup

        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            shape = cls.as_const_shape(new_name, shape, srcinfo, min_dim=2)
            array_shape = shape[:-2]
            assert prim_type == "float"  # TODO
            sname = helper.rmem_d_struct_name()
            arrays = "".join(f"[{s}]" for s in array_shape)
            return f"{sname} {new_name}{arrays};"

        @classmethod
        def free(cls, new_name, prim_type, shape, srcinfo):
            return ""

        @classmethod
        def packed_tensor_shape(cls, typ):
            return (M, N)

    return Sm90_RmemMatrixD


__all__.append("Sm90_RmemMatrixA")
__all__.append("Sm90_RmemMatrixD")


class RmemIndexer(WindowIndexer):
    def index(self, utils, features: WindowFeatures):
        code = features.get_dataptr()
        for i in range(features.n_array_dims()):
            code = code[features.get_array_offset(i)]
        return self.pack_result(code, False)


class mma_async_impl(InstrInfo):
    __slots__ = ["helper"]

    def instance_impl(self, M, N, ptx_dtype, ptx_atype, ptx_btype):
        helper = WgmmaHelper(M, N, ptx_dtype, ptx_atype, ptx_btype)
        self.helper = helper
        self.instr_tl = wgmma_async_instr
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()
        self.access_info["a"].out_of_order = True
        self.access_info["b"].out_of_order = True
        self.access_info["d"].out_of_order = False
        self.access_info["d"].mem = Sm90_RmemMatrixD(M, N)

    def codegen(self, args):
        helper = self.helper
        fname = "exo_CudaUtil::" + helper.wgmma_ss_function_name()
        lines = []
        lines.append(f"{fname}(")
        for m in range(0, args.M, 64):
            ref = args.a.index(for_wgmma=True)
            strides = args.a.to_strides_as_packed()
            lines.append(
                f"  exo_CudaUtil::exo_matrix_descriptor(&{ref}, {strides[0]}, {m}),"
            )
        ref = args.b.index(for_wgmma=True)
        strides = args.b.to_strides_as_packed()
        lines.append(f"  exo_CudaUtil::exo_matrix_descriptor(&{ref}, {strides[0]}),")
        d = args.d.index()
        lines.append("  " + "".join(f"{d}.{rname}," for rname in helper.dreg_names()))
        lines.append(f"  {d}.scale_d);")
        lines.append(f"{d}.scale_d = 1;")
        return lines


# For a wgmma D-matrix (in RMEM), set the scale-d flag to 0, so
# the NEXT wgmma.mma.async instruction will zero-initialize D.
# This is modelled in Exo as a zero-clear, even though the effect
# does not actually happen unless a subsequent mma.async occurs.
# We use the wgmma_zero_instr instr-tl to model this.
@instr
class Sm90_zero_scale_d_f32:
    def behavior(M: size, N: size, d: [f32][M, N]):
        for m in seq(0, M):
            for n in seq(0, N):
                d[m, n] = 0

    def instance(self, M, N):
        self.instr_tl = wgmma_zero_instr
        self.coll_unit = cuda_warpgroup
        self.access_info["d"].mem = Sm90_RmemMatrixD(M, N)
        self.access_info["d"].out_of_order = False

    def codegen(self, args):
        return [f"{args.d.index()}.scale_d = 0;"]


__all__.append("Sm90_zero_scale_d_f32")


@instr
class Sm90_mma_async_tf32(mma_async_impl):
    def behavior(
        M: size,
        N: size,
        d: [f32][M, N],  # @ Sm90_RmemMatrixD
        a: [f32][M, 8] @ Sm90_SmemSwizzled(128),
        b: [f32][N, 8] @ Sm90_SmemSwizzled(128),
    ):
        assert M >= 64
        assert M % 64 == 0
        assert N >= 8
        assert N % 8 == 0
        assert stride(a, 1) == 1
        assert stride(b, 1) == 1

        # The 32 is swizzle / sizeof(Element)
        # Basically, this has to match the inner 2 dimensions of a TMA copy.
        # This would get harder if we parameterize this.
        assert stride(a, 0) == 32
        assert stride(b, 0) == 32

        for m in seq(0, M):
            for n in seq(0, N):
                for k in seq(0, 8):
                    d[m, n] += a[m, k] * b[n, k]

    def instance(self, M, N):
        self.instance_impl(M, N, "f32", "tf32", "tf32")


__all__.append("Sm90_mma_async_tf32")


class Sm90_mma_store_d_impl(InstrInfo):
    __slots__ = ["helper", "col_major"]

    def instance_impl(self, helper, col_major):
        self.helper = helper
        self.col_major = 1 if col_major else 0
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()
        self.access_info["src"].mem = Sm90_RmemMatrixD(helper.M, helper.N)

    def codegen(self, args):
        lines = []
        dst = str(args.dst)
        src = args.src.index()
        lines.append("{")
        lines.append(f"  const auto exo_Sm90_dst = {dst};")
        lines.append(f"  const int32_t exo_Sm90_M_end = int32_t({args.M_end});")
        lines.append(f"  const int32_t exo_Sm90_N_end = int32_t({args.N_end});")
        for m in range(0, args.M, 64):
            for reg_index, reg_name in enumerate(self.helper.dreg_names(m=m)):
                lines.append(
                    f"  exo_CudaUtil::exo_Sm90_store_d_reg<{self.col_major}, {reg_index}>(exo_Sm90_dst, {src}.{reg_name}, {m}, exo_Sm90_M_end, exo_Sm90_N_end);"
                )
        lines.append("}")
        return lines


@instr
class Sm90_mma_store_d_col_major_tf32(Sm90_mma_store_d_impl):
    def behavior(
        M: size,
        N: size,
        M_end: index,
        N_end: index,
        dst: [f32][N, M] @ CudaDeviceVisibleLinear,
        src: [f32][M, N],  # Sm90_RmemMatrixD
    ):
        for m in seq(0, M):
            if m < M_end:
                for n in seq(0, N):
                    if n < N_end:
                        dst[n, m] = src[m, n]  # Transposed

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, True)


@instr
class Sm90_mma_store_d_row_major_tf32(Sm90_mma_store_d_impl):
    def behavior(
        M: size,
        N: size,
        M_end: index,
        N_end: index,
        dst: [f32][M, N] @ CudaDeviceVisibleLinear,
        src: [f32][M, N],  # Sm90_RmemMatrixD
    ):
        for m in seq(0, M):
            if m < M_end:
                for n in seq(0, N):
                    if n < N_end:
                        dst[m, n] = src[m, n]

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, False)


__all__.append("Sm90_mma_store_d_col_major_tf32")
__all__.append("Sm90_mma_store_d_row_major_tf32")
