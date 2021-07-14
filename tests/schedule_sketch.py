
# TODO THINK:
# Effects?
# Sanity check on the backend
import GEMMINI from SYS_ATL.libs.state 
@instr("gemmini_extended3_config_ld({a}, {b}, {GEMMINI.c}, {d}, {e});")
def extended_load_config(
        a : ..,
        b : ..,
        c : bool,
        d : bool
        ):

    GEMMINI.act = act
    # 
    GEMMINI.d = d
    GEMMINI.e = ...
    .....


@instr("gemmini_extended3_config_ld({a}, {b}, {c}, {d});")
def extended_load_config(
        a : ..,
        b : ..,
        c : bool,
        d : bool,
        state : context
        ):

    state.c = True
    .....


@proc
def bar(n : size, src : [R][n]):
  for i in par(0,n):
    src[i] = 1.0

@proc
def foo(x : R[10, 20, 30]):
  for j in par(0, 30):
    x[0, 1, j] = 1.0

# ->
def foo(x : R[10, 20, 30]):
  bar(30, x[0, 1, 0:30])




# Bit 32 ---- Should always be 1 for accumulator address
# Bit 31 ---- If 0, overwrite data in the accumulator. Relevant for matmul and mvin. For mvout it's don't care.
# Bit 30 ---- If 0, mvout 8 bit data from accum. If 1, mvout 32 bit data. This is relevant only in mvout, so for mvin and matmul bit 30 is don't care.

# Unification memo
_gemm_ld_i8   = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8)
def ld_i8(
    n   : size,
    m   : size,
    src : [i8][n, m] @ DRAM,
    dst : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    # unknown
    # n, m, dst, src
    for i in par(0, n):
        for j in par(0, m):
            dst_[i+klo,j+llo] = src_[i+ilo,j+jlo]

    ld_i8(n_, m_, src_[ilo:ihi, jlo:jhi], dst_[klo:khi, llo:lhi])

    # body
    # known
    # AG, A, i, k
    for i_in in par(0, 16):
        for k_in in par(0, 16):
            AG[i_in,k_in] = A[16 * i + i_in, 16 * k + k_in]


    # stmt : syntactic match
    # numeric : syntactic match
    # sizes, index : linear equations equality pysmt
    # boolean (affine (non)equality) : syntactic match
    # Solve the problem and check the result for now

    # TODO think:
    # asserts, windowing, inequality for sizes??



# ------- Tensor expression test ------

def gen_tensor():
    @proc
    def tensor(n : size, m : size,
               x : R[n+m], y: R[n+m], res : R[n+m]):
        for i in par(0,n):
            res[i] = x[i] + y[i]
        for j in par(0,m):
            res[n+j] = x[n+j] + y[n+j]

    return tensor
@pytest.mark.skip
def test_tensor():
    tensor = gen_tensor()
    assert type(tensor) is Procedure

    filename = "test_tensor"

    tensor.compile_c(TMP_DIR, filename)

# -------- Support partial windowing call ------
def gen_dot():
    @proc
    def dot(m: size, x : [f32][m] , y : [f32][m] , r : f32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    return dot


def gen_proj(dot):
    @proc
    def proj(n : size, m : size, x : f32[n,m], y : f32[m,n]):
        xy : f32
        y2 : f32
        dot(m, x[1,:], y[:,2], xy)
        dot(m, y[:,3], y[:,3], y2)
        mv_gemmini(x[i:i+16,j:j+16])
        # WindowExpr( sym base, w_access *idx ) -- oh wait, this is broken
        # UAST
        # w_access = Interval( expr? lo, expr? hi )
        #          | Point( expr pt )
        # LoopIR
        # w_access = Interval( expr lo, expr hi )
        #          | Point( expr pt )
        #
        s : f32
        s = xy / y2
        for i in par(0,n):
            x[i] = s * y[i]

    return proj

@pytest.mark.skip
def test_normalize():
    dot  = gen_dot()
    proj = gen_proj(dot)

    assert type(dot) is Procedure
    assert type(proj) is Procedure

    filename = "test_proj_partial"

    proj.compile_c(TMP_DIR, filename)

def gen_assign():
    @proc
    def assign(n : size, m : size, x : f32[n, m]):
        #y : f32[m]
        # WindowStmt( sym lhs, WindowExpr window_e )
        y = x[:, 0]
        # y
        # y --> (float *)
        # y : R[n] windowed from (x : R[n,m]) by x[0:n,0]
        # y : R[n,1] windowed from (x : R[n,m]) by x[0:n,0:1]
        #
        # y : R[n]    ----    i.e. y is a tensor of R-values and dim (n)
        #               --> <R>*
        # y : [R][n,m] ----- i.e. y is a window of R-values and dim (n)
        #               --> struct win_R_2 { <R>* data; int strides[2]; }
        # [R][n,m]
        # i8 f32
        #  R  f32
        #
        # ____ = z : R[n,m] windowed from (x : R[n,m,p]) by x[:,:,3]
        # y : R[n] windowed from (_____) by z[0:n,0]
        #
        #   (x[ilo:ihi, jlo:jhi, 3])[iilo:iihi, 5]
        #
        #   x[ ilo+iilo:ilo+iihi, jlo+5, 3 ]
        #
        # WindowType = ( type base_type, expr* dims,
        #                TensorType orig_tensor,
        #                WindowExpr window )
        #
        #   hi - lo : index
        #
        #   (d0, d1, d2)     --> strides = (d1*d2, d2, 1)
        #   [:,1,:]
        #                    --> strides = (d1*d2,1)
        #
        # --> float *y = x;
        # --> int y_stride = m;
        # --> what about strides?
        y[2] = 3.0
        # --> y[ (2)*m ] = 3.0;
        z : f32[3]
        z = y[0 : 3]
    return assign

@pytest.mark.skip
def test_assign():
    assign = gen_assign()
    filename = "test_assign"
    assign.compile_c(TMP_DIR, filename)
"""
static void tiled_resadd(const size_t I, const size_t J,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    gemmini_config_st(J * sizeof(elem_t));
    gemmini_config_ex(WS, relu ? RELU : NO_ACTIVATION, 0, C_scale, 0);

    gemmini_extended4_config_ld(J * sizeof(elem_t), A_scale, true, DIM, 0);
    gemmini_extended4_config_ld(J * sizeof(elem_t), B_scale, true, DIM, 1);

    for (size_t i = 0; i < I; i += tile_I) {
        for (size_t j = 0; j < J; j += tile_J) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            sp_tiled_resadd(I_tile, J_tile,
                    A_scale, B_scale, a, b, c,
                    J, J, J,
                    relu);
        }
    }

    gemmini_fence();
}
"""

"""
static void tiled_matmul_auto(
        size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type);
scale_t = f32
"""

# C == D
# A == B
# A == D, B == D if C != D

# implements:
#   A READ, B READ, D READ, C WRITE
#       This code will zero out C before accumulating
#   C = A * B + D

# A has dim_I rows and dim_K columns
# B has dim_K rows and dim_J columns
# C has dim_I rows and dim_J columns
# D has (repeating_bias ? 1 : dim_I) rows and dim_J columns

# stride is the second dimension
# stride_A = 500
# 0 to dim_K-1 has the first row of A
# 500 to 500+dim_K-1 has the second row of A
# for i in par(0, dim_I):
#     for k in par(0, dim_K):
#         a = A + i * stride_A + k;


# questions for Hasan:
#   are strides measured in n_elements or in bytes?

#   C = (A_scale_factor * A * B_scale_factor * B + D_scale_factor * D) * scale

# act is activation function. 0 means no activation. 1 means RELU
# Scan C and if <0 then put 0

# ignore relu6_shift (deprecated)
# repeating_bias says that all rows of D are the same, so D only needs to have one row
# Make sure we copy D dim_I times to scratchpad
# call gemmini_extended_config_ex with A_transpose == 1 before calling
# gemmini_extended_preload.
# gemmini_extended_config_ex(dataflows, act, sys_shift, acc_scale, relu6_shift, A_stride, A_transpose, B_tranpose)
#                            dataflows -> 0 : OS, 1 : WS
#                            act       -> 0 : no activation, 1 : RELU
#                            sys_shift -> 0 (deprecated)
#                            acc_scale -> type f32, same as "scale" in tiled_matmul_auto
#                            relu6_shift -> 0 (deprecated)
#                            A_stride  -> 1 (default), # of rows between A's rows
#                            A_transpose -> 1 please transpose, 0 not
#                            B_transpose -> 1 please transpose, 0 not


# full_C -> False : C is elem_t (int8), True: C is acc_t (int32)
# low_D  -> False : D is acc_t (int32), True: D is elem_t (int8)
# weightA -> set to 3 all the time? for each element of B, move in weightA elements of A.
#            After moving in all of A, just moving in remaing B
# tiled_matmul_type := 0 -> OS | 1 -> WS | 2 -> CPU
@proc
def tiled_matmul_auto(
    dim_I : size, dim_J : size, dim_K : size,
    A, B, D, C, # data # TODO:
    stride_A-D,
    A-D_scale_factor,
    act, # 0 means no RELU, 1 means RELU
    scale,
    relu6_shift, # deprecated (use 0 as value)
    repeating_bias, # boolean, if true all rows of D are the same
    transpose_A, # ok duh
    transpose_B, # ok duh
    full_C,     # boolean, false = C has type i8; true = C has type i32
    low_D,      # boolean, false = D has type i32; true = D has type i8
    weightA,    # set to 3; do we need to worry?
    tiled_matmul_type   # 0 -> OS, 1 -> WS, 2 -> CPU   (can ignore)
):
    pass

# actual matmul...
def gen_matmul(use_relu, D_repeat, transpose_A, transpose_B):
    assert not use_relu
    assert not D_repeat
    assert not transpose_A
    assert not transpose_B
    @proc
    def matmul(
        dim_I : size, dim_J : size, dim_K : size,
        A : [R][dim_I,dim_K],
        B : [R][dim_K,dim_J],
        C : [R][dim_I,dim_J],
        D : [R][dim_I,dim_J],
        scale_A : R,
        scale_B : R,
        scale_C : R,
        scale_D : R,
    ):
        for i in par(0,dim_I):
            for j in par(0,dim_J):
                C[i,j] = scale_C * scale_D * D[i,j]
                for k in par(0,dim_K):
                    C[i,j] += scale_C * ((scale_A * A[i,k]) *
                                         (scale_B * B[k,j]))

    return matmul

"""
static void tiled_matmul_auto(
        size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type)
{
    if( act == 0 && relu6_shift == 0 && repeating_bias == 0 &&
        transpose_A == 0 && transpose_B == 0 && full_C == 1 && low_D == 0 &&
        weightA == 3 && tiled_matmul_type == 2 )
    {
        matmul_systl(dim_I, dim_J, dim_K,
                     systl_win_2f32 { A, {stride_A,1} },
                     systl_win_2f32 { B, {stride_B,1} },
                     systl_win_2f32 { C, {stride_C,1} },
                     systl_win_2f32 { D, {stride_D,1} },
                     A_scale_factor, B_scale_factor,
                     scale, D_scale_factor);
    }
}
"""






"""
API:

static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type)

static void tiled_conv_A_stride_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type)

static void tiled_conv_downsample(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,

        enum tiled_matmul_type_t tiled_conv_type)

static void tiled_conv_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type)

static void tiled_resadd_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type)

static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type)

"""

def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld({dst.strides[0]}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( {src.data}, ((uint32_t) {dst.data}),"+
                                  "{m}, {n} );")
    def gemmini_ld(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert n <= 16
        assert m <= 16
        #assert stride(src, 1) == 1
        #assert stride(dst, 0) == 16
        #assert stride(dst, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    return gemmini_ld

    """
    { P } S { Q }

    # if x is an order-m tensor (m=1 vector, m=2 matrix, etc.)
    # then let 0 <= i < m index the i-th stride
    # let n be an arbitrary value for the stride in the i-th dimension
    # stride(x, 0, n1) /\ stride(x, 1, n2) /\ .... /\ stride(x, i, ni)
    ------- { stride(x, i, n) }
    xx = x[...] # (if k indices before the ith index are point indices)
    ------- { stride(xx, i-k, n) }

    # e.g.
    ------- { stride(x, 1, n) }
    xx = x[31,:]
    ------- { stride(xx, 0, n) }

    # other basic axiom about strides
    ------- {}
    x : R[n_0,n_1,...,n_m-1]
    ------- { stride(x, 0, n_1*n_2*...) /\ ... /\ stride(x, m-1, 1) }
    return [x[-1], 1]


    x : R[n,m/16,16] @ GEMM_SCRATCH
    def get_alloc_strides(dims=[None, None, 16]):
        strides = [ (len(dims)-1, 1)) ]
        sz      = 1
        for i,d in reversed(enumerate(dims)):
            if d is not None:
                sz = sz * d
                strides.push_front((i-1,d))
            else:
                break

        return strides

    #return [(len(x)-2,16),(2,1)]
        #   (d0, d1, d2)     --> strides = (d1*d2, d2, 1)
        #   [:,1,:]
        #                    --> strides = (d1*d2,1)


    gemmini_ld(n,m,xx,yy)
    """


# A, B, C: scratchpad addresses
# gemmini_extended_preload(B, C, B_cols, B_rows, C_cols, C_rows);
# gemmini_extended_compute_preloaded(A, ~((0(uint32_t))), A_cols, A_rows, 16, 16);

# 1 <= *_cols <= 16
# 1 <= *_rows <= 16

# A and B are padded with zeros until 16
# C is masked out

# Gemmini matmuls compute C = A * B
# Gemmini-supported dataflows: OS mode, WS mode
# (0 := Output-stationary)
# 1 := Weight-stationary
# In WS mode, we preload B into the systolic array
# (In OS mode, we preload zeroes into the systolic array)
# FOR NOW, USE WS
def gen_matmul():
    @proc
    @instr("gemmini_extended_preload("+
                "({B}) + ({B_row_off}), (({C}) + ({C_row_off})) | (1 << 30), "+
                "{M}, {K}, "+
                "{M}, {N}"+
           ");\n"+
           "gemmini_extended_compute_preloaded("+
                "({A}) + ({A_row_off}), ~((uint32_t)0), "+
                "{K}, {N}, "+
                "16, 16"+
           ");")
    def gemmini_matmul_acc(
        N : size,
        M : size,
        K : size,
        A_row_off  : index,
        B_row_off  : index,
        C_row_off : index,
        nA : size,
        nB : size,
        nC : size,
        A : R[nA,16] @ GEMM_SCRATCH,
        B : R[nB,16] @ GEMM_SCRATCH,
        C : R[nC,16] @ GEMM_ACC,
        1 <= N <= 16,
        1 <= M <= 16,
        1 <= K <= 16
    ):
        for i in par(0,N):
            for j in par(0,M):
                for k in par(0,K):
                    C[C_row_off+i, j] += A[A_row_off+i, k] * B[B_row_off+k, j]


    @proc
    @instr("gemmini_extended_preload("+
                "({B}) + ({B_row_off}), ({C}) + ({C_row_off}), "+
                "{M}, {K}, "+
                "{M}, {N}"+
           ");\n"+
           "gemmini_extended_compute_preloaded("+
                "({A}) + ({A_row_off}), ~((uint32_t)0), "+
                "{K}, {N}, "+
                "16, 16"+
           ");")
    def gemmini_matmul(
        N : size,
        M : size,
        K : size,
        A_row_off  : index,
        B_row_off  : index,
        C_row_off : index,
        nA : size,
        nB : size,
        nC : size,
        A : R[nA,16] @ GEMM_SCRATCH,
        B : R[nB,16] @ GEMM_SCRATCH,
        C : R[nC,16] @ GEMM_ACC,
        1 <= N <= 16,
        1 <= M <= 16,
        1 <= K <= 16
    ):
        for i in par(0,N):
            for j in par(0,M):
                C[C_row_off+i, j] = 0.0
                for k in par(0,K):
                    C[C_row_off+i, j] += A[A_row_off+i, k] * B[B_row_off+k, j]

def gen_zeromem():
    @proc
    @instr("gemmini_extended_mvin( NULL, ({x}) + ({off}) );")
    def gemmini_zeromem(
        N       : size,
        N_rows  : size,
        off     : index,
        x : R[N_rows,16] @ GEMM_SCRATCH,
        1 <= N <= 16
    ):
        for i in par(0,N):
            for j in par(0,16):
                x[i,j] = 0.0

    @proc
    @instr("gemmini_extended_mvin( NULL, ({x}) + ({off}) );")
    def gemmini_zeromem_acc(
        N       : size,
        N_rows  : size,
        off     : index,
        x : R[N_rows,16] @ GEMM_ACC,
        1 <= N <= 16
    ):
        for i in par(0,N):
            for j in par(0,16):
                x[i,j] = 0.0


def gen_store():
    @proc
    @instr("gemmini_config_st(4 * {dst_m});\n"+
           "gemmini_extended_mvout( "+
                "({dst}) + ({dst_r})*({dst_m}) + ({dst_c}),"+
                "({src}) + ({src_r}) );")
    def gemmini_st(
        src_n : size,
        src_r : index,
        dst_n : size,
        dst_m : size,
        dst_r : index,
        dst_c : index,
        col_dim : size,
        row_dim : size,
        #scale : R @ IN @ DRAM,
        src : R[src_n,16]    @ OUT  @ GEMM_SCRATCH,
        dst : R[dst_n,dst_m] @ IN   @ DRAM,
        1 <= col_dim <= 16,
        1 <= row_dim <= 16
    ):
        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, dst_c + j] = src[src_r + i, j]


    @proc
    @instr("gemmini_config_st(4 * {dst_m});\n"+
           "gemmini_extended_mvout( "+
                "({dst}) + ({dst_r})*({dst_m}) + ({dst_c}),"+
                "({src}) + ({src_r}) );")
    def gemmini_st_acc(
        src_n : size,
        src_r : index,
        dst_n : size,
        dst_m : size,
        dst_r : index,
        dst_c : index,
        col_dim : size,
        row_dim : size,
        #scale : R @ IN @ DRAM,
        src : R[src_n,16]    @ OUT  @ GEMM_ACC,
        dst : R[dst_n,dst_m] @ IN   @ DRAM,
        1 <= col_dim <= 16,
        1 <= row_dim <= 16
    ):
        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, dst_c + j] = src[src_r + i, j]

    # gemmini_extended_mvout(dram_addr: uint64_t (pointer), spad_addr: uint32_t, cols: uint16_t, rows: uint16_t)
    # 1 <= cols <= 16
    # 1 <= rows <= 16
    # 1 is WS; we are accumulating over accumulator
    # gemmini_config_ex(1, 0, 0, 1.0, 0) # This is only useful for moving data out from the Accumulator
    # gemmini_config_st(stride: bytes)

    # notes on GEMMINI ADDRESS SPACE
    #
    #   - addresses going into the accumulator have MSB (Bit 32)=1
    #       i.e. set addr = addr | 0x80000000
    #
    #   - to overwrite what's in the accumulator (Bit 31)=0
    #   - to accumulate on top of what's already in the acc. (Bit 31)=1
    #   - to mvout 8-bit data from acc.,  (Bit 30)=0
    #   - to mvout 32-bit data from acc., (Bit 30)=1

    return gemmini_st

# Scratch for GEMMINI
def gen_load():
    @proc
    @instr("gemmini_extended3_config_ld(4 * {src_m}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( "+
                "({src}) + ({src_r})*({src_m}) + ({src_c}),"+
                "({dst}) + ({dst_r}) );")
    def gemmini_ld(
        src_n : size,
        src_m : size,
        src_r : index,
        src_c : index,
        dst_n : size,
        dst_r : index,
        col_dim : size,
        row_dim : size,
        #scale : R @ IN @ DRAM,
        src : R[src_n,src_m] @ IN  @ DRAM,
        dst : R[dst_n,16]    @ OUT @ GEMM_SCRATCH,
        1 <= col_dim <= 16,
        1 <= row_dim <= 16
    ):
        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, j] = src[src_r + i, src_c + j]

    @proc
    @instr("gemmini_extended3_config_ld(4 * {src_m}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( "+
                "({src}) + ({src_r})*({src_m}) + ({src_c}),"+
                "({dst}) + ({dst_r}) );")
    def gemmini_ld_acc(
        src_n : size,
        src_m : size,
        src_r : index,
        src_c : index,
        dst_n : size,
        dst_r : index,
        col_dim : size,
        row_dim : size,
        #scale : R @ IN @ DRAM,
        src : R[src_n,src_m] @ IN  @ DRAM,
        dst : R[dst_n,16]    @ OUT @ GEMM_ACC,
        1 <= col_dim <= 16,
        1 <= row_dim <= 16
    ):
        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, j] = src[src_r + i, src_c + j]


    # gemmini_extended3_config_ld((stride, scale, shrunk, id)
    # - stride is stride between rows, in bytes, in DRAM
    #       stride is uint64
    # - scale  is floating point number to multiply by
    #       scale is fp32
    # - shrunk is 0 or 1: if true, load 8-bit data into the accumulator
    #                     if false, load 32-bit data
    # - id     is which # load instr you're setting config for (use 0)

    # gemmini_extended_mvin( src, dst, col_dim in #elems, row_dim in #elems )
    # - src (DRAM), dst (scratchpad) are pointers/uint64
    # - col_dim, row_dim are uint16
    # - 1 <= col_dim <= (is a param; 64 in practice for 8-bit, 16 for 32-bit)
    # - 1 <= row_dim <= 16
    # - dead space on columns is garbage

    # gemmini_extended_mvin(x + nx*DIM, y + nx*DIM, DIM, DIM)
    # gemmini_extended_mvin2(x + nx*DIM, y + nx*DIM, DIM, DIM)
    # gemmini_extended_mvin3(x + nx*DIM, y + nx*DIM, DIM, DIM)

    # config chooses the stride of the mvin
    # 3 different possible strides: (not fixed in the language, e.g. ...)
    #   - weights
    #   - image
    #   - bias  (output)

    return gemmini_ld

def gen_gemm_app():
    @proc
    def gemm_app( n : size, x : R[n] @ IN, y : R[n] @ OUT):
        for i0 in par(0,n/16):
            if i0 == n/16-1:
                for i1 in par(0,n%16):
                    y[i0*16+i1] = x[i0*16+i1]
            else:
                load(n, 16, i0, x, y)
                #for i1 in par(0,16):
                #    y[i0*16+i1] = x[i0*16+i1]

    return gemm_app

def test_alloc():
    gemm_app = gen_gemm_app()
    load  = gen_load()
    gemm_app = gemm_app.abstract(load, "for _ in par(0,16): _")
    #alloc.compile_c(directory, "test_alloc")
    #print(alloc)


# generaated C
// alloc(
//     n : size,
//     x : R[n] @IN,
//     y : R[n] @OUT
// )
void alloc( int n, float* x, float* y ) {
for (int i0=0; i0 < _ceil_div(n, 16); i0++) {
  if (i0 == _ceil_div(n, 16) - 1) {
    for (int i1=0; i1 < n % 16; i1++) {
      y[i0 * 16 + i1] = x[i0 * 16 + i1];
    }
  } else {
    gemmini_extended_mvin(i1 + i0*16, i1 + i0*16, 16, 16)
    //for (int i1=0; i1 < 16; i1++) {
    //  y[i0 * 16 + i1] = x[i0 * 16 + i1];
    //}
  }
}
}



# step 1: explore tail strategies w/o GEMMINI
# step 2: extend exploration of tail strategies to consider GEMMINI
@proc
def foo( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i in par(0,n):
        y[i] = x[i] + 1

foo = foo.split(i,16,[i0,i1],"split_case0")
         .instr(i1, GEMMINI)

    foo.split()
       .peel_off_last_iter(i0)
       .simplify(i1[0])

@proc
def foo_split_guard( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        for i1 in par(0,16):
            if i0*16 + i1 < n:
                y[i0*16+i1] = x[i0*16+i1] + 1

#peel_off_last_iter ->>
    for i0 in par(0,ceil(n/16)-1):
        for i1 in par(0,16):
            if i0*16 + i1 < n:
                y[i0*16+i1] = x[i0*16+i1] + 1
    i0' = ceil(n/16)-1
    for i1 in par(0,16):
        if i0'*16 + i1 < n:
            y[i0'*16+i1] = x[i0'*16+i1] + 1

#simplify(i1[0]) ->>
    for i0 in par(0,ceil(n/16)-1):   i0 < ceil(n/16) - 1
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1
    i0' = ceil(n/16)-1
    for i1 in par(0,16):
        if i0'*16 + i1 < n:
            y[i0'*16+i1] = x[i0'*16+i1] + 1


@Proc
Foo( n : size, m : size, p : size )
    RESERVE SPACE globmem in DRAM

    buf : R[n]
        --> elem* buf = globmem

    A   : R[m]
        --> elem* A   = globmem + n

    for k in ...:
        B : R[p]
        --> elem* B   = globmem + n + m
        for j
            B = A lbuf
        for j
            x += B
        free B
        --> noop;

    free A
    free buf

    FREE globmem

required_bytes = Query_Foo_memory(n,m,p)
globmem = my_globmem + offset;
Foo_manual_mem(globmem, n,m,p, ...)

Foo(n,m,p, ...)


A : R[n,m]

buf : R[n,m] @ GEMMINI_SCRATCHPAD

buf : R[n,16] @ GEMM
instr(GEMM_LD)
for i0 in par(0,16):
    for i1 in par(0,16):
        buf[i0,i1] = input[i0][i1]


=>
gemmini_extended_config_ld(0,1)
gemmini_extended_mvin(input, sp_start_addr, 3, 16)

LOAD ... into buf

Valid GEMMINI
@proc
def foo_split_case0( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        if i0 == ceil(n/16)-1:
            instr(GEMMINI)
            for i1 in par(0,n%16):
                y[i0*16+i1] = x[i0*16+i1] + 1
        else:
            instr(GEMMINI)
            for i1 in par(0,16):
                y[i0*16+i1] = x[i0*16+i1] + 1

Valid GEMMINI
@proc
def foo_split_case1( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        instr(GEMMINI)
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1
    instr(GEMMINI)
    for i1 in par(0,n%16):
        y[(ceil(n/16)-1)*16+i1] = x[(ceil(n/16)-1)*16+i1] + 1

@proc
def foo_split_shift_in( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        i_base = i0*16
        if i0 == ceil(n/16)-1:
            i_base = n - 16
        for i1 in par(0,16):
            i = ibase + i1
            y[i] = x[i] + 1

@proc
def foo_split_shift_in_case1( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,floor((n+1)/16)):
        @instr(HWACHA)
        for i1 in par(0,16):
            y[i0*16 + i1] = x[i0*16 + i1] + 1
    @instr(HWACHA)
    for i1 in par(0,16):
        y[n-16+i1] = x[n-16+i1] + 1

@proc
def foo_split_overcompute( n : size, x : R[16*ceil(n/16)] @ IN, y : R[16*ceil(n/16)] @ OUT):
    for i0 in par(0,ceil(n/16)):
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1



# // <- integer division
#
# conv1d.reorder(j,i)
# ->
#    for i in par(0,r):
#      for j in par(0,n):
#        if i <= j < i + m:
#          res[i] += x[j]*w[i-j+m-1]
#
# conv1d.split(j,2)
# conv1d.split
# ->
#    for i in par(0,r):
#      res[i] = 0.0
#    for i in par(0,r):
#      for j_1 in par(0,n/2):
#        for j_2 in par(0,2):
#          j = j_1*2 + j_2
#          if j < n:
#            if i <= j < i + m:
#              res[i] += x[j]*w[i-j+m-1]
#
# conv1d.fuse(i)
# ->
#    for i in range(0,r):
#      res[i] = 0.0
#      for j in range(0,n):
#        if i <= j < i + m:
#          res[i] += x[j]*w[i-j+m-1]
#
# conv1d.unroll(i,3)
# conv1d.split(i,3,i1,i2).unroll(i_2)
# ->
# TODO: What's r/3?
#    for i in range(0,r/3):
#      res[i*3+1] = 0.0
#      res[i*3+2] = 0.0
#      res[i*3+3] = 0.0
#    for i in range(0,r/3):
#      for j in range(0,n):
#        if i*3+1 <= j < i*3+1 + m:
#          res[i*3+1] += x[j]*w[i*3+1-j+m-1]
#      for j in range(0,n):
#        if i*3+2 <= j < i*3+2 + m:
#          res[i*3+2] += x[j]*w[i*3+2-j+m-1]
#      @instr(GEMMINNI_MATMUL)
#      for j in range(0,n):
#        if i*3+3 <= j < i*3+3 + m:
#          res[i*3+3] += x[j]*w[i*3+3-j+m-1]
#

"""
class Instruction:
    def compile(self,...):
        raise NotImplementedError("compile unimplemented")

    def check(self,...):
        raise NotImplementedError("check unimplemented")


class GEMMINNI_MATMUL(Instruction):
    def __init__(self):
        pass

    def compile(self,...):
        return "some string"

    # do this during typechecking?
    def check_LoopIR(self,...):
        pass

    # do this during instruction-validity checking
    def check_Instr(self,...):
        pass
"""



#
#
# TODO: .parallel() and .vectorize() are not rewrite operations?
#       Need to introduce notation which is not a Loop IR
#
# Idea on vectorize:
#def foo():
#    ...
#    @instr(AVX_add)
#    for i in par(0,n):
#        A[i+j] = B[i+j] + C[i+j]
#
##-->
#    for i_hi in par(0,(n+1)/4):
#        @instr(AVX_add)
#        for i_lo in par(0,4):
#            A[4*i_hi + i_lo + j] = B[...] + C[...]
#
#foo = foo.instr('i','AVX_add')

    # In the future..
    #@sched(blur)
    #def tiled_blur():
    #    j_hi, j_lo = split(j[1], 2)
    #    i_hi, i_lo = split(i[1], 2)
    #    reorder(i_lo,j_hi)

"""
instruction { .... }

instr(GEMM_Load)
for i in par(0,n):
    x[i] = y[i]

# basic source
for i :
    alloc A
    # do something with A
    comp1(A)
    alloc B
    # do something with B, but not A
    comp2(B)


for i :
    alloc A
    # do something with A
    comp1(A)
    #alloc B
    # change B to A in the following
    # do something with B, but not A
    comp2(A)


alloc A[2]
for i :
    # do something with A
    comp1(A[i%2])
    #alloc B
    # change B to A in the following
    # do something with B, but not A
    comp2(A[(i+1)%2])

def sub\_mul(size n, size m, )
for i in par(0, n):
  for j in par(0, m):
    res[..] = A[..] * B[..]

-> "GEMM\_MUL(res, A, B, C, D..)"


# source
@proc
def source(...):
    for i in par(0,n):
        z[i] = x[i] + y[i]

# vec-instr
@proc
def add4v( dst, src0, src1, i ):
    for k in par(0,4):
        dst[i+k] = src0[i+k] + src1[i+k]

source = source.split(i,4)

> for i in par(0,n/4):
>     for j in par(0,4):
>         z[4*i+j] = x[4*i+j] + y[4*i+j]

source = source.abstract(for j, add4v)


> for i in par(0,n/4):
>     add4v(?dst, ?src0, ?src1, ?i)



for k in par(0,4):
    dst'[i'+k] = src0'[i'+k] + src1'[i'+k]

=

for j in par(0,4):
    z[j+4*i+0] = x[4*i+j] + y[4*i+j]



k = j
{dst'} = {z}
{src0'} = {x}
{src1'} = {y}
{0 <= j < 4}
{0 <= i < n/4}
{i'+j} = {j+4*i+0}
{i'+j} = {4*i+j}
{i'+j} = {4*i+j}


@proc
def inc( n : size, x : R[n] ):
    for i in par(0,n):
        x[i] = x[i] + 1

@proc
def foo(...):
    ...
    y : R[m]
    # p > m

    z alias= y[p-m:p]

    for j in par(0,m):
        y[(p-m)+j] = y[(p-m)+j] + 1
