

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
        src : R[src_n,16]       @ OUT  @ GEMM_SCRATCH,
        dst : R[dst_n,dst_m,16] @ IN   @ DRAM,
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
        src : R[src_n,16]       @ OUT  @ GEMM_ACC,
        dst : R[dst_n,dst_m,16] @ IN   @ DRAM,
        1 <= col_dim <= 16,
        1 <= row_dim <= 16
    ):
        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, dst_c + j] = src[src_r + i, j]

    # gemmini_extended_mvout(dram_addr: uint64_t (pointer), spad_addr: uint32_t, cols: uint16_t, rows: uint16_t)
    # 1 <= cols <= 16
    # 1 <= rows <= 16
    # 2 is WS; we are accumulating over accumulator
    # gemmini_config_ex(2, 0, 0, 1.0, 0) # This is only useful for moving data out from the Accumulator
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
