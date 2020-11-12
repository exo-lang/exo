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
"""
