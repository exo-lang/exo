include $(abs_top_srcdir)/Makefrag

tests = \
	mvin_mvout \
	mvin_mvout_zeros \
	mvin_mvout_stride \
	mvin_mvout_block_stride \
	mvin_mvout_acc \
	mvin_mvout_acc_zero_stride \
	mvin_mvout_acc_stride \
	mvin_mvout_acc_full \
	mvin_mvout_acc_full_stride \
	matmul_os \
	matmul_ws \
	matmul \
	raw_hazard \
	aligned \
	padded \
	mvin_scale \
	conv \
	conv_with_pool \
	conv_with_rot180 \
	conv_with_kernel_dilation \
	conv_with_input_dilation \
	conv_with_input_dilation_and_rot180 \
	conv_with_input_dilation_and_neg_padding \
	conv_trans_output_1203 \
	conv_trans_weight_1203 \
	conv_trans_weight_0132 \
	conv_trans_input_3120 \
	conv_trans_input_3120_with_kernel_dilation \
	tiled_matmul_os \
	tiled_matmul_ws \
	tiled_matmul_ws_At \
	tiled_matmul_ws_Bt \
	tiled_matmul_ws_full_C \
	tiled_matmul_ws_low_D \
	tiled_matmul_cpu \
	tiled_matmul_option \
	tiled_matmul_ws_perf \
	transpose \
	matrix_add \
	resadd \
	global_average \
	template

tests_baremetal = $(tests:=-baremetal)

ifeq ($(findstring spike,$(RUNNER)),spike)
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif

ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/bareMetalC/$^

junk += $(tests_baremetal) $(tests_linux)

