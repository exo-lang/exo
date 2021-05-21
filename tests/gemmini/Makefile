

ROOT=/nscratch/gilbo/chipyard/generators/gemmini/software/gemmini-rocc-tests
BCOMMON=$(ROOT)/riscv-tests/benchmarks/common

CC_BAREMETAL=riscv64-unknown-elf-gcc
CFLAGS_BAREMETAL= \
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
  -I$(ROOT)/riscv-tests \
  -I$(ROOT)/riscv-tests/env \
  -I$(ROOT) \
  -I$(BCOMMON) \
  -DID_STRING= \
  -nostdlib \
  -nostartfiles \
  -static \
  -T $(BCOMMON)/test.ld \
  -DBAREMETAL=1

SYSTL_ROOT=../..
TEST_ROOT=..
TMP_DIR=$(TEST_ROOT)/tmp

COMPILE=$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $(BCOMMON)/*.c $(BCOMMON)/*.S

TEST_LIST=\
  load_16 store_16 ld_st_16 matmul_16 \
  alloc_nest_malloc matmul_16_malloc \
  ld_2d st_2d ld_st_2d matmul_2d

TEST_BINS=$(TEST_LIST:=.test)
TEST_EXECS=$(TEST_LIST:=.run)

all: $(TEST_EXECS)
	@echo "done running all tests"

%.run: %.test
	spike --extension=gemmini $^

%.test: %_main.c $(TMP_DIR)/test_%.c
	$(COMPILE) -I$(TMP_DIR) $^ -o $@

.PHONY: clean
clean:
	rm $(TEST_BINS)


