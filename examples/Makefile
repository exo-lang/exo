CFLAGS ?= -march=native

avx2_matmul: avx2_matmul.o main.o

avx2_matmul.c: x86_matmul.py
	exocc -o . --stem $(*F) $^

main.c: avx2_matmul.c

.PHONY: clean
clean:
	$(RM) avx2_matmul avx2_matmul.* *.o exo_demo
	$(RM) -r __pycache__/
