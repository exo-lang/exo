PROG = conv1d
OUT = out/
CC = "${RISCV}/bin/clang"
SPIKE = "${RISCV}/bin/spike"
ASFLAGS = -march=rv32imc_xtheadmatrix0p1 -menable-experimental-extensions 
CFLAGS = -O2 -g3 $(ASFLAGS) 

default: sim
exo_comp: exo/conv1d_exo.c

$(OUT)/$(PROG).elf: $(OUT)/$(PROG).o $(OUT)/conv1d_exo.o
	$(CC) $(LDFLAGS) -o $@ $^

$(OUT)/$(PROG).o: main.c exo/conv1d_exo.h conv1Di32.h $(OUT) 
	$(CC) $(CFLAGS) -o $@ -c $<

$(OUT)/conv1d_exo.o: exo/conv1d_exo.c $(OUT)
	$(CC) $(CFLAGS) -o $@ -c $<
	
$(OUT):
	@mkdir -p $(OUT)

exo/conv1d_exo.h: exo/conv1d_exo.c
exo/conv1d_exo.c: exo/conv1d.py
	exocc -o exo/ --stem conv1d_exo exo/conv1d.py

conv1Di32.h: gen_stimuli.py
	python3 $<

sim: $(OUT)/$(PROG).elf
	@$(SPIKE) --isa=RV32IMC_xmatrix pk -s $<