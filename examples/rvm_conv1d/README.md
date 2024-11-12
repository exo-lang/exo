# Conv1D on RVM example

This is an implementation of a simplified 1D convolution routine, using a custom [RISC-V ISA extension called RVM](https://github.com/esl-epfl/xheep_matrix_spec/tree/main).

The tutorial accompanying this example is on [the main website](https://exo-lang.dev/tutorial.html). This page will just show you how to first compile the Exo program to C, and how to run it as well (optional.)

## File organization

* `main.c` - driver program testing handwritten vs Exo routine
* `gen_stimuli.py` - generate C arrays used as test vectors for conv1d routine, with expected output
* `conv1Di32.h` - generated output from `gen_stimuli.py`
* `exo/conv1d.py` - Exo code for conv1d
* `exo/conv1d_exo.{c,d,h}` - generated outputs from Exo


## Setup Exo & Compile

First follow [the documentation](https://github.com/exo-lang/exo#install-exo) to install Exo, if you have not already. We assume `exocc` is in `$PATH`, and you have `make` installed. To compile the exo program in `exo/conv1d.py`, run:

```bash
make exo_comp
```

The resulting C code for the example will be in `exo/conv1d_exo.c`.

From here, if you would like to also compile the program to a RISC-V binary, and run it in a simulator, you will need the custom RVM toolchain. The following steps walk through that process. Otherwise, you can stop here.

## Install RVM toolchain

RVM is the custom RISC-V extension, which supports instructions and registers to do matrix operations. It requires a custom LLVM toolchain to build code, and in order to run programs, a fork of the Spike simulator. [The repo for RVM has a guide to set up these components](https://github.com/esl-epfl/xheep_matrix_spec/blob/main/BUILDING.md). In the end you should have the LLVM tools as well as Spike installed under `$RISCV/bin`.


## Build

Run `make` to build the driver program, and simulate it in spike. **This assumes you have `$RISCV` defined from the installation step.** You should see an output like this:

```
$ make
...
handwritten err: 0
exo err: 0
2350 ticks
93797 cycles
93799 instructions
0.99 CPI
```

Note that the cycle counts are *not* accurate, and they should not be used to measure performance. Unfortunately, the hardware for RVM is not public as of today, and the Spike simulator is not meant to simulate these details, so it is only used for testing functional correctness.