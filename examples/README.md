# Scheduling Examples

This directory contains several examples, along with documentation and code.
If you are new to Exo, we recommend going through the examples in the following order:

1. [AVX2 Matmul](./avx2_matmul/README.md): This example demonstrates how to take a simple matrix multiplication kernel and transform it into an implementation that can make use of AVX2 instructions. It provides an overview of Exo and its scheduling system.

2. [Cursor](./cursors/README.md): This example shows how to use Cursors to efficiently write schedules and define a new scheduling operator.

3. [RVM](./rvm_conv1d/README.md): This example illustrates how to use Exo to define and target a new hardware accelerator entirely in the user code.

4. Quizzes ([quiz1](./quiz1/README.md), [quiz2](./quiz2/README.md), [quiz3](./quiz3/README.md)) contain common scheduling mistakes in Exo and solutions to fix them. The best way to learn a programming language is by debugging code.

