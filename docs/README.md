# Documentation

This directory provides detailed documentation about the Exo interface and internal system.

- To learn about the design principles of Exo, read [Design.md](Design.md).
- To understand how the Exo system is implemented, read [System.md](System.md).
- For information on writing Exo object code, APIs, and imports, refer to [Procedures.md](Procedures.md), [object_code.md](object_code.md), and [Imports.md](Imports.md).
- To learn how to define memory, instructions, and externs externally to the compiler in the user code, explore [externs.md](externs.md), [instructions.md](instructions.md), and [memories.md](memories.md).
- To understand the available scheduling primitives and how to use them, look into the primitives/ directory.

The scheduling primitives are classified into six categories:

1. [Buffer Transformations](primitives/buffer_ops.md)
2. [Loop and Scope Transformations](primitives/loop_ops.md)
3. [Configuration States](primitives/config_ops.md)
4. [Subprocedure Operations](primitives/subproc_ops.md)
5. [Memory, Precision, and Parallelism Transformations](primitives/backend_ops.md)
6. [Other Operations](primitives/other_ops.md)

# Further Reading

The following papers provide a high-level and holistic view of Exo as a project:

- [PLDI '22 paper](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf)
- [ASPLOS '25 paper](.)
- [Kevin Qian's MEng thesis](https://dspace.mit.edu/handle/1721.1/157187)
- [Samir Droubi's MEng thesis](https://dspace.mit.edu/handle/1721.1/156752)

For more documentation and actual Exo code, refer to the [Examples](../examples/README.md) directory.
