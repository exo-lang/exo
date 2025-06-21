# Documentation

This directory provides detailed documentation about Exo's interface and internal systems.

- To learn about the design principles of Exo, read [Design.md](Design.md).
- To understand how the Exo system is implemented, read [System.md](System.md).
- For information on writing Exo object code, APIs, and imports, refer to [Procedures.md](Procedures.md), [object_code.md](object_code.md), and [Imports.md](Imports.md).
- To learn how to define **hardware targets externally to the compiler**, refer to [externs.md](externs.md), [instructions.md](instructions.md), and [memories.md](memories.md).
- To learn how to define **new scheduling operations externally to the compiler**, refer to [Cursors.md](./Cursors.md) and [inspection.md](./inspection.md).
- To understand the available scheduling primitives and how to use them, look into the [primitives/](./primitives) directory.
- To learn about metaprogramming as a method for writing cleaner code, see [Metaprogramming.md](Metaprogramming.md).

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
- [ASPLOS '25 paper](https://arxiv.org/abs/2411.07211)
- [Kevin Qian's MEng thesis](https://dspace.mit.edu/handle/1721.1/157187)
- [Samir Droubi's MEng thesis](https://dspace.mit.edu/handle/1721.1/156752)

For more documentation with running Exo code, refer to the [Examples](../examples/README.md) directory.
