# Design document for Exo

Exo is a domain-specific language designed to enable productive development of high-performance kernel libraries that target specialized hardware accelerators.

The key design principles of Exo are:
- Performance transparity: We do not do "magic optimization" that are surprising and opaque to users.
- WYSWYG: Exo IR closely models C-style code and will be trivially lowered to C code.
- Give the performance control back to users

# Exocompilation: Externalizing Hardware Targets

One of the main ideas behind Exo is **exocompilation**, which allows users to define hardware targets externally to the compiler in user-level libraries. This has several advantages:

- Hardware vendors can support new accelerators without maintaining compiler forks
- The cost of adding support for new hardware is significantly reduced
- Proprietary details of hardware can be protected

Users can model custom memories, instructions, and configuration state in libraries to target a specific accelerator. These hardware abstractions can then be used to write hand-optimized code or as building blocks for higher-level scheduling transformations.

More info can be found in the [PLDI paper](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf) and [./instructions.md] and [./memories.md].

## Fine-Grained Primitives for Performance Control

Exo offers a set of fine-grained scheduling primitives that give users low-level control over performance-critical details. These primitives can be composed to build complex transformation schedules. Some examples of these primitives include:

- `split` and `reorder` for loop transformations
- `stage_mem` for explicit data movement between memories
- `replace` for mapping code fragments to custom instructions

Having explicit control over these low-level details enables Exo to achieve performance competitive with highly-tuned vendor libraries and hand-optimized assembly code.
Primitives can be found in [./primitives/].

## Rewrite-based Scheduling Language

Unlike previos popular frameworks like Halide and TVM which uses _lowering based_ compilation process, Exo uses _rewrite based_ compilation process.

This has a few advantages:
- Less magic
- Easy to print in the middle of scheduling process and see what is going on.

# User-Defined Scheduling Operations

While the flexibility of fine-grained primitives is necessary for achieving peak performance, directly using them can be verbose and laborious. To address this, Exo allows users to define new higher-level scheduling operations by composing the core primitives.

These user-defined scheduling operations can encapsulate common optimization patterns and hardware-specific transformations, greatly improving productivity. They can be put together in reusable libraries, further enabling modularity and portability.

More info can be found in the ASPLOS paper and Cursor.md.

## The AIR Framework: Action, Inspection, Reference

We identified that Action, Inspection, and Reference are the key scheduling language design mechanisms that enable user-defined scheduling operations.

- **Actions** are the scheduling primitives that transform the code (e.g., `split`, `reorder`).
- **Inspection** queries properties of the code (e.g., loop bounds, memory access patterns).
- **References** point to specific parts of the code to apply actions to.

Together, AIR allows scheduling operations to be defined as composable rewrites on the code. The language implementation guarantees the correctness of these rewrites with a set of effect analyses.

## Cursors: Enabling Relative References

A novel feature in Exo's design is the concept of cursors, which serve as relative references into the code. Similar to a text editing cursor, an Exo cursor identifies a specific location in the program AST, such as a statement, loop nest, or even the gap between statements.

Cursors support navigation operations such as `next`, `prev`, `parent`, enabling powerful code transformations using relative positions. Multiple cursors can coexist, allowing different parts of the code to be referenced and modified simultaneously.

Using cursors, complex scheduling operations can be built using simple navigation and rewrite rules, with the cursor abstracting away the details of manual AST manipulation.
