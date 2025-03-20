# Design Document for Exo

Exo is a domain-specific language designed to enable productive development of high-performance kernel libraries that target specialized hardware accelerators.

The key design principles of Exo are:
- **Performance Transparency**: We do not do "magic optimizations" that are surprising and opaque to users.
- **WYSIWYG**: Exo IR closely models C-style code and will be trivially lowered to C code.
- **User Control**: Give the performance control back to users.

---

# Exocompilation: Externalizing Hardware Targets

One of the main ideas behind Exo is **exocompilation**, which allows users to define hardware targets externally to the compiler in user-level libraries. This has several advantages:

- Hardware vendors can support new accelerators without maintaining compiler forks.
- The cost of adding support for new hardware is significantly reduced.
- Proprietary details of hardware can be protected.

Users can model custom [memories](./memories.md), [instructions](./instructions.md), and configuration state in libraries to target a specific accelerator. These hardware abstractions can then be used to write hand-optimized code or as building blocks for higher-level scheduling transformations.

More info can be found in the [PLDI paper](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf), [instructions.md](./instructions.md), and [memories.md](./memories.md).

## Fine-Grained Primitives for Performance Control

Exo provides a set of fine-grained scheduling primitives that offer users low-level control over performance-critical aspects. These primitives can be combined to create complex transformation schedules. Some examples of these primitives include:

- `replace`: Maps code fragments to custom instructions
- `delete_config`: Removes redundant configuration statements

The key research contributions of Exo were supporting `replace` through unification and the ability to reason about configuration states. Explicit control over these low-level details allows Exo to achieve performance comparable to highly-tuned vendor libraries and hand-optimized assembly code. All the primitives can be found in the [primitives/](./primitives/) directory.

## Rewrite-based Scheduling Language

Exo employs a *rewrite-based* compilation process, which differs from the *lowering-based* approach used by popular frameworks like Halide and TVM.

The rewrite-based approach offers several advantages:

- Reduced complexity and less "magic" involved
- Easier to print and inspect the state of the scheduling process at any point

---

# User-Defined Scheduling Operations

While the flexibility of fine-grained primitives is necessary for achieving peak performance, directly using them can be verbose and laborious. To address this, Exo allows users to define new higher-level scheduling operations by composing the core primitives.

These user-defined scheduling operations can encapsulate common optimization patterns and hardware-specific transformations such as auto-vectorize, tiling, and even simulate scheduling operations from other USLs (like Halide's `compute_at`).
They can be put together in reusable libraries, further enabling modularity and portability.

More information can be found in the [ASPLOS paper](https://arxiv.org/abs/2411.07211) and [Cursor.md](./Cursor.md).

## The AIR Framework: Action, Inspection, Reference

We identified that Action, Inspection, and Reference are the key scheduling language design mechanisms that enable user-defined scheduling operations.

- **[Actions](./primitives)** are scheduling operations that transform the code. This could be compiler-provided *primitive actions* (e.g., `divide_loop`, `reorder`), or *user-defined* (e.g., tile2D in the ASPLOS paper).
- **[Inspections](./inspection.md)** query properties of the code (e.g., loop bounds, memory access patterns).
- **References** point to specific parts of the code to apply actions to.

Together, AIR allows scheduling operations to be defined as composable rewrites on the code. The language implementation guarantees the correctness of these primitive rewrites with a set of effect analyses.

## Cursors: Enabling Relative References

A novel feature in Exo's design is the concept of cursors, which serve as relative references into the code. Similar to a text editing cursor, an Exo cursor can refer to a specific location in the program AST, such as a statement, loop nest, or even the gap between statements.

Cursors support navigation operations such as `next`, `prev`, `parent`, enabling powerful code transformations using relative positions.
Furthermore, Cursor _forwarding_ let users reuse the cursor from the previous procedure in the current procedure.
Multiple cursors can coexist, allowing different parts of the code to be referenced and modified simultaneously.

