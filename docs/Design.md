# Design document for Exo

Here is a summary of the key design decisions of the Exo language in github markdown format:

# Exo: A Language for Hardware-Accelerated Kernel Libraries

Exo is a domain-specific language designed to enable productive development of high-performance kernel libraries that target specialized hardware accelerators. The key design principles of Exo are:

## Exocompilation: Externalizing Hardware Targets

One of the main ideas behind Exo is **exocompilation**, which allows users to define hardware targets externally to the compiler in user-level libraries. This has several advantages:

- Hardware vendors can support new accelerators without maintaining compiler forks
- The cost of adding support for new hardware is significantly reduced
- Proprietary details of hardware can be protected

Users can model custom memories, instructions, and configuration state in libraries to target a specific accelerator. These hardware abstractions can then be used to write hand-optimized code or as building blocks for higher-level scheduling transformations.

## Fine-Grained Primitives for Performance Control

Exo offers a set of fine-grained scheduling primitives that give users low-level control over performance-critical details. These primitives can be composed to build complex transformation schedules. Some examples of these primitives include:

- `split` and `reorder` for loop transformations
- `stage_mem` for explicit data movement between memories
- `replace` for mapping code fragments to custom instructions

Having explicit control over these low-level details enables Exo to achieve performance competitive with highly-tuned vendor libraries and hand-optimized assembly code.

## User-Defined Scheduling Operations

While the flexibility of fine-grained primitives is necessary for achieving peak performance, directly using them can be verbose and laborious. To address this, Exo allows users to define new higher-level scheduling operations by composing the core primitives.

These user-defined scheduling operations can encapsulate common optimization patterns and hardware-specific transformations, greatly improving productivity. They can be put together in reusable libraries, further enabling modularity and portability.

## The AIR Abstraction: Action, Inspection, Reference

To enable user-defined scheduling operations, Exo introduces a powerful abstraction called AIR, which stands for Action, Inspection, and Reference.

- **Actions** are the scheduling primitives that transform the code (e.g., `split`, `reorder`).
- **Inspection** queries properties of the code (e.g., loop bounds, memory access patterns).
- **References** point to specific parts of the code to apply actions to.

Together, AIR allows scheduling operations to be defined as composable rewrites on the code. The language implementation guarantees the correctness of these rewrites with a set of effect analyses.

## Cursors: Enabling Relative References

A novel feature in Exo's design is the concept of cursors, which serve as relative references into the code. Similar to a text editing cursor, an Exo cursor identifies a specific location in the program AST, such as a statement, loop nest, or even the gap between statements.

Cursors support navigation operations such as `next`, `prev`, `parent`, enabling powerful code transformations using relative positions. Multiple cursors can coexist, allowing different parts of the code to be referenced and modified simultaneously.

Using cursors, complex scheduling operations can be built using simple navigation and rewrite rules, with the cursor abstracting away the details of manual AST manipulation.

## Evaluation

The effectiveness of Exo's design is demonstrated through case studies targeting specialized accelerators like Gemmini and x86 CPUs with AVX-512 extensions. With Exo, state-of-the-art performance is achieved on key computational kernels like matrix multiplication and convolution, using an order of magnitude fewer lines of code compared to handwritten libraries.
