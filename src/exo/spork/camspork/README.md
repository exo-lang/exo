C++ implementation of an interpreter for the Exo-GPU abstract machine

The code quality is really poor: Performance-chasing caused the core logic to develop into a memory-unsafe cyclic graph manipulation algorithm.
There unfortunately don't seem to be existing graph libraries that fit the use case well.
And it's really questionable the choice of using C++ in an otherwise Python codebase.

Don't modify this unless you really have to.
If you do modify this, don't modify the internals (`syncv/`) (as opposed to the interface, `program/`), unless you really have to.

`./jit.py tmp_build` to test building this independent of the Exo codebase.
