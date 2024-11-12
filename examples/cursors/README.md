# Cursor Step-by-Step Tutorial

This example demonstrates Cursors using the tile2D example (as shown in our [ASPLOS '25 paper](https://arxiv.org/abs/2411.07211)).

## Overview

This example covers the key concepts presented in the paper:
- Finding Cursors with pattern-matching
- Cursor navigation
- Applying scheduling primitives using cursors
- Cursor forwarding after code transformations
- Defining a new scheduling operation

## Getting Started

To run this example:
```bash
exocc cursors.py
```
Running `exocc` on `cursors.py` will generate the C code in the `cursors/cursors.c` file.
It will also print out the intermediate steps of the example.

