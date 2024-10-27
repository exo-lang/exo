# Cursor Step-by-Step Tutorial

This tutorial demonstrates a simple application of Cursors using the tile2D example (as shown in our ASPLOS '25 paper).

## Overview

Learn how to use Cursors to navigate and transform Exo object code. Cursors allow you to:
- Select and reference specific code elements (expressions, statements, blocks)
- Navigate spatially within procedures
- Apply optimization

## Key Concepts

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

