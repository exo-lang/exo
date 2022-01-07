// Copyright 2018--2020 IBM
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ROCC_SOFTWARE_SRC_XCUSTOM_H_
#define ROCC_SOFTWARE_SRC_XCUSTOM_H_

#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

/** Assembly macro for creating "raw" Rocket Custom Coproessor (RoCC)
  * assembly language instructions that will return data in rd. These
  * are to be used only in assembly language programs (not C/C++).
  *
  * Example:
  *
  * Consider the following macro consisting of a CUSTOM_0 instruction
  * with func7 "42" that is doing some operation of "a0 = op(a1, a2)":
  *
  *     ROCC_INSTRUCTION_RAW_R_R_R(0, a0, a1, a2, 42)
  *
  * This will produce the following pseudo assembly language
  * instruction:
  *
  *     .insn r CUSTOM_0, 7, 42, a0, a1, a2
  *
  * @param x the custom instruction number: 0, 1, 2, or 3
  * @param rd the destination register, e.g., a0 or x10
  * @param rs1 the first source register, e.g., a0 or x10
  * @param rs2 the second source register, e.g., a0 or x10
  * @param func7 the value of the func7 field
  * @return a raw .insn RoCC instruction
  */
#define ROCC_INSTRUCTION_RAW_R_R_R(x, rd, rs1, rs2, func7) \
  .insn r CAT(CUSTOM_, x), 7, func7, rd, rs1, rs2

/** Assembly macro for creating "raw" Rocket Custom Coproessor (RoCC)
  * assembly language instructions that will *NOT* return data in rd.
  * These are to be used only in assembly language programs (not
  * C/C++).
  *
  * Example:
  *
  * Consider the following macro consisting of a CUSTOM_1 instruction
  * with func7 "42" that is doing some operation of "op(a1, a2)". *NO*
  * data is returned:
  *
  *     ROCC_INSTRUCTION_RAW_R_R_R(1, a1, a2, 42)
  *
  * This will produce the following pseudo assembly language
  * instruction:
  *
  *     .insn r CUSTOM_1, 3, 42, x0, a1, a2
  *
  * @param x the custom instruction number: 0, 1, 2, or 3
  * @param rs1 the first source register, e.g., a0 or x10
  * @param rs2 the second source register, e.g., a0 or x10
  * @param func7 the value of the func7 field
  * @return a raw .insn RoCC instruction
  */
#define ROCC_INSTRUCTION_RAW_0_R_R(x, rs1, rs2, func7) \
  .insn r CAT(CUSTOM_, x), 3, func7, x0, rs1, rs2

/** C/C++ inline assembly macro for creating Rocket Custom Coprocessor
  * (RoCC) instructions that return data in rd. These are to be used
  * only in C/C++ programs (not bare assembly).
  *
  * This is equivalent to ROCC_INSTRUCTION_R_R_R. See it's
  * documentation.
  */
#define ROCC_INSTRUCTION(x, rd, rs1, rs2, func7) \
  ROCC_INSTRUCTION_R_R_R(x, rd, rs1, rs2, func7)

/** C/C++ inline assembly macro for creating Rocket Custom Coprocessor
  * (RoCC) instructions that return data in C variable rd. 
  * These are to be used only in C/C++ programs (not bare assembly).
  *
  * Example:
  *
  * Consider the following macro consisting of a CUSTOM_2 instruction
  * with func7 "42" that is doing some operation of "a0 = op(a1, a2)"
  * (where a0, a1, and a2 are variables defined in C):
  *
  *     ROCC_INSTRUCTION(2, a0, a1, a2, 42)
  *
  * This will produce the following inline assembly:
  *
  *     asm volatile(
  *         ".insn r CUSTOM_2, 0x7, 42, %0, %1, %2"
  *         : "=r"(rd)
  *         : "r"(rs1), "r"(rs2));
  *
  * @param x the custom instruction number: 0, 1, 2, or 3
  * @param rd the C variable to capture as destination operand
  * @param rs1 the C variable to capture for first source register
  * @param rs2 the C variable to capture for second source register
  * @param func7 the value of the func7 field
  * @return an inline assembly RoCC instruction
  */
#define ROCC_INSTRUCTION_R_R_R(x, rd, rs1, rs2, func7)                               \
  {                                                                                  \
    asm volatile(                                                                    \
        ".insn r " STR(CAT(CUSTOM_, x)) ", " STR(0x7) ", " STR(func7) ", %0, %1, %2" \
        : "=r"(rd)                                                                   \
        : "r"(rs1), "r"(rs2));                                                       \
  }

/** C/C++ inline assembly macro for creating Rocket Custom Coprocessor
  * (RoCC) instructions that return data in C variable rd.
  * These are to be used only in C/C++ programs (not bare assembly).
  *
  * Example:
  *
  * Consider the following macro consisting of a CUSTOM_3 instruction
  * with func7 "42" that is doing some operation of "a0 = op(a1, a2)"
  * (where a0, a1, and a2 are variables defined in C):
  *
  *     ROCC_INSTRUCTION(3, a0, a1, a2, 42)
  *
  * This will produce the following inline assembly:
  *
  *     asm volatile(
  *         ".insn r CUSTOM_3, 0x7, 42, %0, %1, %2"
  *         :: "r"(rs1), "r"(rs2));
  *
  * @param x the custom instruction number: 0, 1, 2, or 3
  * @param rs1 the C variable to capture for first source register
  * @param rs2 the C variable to capture for second source register
  * @param funct7 the value of the funct7 f
  * @return an inline assembly RoCC instruction
  */
#define ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, func7)                                   \
  {                                                                                  \
    asm volatile(                                                                    \
        ".insn r " STR(CAT(CUSTOM_, x)) ", " STR(0x3) ", " STR(func7) ", x0, %0, %1" \
        :                                                                            \
        : "r"(rs1), "r"(rs2));                                                       \
  }

// [TODO] fix these to align with the above approach
// Macro to pass rs2_ as an immediate
/*
#define ROCC_INSTRUCTION_R_R_I(XCUSTOM_, rd_, rs1_, rs2_, funct_) \
  asm volatile (XCUSTOM_" %[rd], %[rs1], %[rs2], %[funct]"        \
                : [rd] "=r" (rd_)                                 \
                : [rs1] "r" (rs1_), [rs2] "i" (rs2_), [funct] "i" (funct_))

// Macro to pass rs1_ and rs2_ as immediates
#define ROCC_INSTRUCTION_R_I_I(XCUSTOM_, rd_, rs1_, rs2_, funct_) \
  asm volatile (XCUSTOM_" %[rd], %[rs1], %[rs2], %[funct]"        \
                : [rd] "=r" (rd_)                                 \
                : [rs1] "i" (rs1_), [rs2] "i" (rs2_), [funct] "i" (funct_))
*/

#endif  // ROCC_SOFTWARE_SRC_XCUSTOM_H_
