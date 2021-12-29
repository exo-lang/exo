AC_INIT([rocket-rocc-examples], [0.1], [schuyler.eldridge@gmail.com])

cross_compiling=yes
AC_PROG_CC

AC_ARG_WITH(xlen,
        [AS_HELP_STRING([--with-xlen=XLEN],
                [Set XLEN, the X-register bit width (default is 64)])],
        AC_SUBST(XLEN, $withval),
        AC_SUBST(XLEN, 64)
        )

AC_OUTPUT(
  Makefile
)
