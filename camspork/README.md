# Camspork

C++ Abstract Machine ("Spork") interpreter for Exo-GPU

Because this is written in C++, we build and install this separately from the pure Python `exo` module.
This is only required for `sync_check` functionality, so this is an optional manual dependency.


# Build

This is assuming you are using the same venv as where you installed `exo`.

`python3 setup.py build && pip install .  # Or pip install -e`

You need the `c++` command to refer to an up-to-date GCC or clang.
Edit the `ninja` file if you need to further develop the library.


# Directories

`src/camspork`: Python wrapper module. The build C++ library is also placed here.

`libcamspork`: C++ sources.


