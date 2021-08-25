**Step 0: Acquire recent CMake**

Pick your favorite method. On Ubuntu/derivatives, use the [APT repo](https://apt.kitware.com).
If you have access to the snap store, you can try `snap install cmake`.

Pip is also convenient (but doesn't work in venv):

```
$ python3 -m pip install --upgrade pip wheel
$ python3 -m pip install --user cmake
```

If all else fails, download the statically linked binaries (but do try to get
it installed globally):

```
$ wget -qO- https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.tar.gz | tar xz
$ export PATH="$PWD/cmake-3.21.1-linux-x86_64/bin:$PATH"
```

On macOS, `brew install cmake` is sufficient.

**Step 1: Install BLIS**

```
$ git clone git@github.com:flame/blis.git
$ cd blis
blis$ ./configure --prefix=$PWD/install haswell
blis$ make -j $(nproc)
blis$ make install
blis$ cd ..
```

**Step 2: Run `test_avx2_sgemm_6x16`**

```
$ rm -rf ../tmp
$ python3 -m pytest -k test_avx2_sgemm_6x16 ..
```

**Step 3: Build and run benchmarks**

```
$ cmake --preset=default
$ cmake --build build -j $(nproc)
$ ./build/run
```
