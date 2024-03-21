Run the following command to compile and run.
```
exocc -o blur --stem blur blur.py
g++ -o png_process main.cpp blur/blur.c -lpng -mavx2 -fopenmp; ./png_process
```

The Halide schedule shoudl be around 17x faster. Then run this to check output equivalence:
```
diff blur.png exo_blur_halide.png
```