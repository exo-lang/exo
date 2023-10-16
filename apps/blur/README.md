Run the following command to compile and run.
```
exocc -o blur --stem blur blur.py
g++ -o png_process main.cpp blur/blur.c -lpng ; ./png_process
```
