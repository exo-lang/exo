#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include <sgemm.h>

float num_gflops(int k) { return 1e-9 * 2 * 6 * 64 * k; }

int main (int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <k>\n";
        return 1;
    }

    int k = std::atoi(argv[1]);
    if (k < 1) {
        std::cerr << "k must be positive!\n";
        return 1;
    }

    std::vector<float> a(6 * k);
    std::vector<float> b(k * 64);
    std::vector<float> c(6 * 64);

    int num_trials = 10000;

    const auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_trials; i++) {
        sgemm_kernel_avx512_6x4(nullptr, k, a.data(), b.data(),
                                { c.data(), {64, 1}});
    }

    const auto end = std::chrono::steady_clock::now();
    double duration =
        std::chrono::duration<double>(end - start).count() / num_trials;

    double gflops = num_gflops(k) / duration;

    printf("Ran 6x64 rank-k (k=%d) reduce in %lf s (%lf GFLOPs)\n", k,
           duration, gflops);
}
