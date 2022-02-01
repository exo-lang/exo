#include <gtest/gtest.h>
#include <mkl.h>
#include <util.hpp>

#include "ssyrk.h"

// void SSYRK( ssyrk_Context *ctxt, int_fast32_t M, int_fast32_t K, float* A,
// float* C );

namespace {

using ::testing::TestWithParam;
using ::testing::Values;

class Test : public TestWithParam<int> {
public:
  ~Test() override {}
  void SetUp() override { m = GetParam(); }

protected:
  int m;
};

TEST_P(Test, Accuracy) {
  std::vector<float> c = util::gen_matrix<float>(m, m);
  std::vector<float> a = util::gen_matrix<float>(m, m);

  std::vector<float> c2 = c;

  cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans,  // layout
              m, m,                                     // dimensions
              1.0,                                      // alpha
              a.data(), m,                              // A (lda)
              1.0,                                      // beta
              c.data(), m                               // C (ldc)
  );

  systl_ssyrk(nullptr, m, m, a.data(), c2.data());

  EXPECT_TRUE(util::all_close(c2, c));
}

const int sizes[] = {
    64,  192, 221,  256,  320,  397,  412,  448,  512,  576,  704,  732,  832,
    911, 960, 1024, 1088, 1216, 1344, 1472, 1600, 1728, 1856, 1984, 2048,
};
INSTANTIATE_TEST_SUITE_P(SSYRK, Test, testing::ValuesIn(sizes));

}  // namespace
