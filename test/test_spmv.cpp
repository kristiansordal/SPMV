#include "../src/spmv.hpp"
#include <array>
#include <csr.hpp>
#include <gtest/gtest.h>
#include <mtx.hpp>
#include <string>

const std::array<std::string, 2> test_mtx = {"karate.mtx", "mycielskian4.mtx"};

TEST(L2Norm, Karate) {
    MTX<int, double> mtx;
    CSR<int, double> g;
    mtx.read_mtx(test_mtx[0]);
    g = mtx.mtx_to_csr();
    auto l2 = l2_norm(g);
    EXPECT_EQ(6.725698e+00, l2);
}
