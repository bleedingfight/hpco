#include "cutlass_common.h"
#include <cuda.h>
#include <cute/tensor.hpp>
using namespace cute;

int test1() {
    auto shape1 = make_shape(Int<8>{});
    std::cout << "Shape = " << shape1 << "\n";
    return 0;
}
