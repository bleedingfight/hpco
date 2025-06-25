#include "unary_ops.cuh"
namespace hpco::unary_ops::cuda {
UNARY_OP_REGISTER(elu, 512, float);
UNARY_OP_REGISTER(silu, 512, float);
} // namespace hpco::unary_ops::cuda
