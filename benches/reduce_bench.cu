#include "csrc/cuda/reduce/reduce_kernel.cuh"
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

static void reduce_max_benchmark(nvbench::state &state) {
    const auto blockSize = state.get_int64("BlockSize");
    const auto elements = state.get_int64("elements");
    state.add_element_count(elements, "Elements");
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    state.add_element_count(elements);
    state.add_global_memory_reads<float>(elements);
    state.add_global_memory_writes<float>(elements);
    auto grid = (elements + blockSize - 1) / blockSize;
    thrust::device_vector<float> d_in(elements);
    thrust::device_vector<float> d_out(1);
    state.exec([&blockSize, &grid, &d_out, &d_in,
                &elements](nvbench::launch &launch) {
        reduce_max_kernel_opt<float, 32>
            <<<grid, blockSize, blockSize * sizeof(float),
               launch.get_stream()>>>(thrust::raw_pointer_cast(d_out.data()),
                                      thrust::raw_pointer_cast(d_in.data()),
                                      elements);
    });
}
NVBENCH_BENCH(reduce_max_benchmark)
    .add_int64_axis("BlockSize", {256, 512})
    .add_int64_power_of_two_axis("elements", nvbench::range(12, 20, 2));

// NVBench will provide the main() function
NVBENCH_MAIN
