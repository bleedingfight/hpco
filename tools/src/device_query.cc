#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <ratio>

int getCudaCores(int major, int minor) {
    // 根据 compute capability 获取每个 SM 的 CUDA cores 数量
    typedef struct {
        int SM;
        int Cores;
    } SMtoCores;

    SMtoCores gpuArchCoresPerSM[] = {{0x30, 192}, // Kepler
                                     {0x32, 192}, // Kepler
                                     {0x35, 192}, // Kepler
                                     {0x37, 192}, // Kepler
                                     {0x50, 128}, // Maxwell
                                     {0x52, 128}, // Maxwell
                                     {0x53, 128}, // Maxwell
                                     {0x60, 64},  // Pascal
                                     {0x61, 128}, // Pascal
                                     {0x62, 128}, // Pascal
                                     {0x70, 64},  // Volta
                                     {0x72, 64},  // Volta
                                     {0x75, 64},  // Turing
                                     {0x80, 64},  // Ampere
                                     {0x86, 128}, // Ampere
                                     {0x90, 128}, // Hopper
                                     {-1, -1}};

    int sm = ((major << 4) + minor);
    for (int i = 0; gpuArchCoresPerSM[i].SM != -1; i++) {
        if (gpuArchCoresPerSM[i].SM == sm) {
            return gpuArchCoresPerSM[i].Cores;
        }
    }

    std::cerr << "Unknown SM version " << major << "." << minor << std::endl;
    return 0;
}
std::string centerMesg(const std::string mesg, const int width,
                       char type = '-') {
    int len = mesg.size();
    assert(len < width);
    int left = (width - len) / 2;
    std::string r = "";
    for (int i = 0; i < left; i++) {
        r.push_back(type);
    }
    return r + mesg + r;
}
void deviceQuery(int dev) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int coresPerSM = getCudaCores(deviceProp.major, deviceProp.minor);
    int totalCores = coresPerSM * deviceProp.multiProcessorCount;
    const int width = 110;

    std::cout << centerMesg("Device: " + std::to_string(dev) + deviceProp.name,
                            width + 10)
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "CUDA Capability: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << std::left << std::setw(width)
              << "Multiprocessors: " << deviceProp.multiProcessorCount
              << std::endl;
    std::cout << std::left << std::setw(width) << "CUDA Cores: " << totalCores
              << std::endl;
    std::cout << std::left << std::setw(width) << "Total Global Memory: "
              << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB ("
              << (deviceProp.totalGlobalMem / (1024 * 1024 * 1024)) << "GB)"
              << std::endl;
    std::cout << std::left << std::setw(width) << "Const Memory"
              << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << std::left << std::setw(width)
              << "Shared Memory Of Thread Block" << deviceProp.sharedMemPerBlock
              << " bytes (" << deviceProp.sharedMemPerBlock / 1024 << " KB)"
              << std::endl;

    std::cout << std::left << std::setw(width)
              << "Registers Per Block(register is 32bit) "
              << deviceProp.regsPerBlock << " bytes ("
              << deviceProp.regsPerBlock / 1024 << " KB)" << std::endl;

    std::cout << std::left << std::setw(width)
              << "Maximum resident threads per multiprocessor:"
              << deviceProp.maxThreadsPerMultiProcessor << std::endl;

    // IO
    std::cout << std::left << std::setw(width)
              << "The maximum value of cudaAccessPolicyWindow:"
              << deviceProp.accessPolicyMaxWindowSize / 1024.f / 1024.f
              << "(MB)" << std::endl;
    float bandwidth = 2.0f * deviceProp.memoryClockRate *
                      (deviceProp.memoryBusWidth / 8) / 1.0e6f;
    std::cout << std::left << std::setw(width)
              << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << std::left << std::setw(width)
              << "Shared memory available per multiprocessor:"
              << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB"
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "32-bit registers available per multiprocessor:"
              << deviceProp.regsPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << std::left << std::setw(width)
              << "GPU Clock Rate: " << deviceProp.clockRate / 1000 << " MHz"
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Memory Clock Rate: " << deviceProp.memoryClockRate / 1000
              << " MHz" << std::endl;
    std::cout << std::left << std::setw(width)
              << "Memory Bus Width: " << deviceProp.memoryBusWidth << " bits"
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB"
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Maximum number of resident blocks per multiprocessor:"
              << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device's maximum l2 persisting lines capacity setting: "
              << deviceProp.persistingL2CacheMaxSize / 1024.f << "(KB)"
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Shared memory reserved by CUDA driver per block:"
              << deviceProp.reservedSharedMemPerBlock / 1024.f << "(KB)"
              << std::endl;
    std::cout
        << std::left << std::setw(width)
        << "Per device maximum shared memory per usable by special opt in:"
        << deviceProp.sharedMemPerBlockOptin / 1024.f << "(KB)" << std::endl;

    std::cout << std::left << std::setw(width) << "Max Threads Dim: "
              << "[" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << "]" << std::endl;
    std::cout << std::left << std::setw(width) << "Max Grid Size: "
              << "[" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2]
              << "]" << std::endl;

    std::cout << centerMesg("Hardware Feature", width + 10) << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device can possibly execute multiple kernels concurrently: "
              << deviceProp.concurrentKernels << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports unified addressing: "
              << deviceProp.unifiedAddressing << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device can coherently access managed memory concurrently "
                 "with the CPU: "
              << deviceProp.concurrentManagedAccess << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports managedMemory: " << deviceProp.managedMemory
              << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports Compute Preemption: "
              << deviceProp.computePreemptionSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Cooperative Groups: "
              << deviceProp.cooperativeLaunch << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Cooperative MultiDevice Launch: "
              << deviceProp.cooperativeMultiDeviceLaunch << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports stream priorities:"
              << deviceProp.streamPrioritiesSupported << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports caching globals in L1:"
              << deviceProp.globalL1CacheSupported << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports localL1CacheSupported:"
              << deviceProp.localL1CacheSupported << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports "
                 "GPUDirect RDMA: "
              << deviceProp.gpuDirectRDMASupported << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports "
                 "GPUDirect RDMA Flush Writes Options: "
              << deviceProp.gpuDirectRDMAFlushWritesOptions << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports RDMAWritesOrdering: "
              << deviceProp.gpuDirectRDMAWritesOrdering << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports "
                 "IPC Events: "
              << deviceProp.ipcEventSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports ECC: " << deviceProp.ECCEnabled << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Cluster Launch:" << deviceProp.clusterLaunch
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Unified Pointers:"
              << deviceProp.unifiedFunctionPointers << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device Supports Mapping CUDA arrays and CUDA mipmapped arrays"
              << deviceProp.deferredMappingCudaArraySupported << std::endl;

    std::cout << std::left << std::setw(width) << "PCIE " << deviceProp.pciBusID
              << "." << deviceProp.pciDeviceID << "." << deviceProp.pciDomainID
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device AsyncEngine Count: " << deviceProp.asyncEngineCount
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports sparse Cuda Array:"
              << deviceProp.sparseCudaArraySupported << std::endl;

    std::cout
        << std::left << std::setw(width)
        << "External timeline semaphore interop is Supported on the device "
        << deviceProp.timelineSemaphoreInteropSupported << "\n";
    std::cout
        << std::left << std::setw(width)
        << "Device Supports host memory registration via cudaHostRegister:"
        << deviceProp.hostRegisterSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports using the ::cudaHostRegister register "
                 "memory that must be mapped as read-only to the GPU:"
              << deviceProp.hostRegisterReadOnlySupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Memory Pool Supported Handle Types:"
              << deviceProp.memoryPoolsSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device can map host memory with "
                 "cudaHostAlloc/cudaHostGetDevicePointer: "
              << deviceProp.canMapHostMemory << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Memory Pool: "
              << deviceProp.memoryPoolsSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device can access host registered memory at the same "
                 "virtual address as the host: "
              << deviceProp.canUseHostPointerForRegisteredMem << std::endl;
    // unsupport
    std::cout << std::left << std::setw(width)
              << "Unique identifier for a group of devices on the same board: "
              << deviceProp.multiGpuBoardGroupID << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device is on a multi-GPU board: "
              << deviceProp.isMultiGpuBoard << std::endl;

    std::cout << std::left << std::setw(width)
              << "Device accesses pageable memory via the host's page tables"
              << deviceProp.pageableMemoryAccessUsesHostPageTables << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Direct Access from Host: "
              << deviceProp.directManagedMemAccessFromHost << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports TTC driver:" << deviceProp.tccDriver
              << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Memory Access from Host: "
              << deviceProp.hostNativeAtomicSupported << std::endl;
    std::cout << std::left << std::setw(width)
              << "Device Supports Page-locked Memory: "
              << deviceProp.pageableMemoryAccess << std::endl;

    std::cout << centerMesg("", width + 10) << std::endl;
}

int main(int argc, char **argv) {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "获取设备数量失败: " << cudaGetErrorString(error)
                  << std::endl;
        return -1;
    }
    if (deviceCount == 0) {
        std::cout << "未检测到 CUDA 设备。" << std::endl;
        return 0;
    }

    std::cout << "共检测到 " << deviceCount << " 个 CUDA 设备。" << std::endl;
    if (argc >= 2) {
        int deviceId = std::stoi(argv[1]);
        assert(deviceId < deviceCount);
        std::cout << "查询设备号：" << deviceId << "\n";
        deviceQuery(deviceId);
    } else {
        for (int i = 0; i < deviceCount; ++i) {
            deviceQuery(i);
        }
    }

    return 0;
}
