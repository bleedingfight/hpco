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
                                     {0x87, 128}, // Hopper
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

int main(int argc, char **argv) {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA Capable device(s)\n"
              << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        int coresPerSM = getCudaCores(deviceProp.major, deviceProp.minor);
        int totalCores = coresPerSM * deviceProp.multiProcessorCount;
        const int width = 80;

        std::cout << std::left << "------------------- Device:" << dev << "["
                  << deviceProp.name << "]--------------------------"
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "CUDA Capability: " << deviceProp.major << "."
                  << deviceProp.minor << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Multiprocessors: " << deviceProp.multiProcessorCount
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "CUDA Cores: " << totalCores << std::endl;
        std::cout << std::left << std::setw(width) << "Total Global Memory: "
                  << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB ("
                  << (deviceProp.totalGlobalMem / (1024 * 1024 * 1024)) << "GB)"
                  << std::endl;
        std::cout << std::left << std::setw(width) << "Const Memory"
                  << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Shared Memory Of Thread Block"
                  << deviceProp.sharedMemPerBlock << " bytes ("
                  << deviceProp.sharedMemPerBlock / 1024 << " KB)" << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Registers Per Block(register is 32bit) "
                  << deviceProp.regsPerBlock << " bytes ("
                  << deviceProp.regsPerBlock / 1024 << " KB)" << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Maximum resident threads per multiprocessor:"
                  << deviceProp.maxThreadsPerMultiProcessor << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Shared memory available per multiprocessor:"
                  << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB"
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << " 32-bit registers available per multiprocessor:"
                  << deviceProp.regsPerMultiprocessor / 1024 << " KB"
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "GPU Clock Rate: " << deviceProp.clockRate / 1000 << " MHz"
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Memory Clock Rate: " << deviceProp.memoryClockRate / 1000
                  << " MHz" << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Memory Bus Width: " << deviceProp.memoryBusWidth
                  << " bits" << std::endl;
        std::cout << std::left << std::setw(width)
                  << "L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB"
                  << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device's maximum l2 persisting lines capacity setting: "
                  << deviceProp.persistingL2CacheMaxSize / 1024.f << "(KB)"
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock
                  << std::endl;
        std::cout << std::left << std::setw(width) << "Max Threads Dim: "
                  << "[" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << std::left << std::setw(width) << "Max Grid Size: "
                  << "[" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << "]" << std::endl;

        std::cout << "-----------------------------------Hardware feature"
                     "--------------------------\n";
        std::cout
            << std::left << std::setw(width)
            << "Device can possibly execute multiple kernels concurrently: "
            << deviceProp.concurrentKernels << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports unified addressing: "
                  << deviceProp.unifiedAddressing << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device can coherently access managed memory concurrently "
                     "with the CPU: "
                  << deviceProp.concurrentManagedAccess << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports managedMemory: "
                  << deviceProp.managedMemory << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device is on a multi-GPU board: "
                  << deviceProp.isMultiGpuBoard << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device can access host registered memory at the same "
                     "virtual address as the host: "
                  << deviceProp.canUseHostPointerForRegisteredMem << std::endl;

        std::cout
            << std::left << std::setw(width)
            << "Unique identifier for a group of devices on the same board: "
            << deviceProp.multiGpuBoardGroupID << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Compute Preemption: "
                  << deviceProp.computePreemptionSupported << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Cooperative Groups: "
                  << deviceProp.cooperativeLaunch << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Cooperative MultiDevice Launch: "
                  << deviceProp.cooperativeMultiDeviceLaunch << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Memory Pool: "
                  << deviceProp.memoryPoolsSupported << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Page-locked Memory: "
                  << deviceProp.pageableMemoryAccess << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device supports stream priorities:"
                  << deviceProp.streamPrioritiesSupported << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device supports caching globals in L1:"
                  << deviceProp.globalL1CacheSupported << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device supports localL1CacheSupported:"
                  << deviceProp.localL1CacheSupported << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Memory Access from Host: "
                  << deviceProp.hostNativeAtomicSupported << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device can map host memory with "
                     "cudaHostAlloc/cudaHostGetDevicePointer: "
                  << deviceProp.canMapHostMemory << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Direct Access from Host: "
                  << deviceProp.directManagedMemAccessFromHost << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports "
                     "GPUDirect RDMA: "
                  << deviceProp.gpuDirectRDMASupported << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device supports "
                     "GPUDirect RDMA Flush Writes Options: "
                  << deviceProp.gpuDirectRDMAFlushWritesOptions << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device supports RDMAWritesOrdering: "
                  << deviceProp.gpuDirectRDMAWritesOrdering << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports "
                     "IPC Events: "
                  << deviceProp.ipcEventSupported << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device Supports ECC: " << deviceProp.ECCEnabled
                  << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Cluster Launch:"
                  << deviceProp.clusterLaunch << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports Unified Pointers:"
                  << deviceProp.unifiedFunctionPointers << std::endl;
        std::cout << std::left << std::setw(width)
                  << "Device supports TTC driver:" << deviceProp.tccDriver
                  << std::endl;
        std::cout
            << std::left << std::setw(width)
            << "Device supports Mapping CUDA arrays and CUDA mipmapped arrays"
            << deviceProp.deferredMappingCudaArraySupported << std::endl;

        std::cout << std::left << std::setw(width)
                  << "Device AsyncEngine Count: " << deviceProp.asyncEngineCount
                  << std::endl;

        float bandwidth = 2.0f * deviceProp.memoryClockRate *
                          (deviceProp.memoryBusWidth / 8) / 1.0e6f;
        std::cout << std::left << std::setw(width)
                  << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << "---------------------------------------------------------"
                     "---------------------\n";
    }
    // int pciBusID;                         /**< PCI bus ID of the device */
    // int pciDeviceID;                      /**< PCI device ID of the device */
    // int pciDomainID;                      /**< PCI domain ID of the device */
    // int singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single
    // precision
    //                                          performance (in floating-point
    //                                          operations per second) to double
    //                                          precision performance */
    // size_t sharedMemPerBlockOptin; /**< Per device maximum shared memory per
    // block
    //                                   usable by special opt in */
    // int pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable
    // memory
    //                                                via the host's page tables
    //                                                */
    // int maxBlocksPerMultiProcessor;     /**< Maximum number of resident
    // blocks per
    //                                        multiprocessor */
    // int accessPolicyMaxWindowSize;      /**< The maximum value of
    //                                        ::cudaAccessPolicyWindow::num_bytes.
    //                                        */
    // size_t reservedSharedMemPerBlock; /**< Shared memory reserved by CUDA
    // driver per
    //                                      block in bytes */
    // int hostRegisterSupported;    /**< Device supports host memory
    // registration via
    //                                  ::cudaHostRegister. */
    // int sparseCudaArraySupported; /**< 1 if the device supports sparse CUDA
    // arrays
    //                                  and sparse CUDA mipmapped arrays, 0
    //                                  otherwise
    //                                */
    // int hostRegisterReadOnlySupported;     /**< Device supports using the
    //                                           ::cudaHostRegister flag
    //                                           cudaHostRegisterReadOnly to
    //                                           register memory that must be
    //                                           mapped as read-only to the GPU
    //                                           */
    // int timelineSemaphoreInteropSupported; /**< External timeline semaphore
    // interop
    //                                           is supported on the device */
    // unsigned int
    //     memoryPoolSupportedHandleTypes; /**< Bitmask of handle types
    //     supported with
    //                                        mempool-based IPC */
    return 0;
}
