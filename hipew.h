#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

const static char *HipLibrarySO = "libamdhip64.so";

extern "C" {

typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotInitialized = 3,
  hipErrorDeinitialized = 4,
} hipError_t;

typedef enum hipMemoryType {
  hipMemoryTypeHost = 0x00,
  hipMemoryTypeDevice = 0x01,
  hipMemoryTypeArray = 0x02,
  hipMemoryTypeUnified = 0x03,
} hipMemoryType;

typedef enum hipMemoryAdvise {
  hipMemAdviseSetReadMostly = 1,
  hipMemAdviseUnsetReadMostly = 2,
  hipMemAdviseSetPreferredLocation = 3,
  hipMemAdviseUnsetPreferredLocation = 4,
  hipMemAdviseSetAccessedBy = 5,
  hipMemAdviseUnsetAccessedBy = 6,
  hipMemAdviseSetCoarseGrain = 100,
  hipMemAdviseUnsetCoarseGrain = 101
} hipMemoryAdvise;

typedef struct hipPointerAttribute_t {
  enum hipMemoryType memoryType;
  int device;
  void *devicePointer;
  void *hostPointer;
  int isManaged;
  unsigned allocationFlags; /* flags specified when memory was allocated*/
                            /* peers? */
} hipPointerAttribute_t;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

typedef struct ihipStream_t *hipStream_t;
typedef struct ihipModuleSymbol_t *hipFunction_t;
typedef struct ihipModule_t *hipModule_t;

typedef hipError_t (*_hipMalloc)(void **, size_t);
typedef hipError_t (*_hipMemset)(void *, int, size_t);
typedef hipError_t (*_hipMemcpy)(void *, const void *, size_t, hipMemcpyKind);
typedef hipError_t (*_hipFree)(void *);
typedef hipError_t (*_hipMallocManaged)(void **, size_t, unsigned int);
typedef hipError_t (*_hipDeviceSynchronize)();
typedef hipError_t (*_hipPointerGetAttributes)(hipPointerAttribute_t *, const void *);
typedef hipError_t (*_hipGetDevice)(int *device);
typedef hipError_t (*_hipMemAdvise)(const void *, size_t, hipMemoryAdvise, int);
typedef hipError_t (*_hipMemPrefetchAsync)(const void *, size_t, int, hipStream_t);

typedef void *(*___hipstdpar_realloc)(void *, std::size_t);
typedef void (*___hipstdpar_free)(void *);
typedef void (*___hipstdpar_operator_delete_aligned_sized)(void *, std::size_t, std::size_t);

typedef struct dim3 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} dim3;

typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void *userData);
typedef hipError_t (*_hipStreamAddCallback)(hipStream_t hStream, hipStreamCallback_t callback, void *userData, unsigned int flags);

typedef void (*___hipRegisterFunction)(std::vector<hipModule_t> *, const void *, char *, const char *, unsigned int, unsigned *, unsigned *,
                                       dim3 *, dim3 *, int *);

typedef enum hipJitOption {
  hipJitOptionMaxRegisters = 0,
  hipJitOptionThreadsPerBlock,
  hipJitOptionWallTime,
  hipJitOptionInfoLogBuffer,
  hipJitOptionInfoLogBufferSizeBytes,
  hipJitOptionErrorLogBuffer,
  hipJitOptionErrorLogBufferSizeBytes,
  hipJitOptionOptimizationLevel,
  hipJitOptionTargetFromContext,
  hipJitOptionTarget,
  hipJitOptionFallbackStrategy,
  hipJitOptionGenerateDebugInfo,
  hipJitOptionLogVerbose,
  hipJitOptionGenerateLineInfo,
  hipJitOptionCacheMode,
  hipJitOptionSm3xOpt,
  hipJitOptionFastCompile,
  hipJitOptionNumOptions,
} hipJitOption;

typedef hipError_t (*_hipModuleLoadDataEx)(hipModule_t *module, const void *image, unsigned int numOptions, hipJitOption *options,
                                           void **optionValues);

typedef hipError_t (*_hipMalloc)(void **, size_t);
typedef hipError_t (*_hipMallocManaged)(void **, size_t, unsigned int);
typedef hipError_t (*_hipLaunchKernel)(const void *, dim3, dim3, void **, size_t, hipStream_t);
typedef enum hipFunction_attribute {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,
  HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  HIP_FUNC_ATTRIBUTE_MAX,
} hipFunction_attribute;
typedef hipError_t (*_hipFuncGetAttribute)(int *pi, hipFunction_attribute attrib, hipFunction_t hfunc);

typedef hipError_t (*_hipModuleGetFunction)(hipFunction_t *hfunc, hipModule_t hmod, const char *name);

typedef hipError_t (*_hipModuleLaunchKernel)(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                             unsigned int sharedMemBytes, hipStream_t hStream, void **kernelParams, void **extra);
}

// XXX Uber hack to get the kernel name
struct amdMonitor {
  std::atomic_intptr_t contendersList_;
  char name_[64];
  std::atomic_intptr_t onDeck_;
  void *volatile waitersList_;
  void *volatile owner_;
  uint32_t lockCount_;
  const bool recursive_;
};

// see hip::DeviceFunc, hipFunction_t and DeviceFunc* is the same thing apparently
struct amdDeviceFunc {
  amdMonitor dflock_;
  std::string name_;
  void *kernel_;
};