#include <unordered_map>
#include <vector>

#include <algorithm>
#include <atomic>

#include "hipew.h"
#include "hsaco.h"
#include "hsaew.h"
#include "intercept_kernel.h"
#include "utpx.h"

namespace utpx {

static std::atomic_bool recordKernelMetadata;
static std::unordered_map<const void *, HSACOKernelMeta> kernelNameToMetadata;
static std::vector<HSACOKernelMeta> kernelMetadata;

extern "C" [[maybe_unused]] hsa_status_t hsa_code_object_reader_create_from_memory( //
    const void *code_object,                                                        //
    size_t size,                                                                    //
    hsa_code_object_reader_t *code_object_reader) {
  // Here we have access to our ELF code object, we extract the .note section and record the metadata.
  auto original = dlSymbol<_hsa_code_object_reader_create_from_memory>("hsa_code_object_reader_create_from_memory", HsaLibrarySO);
  auto result = original(code_object, size, code_object_reader);
  if (recordKernelMetadata && result == HSA_STATUS_SUCCESS) {
    if (auto coMeta = parseHSACodeObject(reinterpret_cast<const char *>(code_object), size); coMeta) {
      kernelMetadata.insert(kernelMetadata.end(), coMeta->begin(), coMeta->end());
      for (const auto &kernelMeta : *coMeta) {
        log("[KERNEL] Recorded: name=%s argCount=%ld, argSize=%ld, argAlignment=%ld", //
            kernelMeta.name.c_str(), kernelMeta.args.size(), kernelMeta.kernargSize, kernelMeta.kernargAlign);
      }
    }
  }
  return result;
}


extern "C" [[maybe_unused]] void __hipRegisterFunction( // NOLINT(*-reserved-identifier)
    std::vector<hipModule_t> *modules,                  //
    const void *hostFunction,                           //
    char *deviceFunction,                               //
    const char *deviceName,                             //
    unsigned int threadLimit,                           //
    unsigned *tid,                                      //
    unsigned *bid,                                      //
    dim3 *blockDim,                                     //
    dim3 *gridDim,                                      //
    int *wSize) {
  static const char *HIP_ENABLE_DEFERRED_LOADING = "HIP_ENABLE_DEFERRED_LOADING";
  log("[KERNEL] Intercepting __hipRegisterFunction(%p, %p, %s, %s, %d, %p, %p, %p, %p, %p)", //
      modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize);

  // We set HIP_ENABLE_DEFERRED_LOADING=0 here to so that all kernels will be loaded here.
  // Without this, HIP defers to the first kernel launch, which makes modifications to the kernel args very difficult.
  auto originalDeferredLoading = getenv(HIP_ENABLE_DEFERRED_LOADING);
  setenv(HIP_ENABLE_DEFERRED_LOADING, "0", /* override */ 1);
  auto original = dlSymbol<___hipRegisterFunction>("__hipRegisterFunction", HipLibrarySO);
  recordKernelMetadata = true;
  original(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize);
  // __hipRegisterFunction internally invokes a series of HSA calls to set up the code object, and what we need is the
  // HSA ELF image which is available when hsa_code_object_reader_create_from_memory is called, so we intercept that function.
  recordKernelMetadata = false;
  if (!originalDeferredLoading) unsetenv(HIP_ENABLE_DEFERRED_LOADING);
  else
    setenv(HIP_ENABLE_DEFERRED_LOADING, originalDeferredLoading, /* override */ 1);
  if (auto it = std::find_if(kernelMetadata.begin(), kernelMetadata.end(), [&](auto &meta) { return meta.name == deviceFunction; });
      it != kernelMetadata.end()) {
    kernelNameToMetadata.emplace(hostFunction, *it);
  }
}

extern "C" [[maybe_unused]] hipError_t hipModuleLoadDataEx( //
    hipModule_t *module,                                    //
    const void *image,                                      //
    unsigned int numOptions,                                //
    hipJitOption *options,                                  //
    void **optionValues) {
  auto original = dlSymbol<_hipModuleLoadDataEx>("hipModuleLoadDataEx", HipLibrarySO);
  log("[KERNEL] Intercepting hipModuleLoadDataEx(module=%p, image=%p, numOpts=%d, jitOpts=%p, options%p)", //
      module, image, numOptions, options, optionValues);

  recordKernelMetadata = true;
  auto result = original(module, image, numOptions, options, optionValues);
  recordKernelMetadata = false;
  return result;
}

static thread_local bool inhibitInterception = {};

void kernel::suspendInterception() { inhibitInterception = true; }
void kernel::resumeInterception() { inhibitInterception = false; }

extern "C" [[maybe_unused]] hipError_t hipLaunchKernel( //
    const void *f,                                      //
    dim3 grid,                                          //
    dim3 block,                                         //
    void **args,                                        //
    size_t sharedMemBytes,                              //
    hipStream_t stream) {
  auto original = dlSymbol<_hipLaunchKernel>("hipLaunchKernel", HipLibrarySO);
  if (!inhibitInterception) {
    log("[KERNEL] Intercepting hipLaunchKernel(f=%p, grid=(%d,%d,%d), block=(%d,%d,%d), args=%p, sharedMemBytes=%ld, stream=%p)", //
        (void *)f, grid.x, grid.y, grid.z, block.x, block.y, block.z, args, sharedMemBytes, stream);

    if (auto it = kernelNameToMetadata.find(f); it != kernelNameToMetadata.end()) {
      log("\t%s<<<>>>", it->second.demangledName.c_str());
      kernel::interceptKernelLaunch(f, it->second, args, grid, block);
    } else
      log("[KERNEL] WARNING: Cannot find kernel metadata for fn pointer %p, interception function not invoked", f);
  }
  auto r =  original(f, grid, block, args, sharedMemBytes, stream);
  return r;
}

extern "C" [[maybe_unused]] hipError_t hipModuleLaunchKernel( //
    hipFunction_t f,                                          //
    unsigned int gridDimX,                                    //
    unsigned int gridDimY,                                    //
    unsigned int gridDimZ,                                    //
    unsigned int blockDimX,                                   //
    unsigned int blockDimY,                                   //
    unsigned int blockDimZ,                                   //
    unsigned int sharedMemBytes,                              //
    hipStream_t stream,                                       //
    void **kernelParams,                                      //
    void **extra) {
  auto original = dlSymbol<_hipModuleLaunchKernel>("hipModuleLaunchKernel", HipLibrarySO);
  log("hipModuleLaunchKernel(%p, ..., kernelParams=%p, sharedMemBytes=%d, stream=%p)", f, kernelParams, sharedMemBytes, stream);
  if (!inhibitInterception) {
    auto name = reinterpret_cast<amdDeviceFunc *>(f)->name_;
    if (auto it = std::find_if(kernelMetadata.begin(), kernelMetadata.end(), [&](auto &m) { return m.name == name; });
        it != kernelMetadata.end()) {
      log("\t%s<<<>>>", it->demangledName.c_str());
      kernel::interceptKernelLaunch(f, *it, kernelParams, dim3{gridDimX, gridDimY, gridDimZ}, dim3{blockDimX, blockDimY, blockDimZ});
    } else
      log("[KERNEL] WARNING: Cannot find kernel metadata for fn pointer %p, interception function not invoked", f);
  }
  return original(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}
} // namespace utpx
