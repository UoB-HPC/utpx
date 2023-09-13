#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <thread>

#include "intercept_kernel.h"
#include "intercept_memory.h"
#include "utpx.h"

//#define LOG

namespace utpx {

//#ifdef LOG
//
//void vlog(const char *fmt, va_list args1) {
//  std::va_list args2;
//  va_copy(args2, args1);
//  std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, fmt, args1));
//  va_end(args1);
//  std::vsnprintf(buf.data(), buf.size(), fmt, args2);
//  va_end(args2);
//  auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
//  std::fprintf(stderr, "[UTPX][+%ld,t=%d] %s\n", epochMs.count(), gettid(), buf.data());
//}
//#else
//void vlog(const char *fmt, va_list args1) {}
//#endif

// void log(const char *fmt, ...) {
//
//   std::va_list args;
//   va_start(args, fmt);
//   vlog(fmt, args);
//   va_end(args);
// }
//
// void fatal(const char *fmt, ...) {
//   std::va_list args;
//   va_start(args, fmt);
//   vlog(fmt, args);
//   va_end(args);
//   std::abort();
// }

static _hipMalloc originalHipMalloc;               //= dlSymbol<_hipMalloc>("hipMalloc", HipLibrarySO);
static _hipMemcpy originalHipMemcpy;               //= dlSymbol<_hipMemcpy>("hipMemcpy", HipLibrarySO);
static _hipDeviceSynchronize hipDeviceSynchronize; //= dlSymbol<_hipDeviceSynchronize>("hipDeviceSynchronize", HipLibrarySO);
static _hipStreamAddCallback hipStreamAddCallback; //= dlSymbol<_hipStreamAddCallback>("hipStreamAddCallback", HipLibrarySO);

struct MirroredAllocation {
  void *devicePtr;
  size_t size;

  void create() {
    log("[MEM] Creating mirrored allocation of of %ld bytes on device", size);
    if (auto result = originalHipMalloc(&devicePtr, size); result != hipSuccess) {
      fatal("\t\tUnable to create mirrored allocation: hipMalloc(%p, %ld) failed with %d", //
            &devicePtr, size, result);
    }
    if (!devicePtr) fatal("\t\tUnable to create mirrored allocation: hipMalloc produced NULL");
  }

  void mirror(void *hostPtr) {
    if (auto result = originalHipMemcpy(devicePtr, hostPtr, size, hipMemcpyHostToDevice); result != hipSuccess) {
      fatal("\t\tUnable to copy to mirrored allocation: hipMemcpy(%p <- %p, %ld) failed with %d", //
            devicePtr, hostPtr, size, result);
    }
  }
};

static std::unordered_map<uintptr_t, MirroredAllocation> hostToMirroredAlloc;
static std::shared_mutex allocationLock{};

// see rocvirtual.cpp VirtualGPU::submitKernelInternal
//                    VirtualGPU::processMemObjects

static auto findHostAllocations(uintptr_t maybePointer) {
  return std::find_if(hostToMirroredAlloc.begin(), hostToMirroredAlloc.end(),
                      [&](auto &kv) { return maybePointer >= kv.first && maybePointer < kv.first + kv.second.size; });
}

static void *findHostAllocationsAndCreateMirrored(uintptr_t maybePointer) {
  for (auto &[hostPtr, alloc] : hostToMirroredAlloc) {
    if (maybePointer >= hostPtr && maybePointer < hostPtr + alloc.size) {
      size_t offset = maybePointer - hostPtr;
      log("\t\tLocated host ptr: %p (offset=%ld) from (0x%lx+%ld)", reinterpret_cast<void *>(maybePointer), offset, hostPtr, alloc.size);
      if (alloc.devicePtr) {
        log("\t\t-> Existing mirrored allocation exists: %p", alloc.devicePtr);
      } else {
        log("\t\t-> No mirrored allocation, creating...");
        kernel::suspendInterception(); // hipMemcpy may launch more kernels, so we suspend interception for now
        alloc.create();
        alloc.mirror(reinterpret_cast<void *>(hostPtr));
        fault::registerPage(reinterpret_cast<void *>(hostPtr), alloc.size);
        kernel::resumeInterception();
      }
      return &alloc.devicePtr;
    }
  }
  return nullptr;
}

void kernel::interceptKernelLaunch(const void *fn, const HSACOKernelMeta &meta, void **args, dim3, dim3) {
  log("\tAttempting to replace host allocations for %p, argCount=%ld, argSize=%ld", fn, meta.args.size(), meta.kernargSize);

  std::unique_lock<std::shared_mutex> write(allocationLock);
  log("\tCurrent host allocations (%zu): ", hostToMirroredAlloc.size());
  {
    int p = 0;
    size_t totalHost = 0;
    size_t totalDevice = 0;
    for (const auto &[hostPtr, alloc] : hostToMirroredAlloc) {
      log("\t\t[%3d] host=(0x%lx+%ld) => device=%p", p, hostPtr, alloc.size, alloc.devicePtr);
      p++;
      totalHost += alloc.size;
      totalDevice += alloc.devicePtr ? alloc.size : 0;
    }
    log("\tTotal host = %ld MB, device = %ld MB", totalHost / 1024 / 1024, totalDevice / 1024 / 1024);
  }

  for (size_t i = 0; i < meta.args.size(); i++) {
    const HSACOKernelMeta::Arg &arg = meta.args[i];
    if (arg.kind == HSACOKernelMeta::Arg::Kind::Hidden) continue;
    if (arg.kind == HSACOKernelMeta::Arg::Kind::Unknown) {
      fatal("\tUnknown arg! [%ld] (%ld + %ld) ptr=%p", i, arg.offset, arg.size, args[i]);
    }
    log("\tChecking argument [%ld] (%ld + %ld) ptr=%p", i, arg.offset, arg.size, args[i]);
    if (arg.size < sizeof(void *)) continue;
    if (arg.size == sizeof(void *)) {                   // same size as a pointer, check if it is one
      auto target = reinterpret_cast<void **>(args[i]); // we're looking for a void* in our arg list
      if (!target) return;
      auto deref = reinterpret_cast<uintptr_t>(*target);
      if (auto that = findHostAllocationsAndCreateMirrored(deref); that) {
        log("\t\t-> Rewritten pointer argument with mirrored: old=%p, new=%p", args[i], that);
        args[i] = that;
      }
    } else {                                      // type larger than a pointer, it may be a struct containing pointers
      auto minIncrement = meta.packed(i) ? 1 : 2; // check every byte if packed, two byte alignment otherwise for (TODO maybe 8 byte align?)
      char *argData = reinterpret_cast<char *>(args[i]);
      if (!argData) continue;
      for (size_t byteOffset = 0; byteOffset < arg.size; byteOffset += minIncrement) {
        auto maybePointer = *reinterpret_cast<uintptr_t *>(argData + byteOffset);
        if (auto that = findHostAllocationsAndCreateMirrored(maybePointer); that) {
          log("\t\t-> Rewritten pointer argument at struct offset %ld with mirrored: old=%p, new=%p", byteOffset, argData + byteOffset,
              that);
          std::memcpy(argData + byteOffset, that, sizeof(void *));
        }
      }
    }
  }

  log("\t----");
}

void fault::handleUserspaceFault(void *faultAddr, void *allocAddr, size_t allocLength) {
  std::shared_lock<std::shared_mutex> read(allocationLock);
  if (auto it = hostToMirroredAlloc.find(reinterpret_cast<uintptr_t>(allocAddr)); it != hostToMirroredAlloc.end()) {
    log("[KERNEL] \t\tfound device ptr in fault handler  host=%p, device=%p+%ld, fault is %p (offset=%lu)", //
        allocAddr, it->second.devicePtr, it->second.size, faultAddr,
        reinterpret_cast<uintptr_t>(faultAddr) - reinterpret_cast<uintptr_t>(allocAddr));
    if (auto result = originalHipMemcpy(allocAddr, it->second.devicePtr, allocLength, hipMemcpyDeviceToHost); result != hipSuccess) {
      log("[KERNEL] hipMemcpy writeback failed: %d", result);
    }
    fault::unregisterPage(allocAddr);
  } else {
    log("[KERNEL] \t\t!found device ptr in fault handler %p+%ld", allocAddr, allocLength);
  }
  //
  //  fault::accessRegisteredPages([&](const auto &registeredPages) {
  //    log("[KERNEL] \tCurrent allocations: %d", registeredPages.size());
  //    int p = 0;
  //    for (const auto &[ptr, size] : registeredPages) {
  //      log("[KERNEL] \t\t[%d] %p + %ld => %x", p, ptr, size, reinterpret_cast<uintptr_t>(ptr));
  //      p++;
  //    }
  //  });
}

extern "C" [[maybe_unused]] void __attribute__((constructor)) preload_main() {
  fault::initialiseUserspacePagefaultHandling();

  originalHipMalloc = dlSymbol<_hipMalloc>("hipMalloc", HipLibrarySO);
  originalHipMemcpy = dlSymbol<_hipMemcpy>("hipMemcpy", HipLibrarySO);
  //  hipDeviceSynchronize = dlSymbol<_hipDeviceSynchronize>("hipDeviceSynchronize", HipLibrarySO);
  //  hipStreamAddCallback = dlSymbol<_hipStreamAddCallback>("hipStreamAddCallback", HipLibrarySO);
}

extern "C" [[maybe_unused]] void __attribute__((destructor)) preload_exit() { fault::terminateUserspacePagefaultHandling(); }

extern "C" [[maybe_unused]] hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto original = dlSymbol<_hipMallocManaged>("hipMallocManaged", HipLibrarySO);
  if (size < fault::hostPageSize()) {
    log("[MEM] Allocation (%zu) less than page size (%zu), skipping", size, fault::hostPageSize());
    return original(ptr, size, flags);
  }
  //  auto r = original(ptr, size + 4096 , flags);
  *ptr = aligned_alloc(fault::hostPageSize(),
                       size + fault::hostPageSize()); // XXX burn extra page worth of memory so that we don't lock the wrong thing
  if (!*ptr) return hipErrorOutOfMemory;
  log("[MEM] Intercepting hipMallocManaged(%p, %ld, %x)", (void *)ptr, size, flags);
  log("[MEM]  -> %p ", *ptr);
  std::unique_lock<std::shared_mutex> write(allocationLock);
  hostToMirroredAlloc.emplace(reinterpret_cast<uintptr_t>(*ptr), MirroredAllocation{.devicePtr = nullptr, .size = size});
  return hipSuccess;
}

// For roc-stdpar, deallocation calls __hipstdpar_hidden_free if the pointer, as queried with hipPointerGetAttributes, is not managed.
// This is a problem because our interposed hipMallocManaged returns a non-managed pointer, so roc-stdpar attempts to do a normal free on a
// pointer from hipMalloc. We intercept all hipPointerGetAttributes calls to work around this.

// thread_local bool __hipstdpar_dealloc_active = false;

extern "C" [[maybe_unused]] hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) {
  auto original = dlSymbol<_hipMemcpy>("hipMemcpy", HipLibrarySO);
  switch (kind) {
    case hipMemcpyHostToHost: // fallthrough
    case hipMemcpyDeviceToDevice: return original(dst, src, size, kind);
    case hipMemcpyDefault:      // fallthrough
    case hipMemcpyHostToDevice: // fallthrough
    case hipMemcpyDeviceToHost: {

      auto kindName = [](hipMemcpyKind kind) {
        switch (kind) {
          case hipMemcpyHostToHost: return "MemcpyHostToHost";
          case hipMemcpyDeviceToDevice: return "MemcpyDeviceToDevice";
          case hipMemcpyDefault: return "MemcpyDefault";
          case hipMemcpyHostToDevice: return "MemcpyHostToDevice";
          case hipMemcpyDeviceToHost: return "MemcpyDeviceToHost";
          default: return "Unknown";
        }
      };

      auto srcIt = hostToMirroredAlloc.find(reinterpret_cast<uintptr_t>(src));
      auto dstIt = hostToMirroredAlloc.find(reinterpret_cast<uintptr_t>(dst));

      if (srcIt != hostToMirroredAlloc.end() && dstIt != hostToMirroredAlloc.end()) {
        log("Intercepting hipMemcpy(%p, %p, %zu, %s) , dst=[host=%p;device=%p], src=[host=%p;device=%p]", //
            dst, src, size, kindName(kind),                                                               //
            reinterpret_cast<void *>(dstIt->first), reinterpret_cast<void *>(dstIt->second.devicePtr),
            reinterpret_cast<void *>(srcIt->first), reinterpret_cast<void *>(srcIt->second.devicePtr));
        auto result = original(dstIt->second.devicePtr, srcIt->second.devicePtr, size, kind);
        fault::registerPage(reinterpret_cast<void *>(dstIt->first), dstIt->second.size);
        return result;
      } else if (srcIt != hostToMirroredAlloc.end()) {                                   // the source ptr is mirrored, and dest is not:
        log("Intercepting hipMemcpy(%p, %p, %zu, %s) , dst=%p, src=[host=%p;device=%p]", //
            dst, src, size, kindName(kind), dst, reinterpret_cast<void *>(srcIt->first), reinterpret_cast<void *>(srcIt->second.devicePtr));
        // just copy to the dest (host/device) ptr, we use the device pointer as the source as it's always up-to-date
        return original(dst, srcIt->second.devicePtr, size, kind);
      } else if (dstIt != hostToMirroredAlloc.end()) {                                   // dest ptr is mirrored, and the source is not:
        log("Intercepting hipMemcpy(%p, %p, %zu, %s) , dst=[host=%p;device=%p], src=%p", //
            dst, src, size, kindName(kind), reinterpret_cast<void *>(dstIt->first), reinterpret_cast<void *>(dstIt->second.devicePtr), src);
        // just copy to the device ptr and register the host page if not already registered, synchronisation happens on next page fault
        if (!dstIt->second.devicePtr) dstIt->second.create();
        auto result = original(dstIt->second.devicePtr, src, size, kind);
        fault::registerPage(reinterpret_cast<void *>(dstIt->first), dstIt->second.size);
        return result;
      } else {
        return original(dst, src, size, kind);
      }
    }
    default: return original(dst, src, size, kind);
  }
}

extern "C" [[maybe_unused]] hipError_t hipMemset(void *ptr, int value, size_t size) {
  auto original = dlSymbol<_hipMemset>("hipMemset", HipLibrarySO);
  // XXX ptr may be an offset from base we need to do a ranged search
  std::shared_lock<std::shared_mutex> read(allocationLock);
  if (auto it = findHostAllocations(reinterpret_cast<uintptr_t>(ptr)); it != hostToMirroredAlloc.end()) {
    log("Intercepting hipMemset(%p, %d, %ld), existing host allocation found", ptr, value, size);
    size_t offsetFromBase = reinterpret_cast<uintptr_t>(ptr) - it->first;
    if (offsetFromBase != 0) fatal("IMPL: hipMemset with offset\n");
    std::memset(ptr, value, size);                  // memset the host using the already offset ptr from the arg
    if (!it->second.devicePtr) it->second.create(); // XXX devicePtr is nullptr if memset is called before any dependent kernel
    if (auto result = original(it->second.devicePtr, value, size); result != hipSuccess) {
      fatal("hipMemset(%p, %d, %ld) failed to memset mirrored allocation: %d", it->second.devicePtr, value, size, result);
    }
    return hipSuccess;
  } else {
    return original(ptr, value, size);
  }
}

extern "C" [[maybe_unused]] hipError_t hipFree(void *ptr) {
  auto original = dlSymbol<_hipFree>("hipFree", HipLibrarySO);
  if (!ptr)
    return original(nullptr); // XXX still delegate to HIP because hipFree(nullptr) can be used as an implicit hipDeviceSynchronize or
                              // initialisation of the HIP runtime
  std::unique_lock<std::shared_mutex> write(allocationLock);
  if (auto it = hostToMirroredAlloc.find(reinterpret_cast<uintptr_t>(ptr)); it != hostToMirroredAlloc.end()) {
    log("Intercepting hipFree(%p), existing host allocation found", ptr);

    if (auto page = fault::lookupRegisteredPage(ptr); page) {
      fault::unregisterPage(page->first);
    }
    free(ptr);
    if (auto result = original(it->second.devicePtr); result != hipSuccess) {
      fatal("hipFree(%p) failed to release mirrored allocation: %d", it->second.devicePtr, result);
    }
    hostToMirroredAlloc.erase(it);
    return hipSuccess;
  } else {
    return original(ptr);
  }
}

extern "C" [[maybe_unused]] hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr) {
  log("Replace hipPointerGetAttributes(%p, %p), isManaged=%d", attributes, ptr, attributes->isManaged);
  {
    std::shared_lock<std::shared_mutex> read(allocationLock);
    if (auto it = findHostAllocations(reinterpret_cast<uintptr_t>(ptr)); it != hostToMirroredAlloc.end()) {
      log(" -> Replace hipPointerGetAttributes(%p, %p), isManaged=%d", attributes, ptr, attributes->isManaged);
      attributes->isManaged = true;
      return hipSuccess;
    }
  }

  auto result = dlSymbol<_hipPointerGetAttributes>("hipPointerGetAttributes", HipLibrarySO)(attributes, ptr);

  if (result == hipSuccess) {
    attributes->isManaged = true;
    log("Intercepting hipPointerGetAttributes(%p, %p), isManaged=%d", attributes, ptr, attributes->isManaged);
  }
  return result;
}
// The following are marked inline in hipstdpar_lib.hpp, so we just always shim hipPointerGetAttributes, seems to work OK for now.
// extern "C" void *__hipstdpar_realloc(void *p, std::size_t n)
// {
//     __hipstdpar_dealloc_active = true;
//     std::fprintf(stderr, "[malloc_switch] Intercepted __hipstdpar_realloc(%p, %ld)", p, n);
//     auto result = dlSymbol<___hipstdpar_realloc>("hipstdpar_realloc")(p, n);
//     __hipstdpar_dealloc_active = false;
//     return result;
// }

// extern "C" void __hipstdpar_free(void *p)
// {
//     __hipstdpar_dealloc_active = true;
//     std::fprintf(stderr, "[malloc_switch] Intercepted __hipstdpar_free(%p)", p);
//     dlSymbol<___hipstdpar_free>("hipstdpar_free")(p);
//     __hipstdpar_dealloc_active = false;
// }

// extern "C" void __hipstdpar_operator_delete_aligned_sized(void *p, std::size_t n, std::size_t a) noexcept
// {
//     __hipstdpar_dealloc_active = true;
//     std::fprintf(stderr, "[malloc_switch] Intercepted __hipstdpar_operator_delete_aligned_sized(%p, %ld, %ld)", p, n, a);
//     dlSymbol<___hipstdpar_operator_delete_aligned_sized>("__hipstdpar_operator_delete_aligned_sized")(p, n, a);
//     __hipstdpar_dealloc_active = false;
// }
// namespace utpx

// extern "C" hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr) {
//
//
// }

} // namespace utpx
