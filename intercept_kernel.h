#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hipew.h"
#include "hsaco.h"
#include "hsaew.h"
#include "utpx.h"

namespace utpx::kernel {

struct KernelMetadata {
  std::string name;
  std::string demangledName;
  size_t argBytes;
  size_t argAlignment;
  std::vector<size_t> pointerOffsets{};
};

void suspendInterception();
void resumeInterception();

void interceptKernelLaunch(const void *fn, const HSACOKernelMeta &meta, void **args, dim3 grid, dim3 block, hipStream_t stream);

} // namespace utpx::kernel
