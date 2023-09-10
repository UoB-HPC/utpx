#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace utpx {



// https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-metadata
struct HSACOKernelMeta {


  struct Arg {
    enum class Kind : uint8_t {
      ByValue, GlobalBuffer, Hidden, Unknown
    };

    size_t offset, size;
    Kind kind;
  };

  std::string name;
  std::string demangledName;
  size_t kernargSize, kernargAlign;
  std::vector<Arg> args;
  [[nodiscard]] bool packed( size_t index) const;
};
using HSACOMeta = std::vector<HSACOKernelMeta>  ;

std::optional<HSACOMeta> parseHSACodeObject(const char *data, size_t length);
std::string  demangleCXXName(const char *abiName) ;

} // namespace utpx