#include <cxxabi.h>
#include <elfio/elfio.hpp>

#include "hsaco.h"
#include "json.hpp"
#include "utpx.h"

// llvm/include/llvm/Support/MathExtras.h
template <uint64_t Align> constexpr inline uint64_t alignTo(uint64_t Value) {
  static_assert(Align != 0u, "Align must be non-zero");
  return (Value + Align - 1) / Align * Align;
}

using namespace ELFIO;

// llvm/include/llvm/Object/ELFTypes.h
struct Elf_Nhdr {
  Elf_Word n_namesz;
  Elf_Word n_descsz;
  Elf_Word n_type;
  static const unsigned int Align = 4;
  /// Get the size of the note, including name, descriptor, and padding.
  [[nodiscard]] size_t size() const { return sizeof(*this) + alignTo<Align>(n_namesz) + alignTo<Align>(n_descsz); }
  std::string name(const char *data) const {
    auto base = data + sizeof(*this);
    return {base, base + n_namesz - 1};
  }
  std::pair<const char *, const char *> desc(const char *data) const {
    auto base = data + sizeof(*this) + alignTo<Align>(n_namesz);
    return {base, base + n_descsz};
  }
};

// See llvm/include/llvm/BinaryFormat/ELF.h
enum {                    // AMDGPU vendor specific notes. (Code Object V3)
  NT_AMDGPU_METADATA = 32 // Note types with values between 0 and 31 (inclusive) are reserved.
};

// see https://stackoverflow.com/a/13059195/896997 and https://stackoverflow.com/a/57862858/896997
struct membuf : std::streambuf {
  membuf(char const *base, size_t size) {
    char *p(const_cast<char *>(base));
    this->setg(p, p, p + size);
  }
  pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which  ) override {
    if (dir == std::ios_base::cur) gbump(int(off));
    else if (dir == std::ios_base::end)
      setg(eback(), egptr() + off, egptr());
    else if (dir == std::ios_base::beg)
      setg(eback(), eback() + off, egptr());
    return gptr() - eback();
  }

  pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
    return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
  }
};
struct imemstream : virtual membuf, std::istream {
  imemstream(char const *base, size_t size) : membuf(base, size), std::istream(static_cast<std::streambuf *>(this)) {}
};

bool utpx::HSACOKernelMeta::packed(size_t index) const {
  if (index > args.size() - 1) return false; // OOB
  else if (index == args.size() - 1)         // last one
    return args[index].offset + args[index].size < kernargSize;
  else
    return args[index].offset + args[index].size < args[index + 1].offset;
}

std::optional<utpx::HSACOMeta> utpx::parseHSACodeObject(const char *data, size_t length) {
  elfio reader;
  // ELFIO is streaming, so we make an istream out of some constant data
  imemstream stream(data, length);
  if (!reader.load(stream, true)) {
    log("[HSACO] Failed to read ELF file at %p+%ld", data, length);
    return {};
  }

  const auto kindName = [](HSACOKernelMeta::Arg::Kind kind) {
    switch (kind) {
      case HSACOKernelMeta::Arg::Kind::ByValue: return "ByValue";
      case HSACOKernelMeta::Arg::Kind::GlobalBuffer: return "GlobalBuffer";
      case HSACOKernelMeta::Arg::Kind::Hidden: return "Hidden";
      case HSACOKernelMeta::Arg::Kind::Unknown: return "Unknown";
      default: return "Undefined";
    }
  };

  auto parseArgKind = [](const std::string &value) -> HSACOKernelMeta::Arg::Kind {
    if (value.rfind("hidden_", 0) == 0) return HSACOKernelMeta::Arg::Kind::Hidden;
    else if (value == "by_value")
      return HSACOKernelMeta::Arg::Kind::ByValue;
    else if (value == "global_buffer")
      return HSACOKernelMeta::Arg::Kind::GlobalBuffer;
    else
      return HSACOKernelMeta::Arg::Kind::Unknown;
  };

  for (const std::unique_ptr<ELFIO::section> &s : reader.sections) {
    if (s->get_type() != SHT_NOTE) continue;
    // We only care about the .note section where the first record has the AMDGPU name
    auto sData = s->get_data();
    auto nhdr = reinterpret_cast<const Elf_Nhdr *>(sData);
    if (nhdr->n_type == NT_AMDGPU_METADATA && nhdr->name(sData) == "AMDGPU") {
      auto [descBegin, descEnd] = nhdr->desc(sData);
      auto kernels = nlohmann::json::from_msgpack(descBegin, descEnd).at("amdhsa.kernels");

//      static int kernelI = 0;
//      std::ofstream myfile;
//      myfile.open(std::to_string(kernelI++) + ".json");
//      myfile << kernels << "\n";
//      myfile.close();

      log("[HSACO] Found %zu kernels:", kernels.size());
      HSACOMeta meta(kernels.size());
      for (size_t i = 0; i < kernels.size(); ++i) {
        auto rawArgs = kernels[i].at(".args");
        std::vector<HSACOKernelMeta::Arg> args(rawArgs.size());
        for (size_t j = 0; j < args.size(); ++j) {
          auto rawArg = rawArgs.at(j);
          args[j] = {.offset = rawArg.at(".offset"), .size = rawArg.at(".size"), .kind = parseArgKind(rawArg.at(".value_kind"))};
        }
        meta[i].name = kernels[i].at(".name").get<std::string>();
        meta[i].demangledName = demangleCXXName(meta[i].name.c_str());
        meta[i].kernargSize = kernels[i].at(".kernarg_segment_size");
        meta[i].kernargAlign = kernels[i].at(".kernarg_segment_align");
        meta[i].args = args;
        log("[HSACO] \t%s", meta[i].name.c_str());
        log("[HSACO] \t - kernargSize:  %zu", meta[i].kernargSize);
        log("[HSACO] \t - kernargAlign: %zu", meta[i].kernargAlign);
        log("[HSACO] \t - args:" );
        for (size_t k = 0; k < meta[i].args.size(); ++k) {
          auto &arg = meta[i].args[k];
          log("[HSACO] \t   - %ld+%ld packed=%d, kind=%s", arg.size, arg.offset, meta[i].packed(k), kindName(arg.kind));
        }
      }
      return meta;
    }
  }
  log("[HSACO] ELF file at %p+%ld does not contain any AMDGPU metadata", data, length);
  return {};
}

// Adapted from https://stackoverflow.com/a/62160937/896997
std::string utpx::demangleCXXName(const char *abiName) {
  int failed;
  char *ret = abi::__cxa_demangle(abiName, nullptr /* output buffer */, nullptr /* length */, &failed);
  if (failed) {
    // 0: The demangling operation succeeded.
    // -1: A memory allocation failure occurred.
    // -2: mangled_name is not a valid name under the C++ ABI mangling rules.
    // -3: One of the arguments is invalid.
    return "";
  } else {
    return ret;
  }
}
