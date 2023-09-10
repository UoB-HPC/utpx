#pragma once

#include <cstdlib>

#include <dlfcn.h>
#include <unistd.h>

namespace utpx {

void log(const char *fmt, ...);
[[noreturn]]  void fatal(const char *fmt, ...);

#define log(fmt, ...) fprintf(stderr, fmt "\n" __VA_OPT__(,) __VA_ARGS__)
#define fatal(fmt, ...) do  { fprintf(stderr,fmt "\n" __VA_OPT__(,)  __VA_ARGS__); std::abort(); } while(0)

#define log(fmt, ...)
#define fatal(fmt, ...) std::abort()



template <typename T> T dlSymbol(const char *symbol_name, const char *so = nullptr) {
  static T fn;
  if (!fn) {
    fn = (T)dlsym(RTLD_NEXT, symbol_name);
    if (fn) log("[DLSYM] Found %s at %p", symbol_name, (void *)fn);
    else {
      if (!so) {
        log("[DLSYM] Missing original %s and no library is specified to find this symbol, terminating...", symbol_name);
        std::abort();
      }
      log("[DLSYM] Missing original %s, trying to load directly from %s", symbol_name, so);
      auto handle = dlopen(so, RTLD_LAZY);
      if (!handle) {
        log("[DLSYM] dlopen failed for %s when resolving for %s, reason=%s, terminating...", so, symbol_name, dlerror());
        std::abort();
      }
      dlerror(); // clear existing errors
      fn = (T)dlsym(handle, symbol_name);
      if (auto e = dlerror(); e) {
        log("[DLSYM] dlsym failed for %s, reason=%s, terminating...", symbol_name, e);
        std::abort();
      }
    }
  }
  return fn;
}

} // namespace utpx
