#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>

#include <thread>

#include <optional>
#include <poll.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>

#include "intercept_memory.h"
#include "utpx.h"

namespace utpx::fault {

namespace detail {

} // namespace detail

static long pageSize{};

static long GUARD_THREAD_TIMEOUT_SECONDS = 10;
static std::atomic_uintptr_t sigFaultAddress = 0;
static sem_t sigHandlerPendingEvent{}, sigHandlerPendingResume{};

static std::shared_mutex allocationLock{};
static std::unordered_map<void *, size_t> allocations{};

// static std::atomic_flag sigFaultLatch = ATOMIC_FLAG_INIT;

// FIXME we really need to adhere to signal-safety(7) in this whole block
// XXX POSIX calls only in the handler!
static void handler(int signal, siginfo_t *siginfo, void *context) {
  // XXX only handle SIGSEGV with ACCERR which is caused by r/w protected pages by mprotect
  if (signal != SIGSEGV || siginfo->si_code != SEGV_ACCERR) return;
  auto x86PC = static_cast<ucontext_t *>(context)->uc_mcontext.gregs[REG_RIP];
  log("[MEM] SIGSEGV: Accessing memory at address %p, code=%d, pc=0x%llx", siginfo->si_addr, siginfo->si_code, x86PC); // FIXME AS unsafe
  sigFaultAddress = reinterpret_cast<uintptr_t>(siginfo->si_addr);                                                     // AS safe
  ::sem_post(&sigHandlerPendingEvent);                                                                                 // AS safe
  timespec ts{};
  if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {               // AS safe
    log("[MEM] SIGSEGV: clock_gettime failed, terminating..."); // FIXME AS unsafe
    ::abort();                                                  // AS safe
  }
  ts.tv_sec += GUARD_THREAD_TIMEOUT_SECONDS;
  int res = sem_timedwait(&sigHandlerPendingResume, &ts); // FIXME AS unsafe
  if (res == -1) {
    log("[MEM] SIGSEGV: resume timeout: guard thread did not respond within %lds, terminating...",
        GUARD_THREAD_TIMEOUT_SECONDS); // FIXME AS unsafe
    ::abort();
  }
  // while (sigFaultLatch.test_and_set(std::memory_order_acquire)) // AS safe
  // {
  // }
  log("[MEM] SIGSEGV: resume %p", siginfo->si_addr); // FIXME AS unsafe
}

std::unique_ptr<std::thread> sigHandlerGuardThread{};
std::atomic_bool sigHandlerTerminate;

static void handleFault(void *faultAddr) {
  log("[MEM]\tUPH guard thread handling fault at address %p", faultAddr);
  if (const auto page = lookupRegisteredPage(faultAddr); page) {
    const auto &[allocAddr, allocLength] = *page;
    log("[MEM]\tSIGSEGV: resuming access to %p with mprotect(%p, %ld, PROT_READ | PROT_WRITE)", faultAddr, allocAddr, allocLength);
    if (mprotect(allocAddr, allocLength, PROT_READ | PROT_WRITE) != 0) {
      fatal("[MEM]\tFATAL: mprotect(%p, %ld, PROT_READ | PROT_WRITE) failed: %s", allocAddr, allocLength, strerror(errno));
    }
    handleUserspaceFault(faultAddr, allocAddr, allocLength);
    sigFaultAddress = 0;
    sem_post(&sigHandlerPendingResume);
    // sigFaultLatch.clear(std::memory_order_release);
  } else {
    fatal("[MEM]\tFATAL: address %p is not a registered page", faultAddr);
  }
}

void initialiseUserspacePagefaultHandling() {
  static_assert(std::atomic<bool>::is_always_lock_free);
  pageSize = sysconf(_SC_PAGE_SIZE);
  if (pageSize != -1) log("[MEM] page size = %ld", pageSize);
  else
    fatal("[MEM] Cannot resolve page size with sysconf, reason=%s, terminating...", strerror(errno));
  if (sem_init(&sigHandlerPendingEvent, 0, 0) == -1)
    fatal("[MEM] FATAL: Cannot create semaphore for sigHandlerPendingEvent, reason=%s, terminating...", strerror(errno));
  if (sem_init(&sigHandlerPendingResume, 0, 0) == -1)
    fatal("[MEM] FATAL: Cannot create semaphore for sigHandlerPendingResume, reason=%s, terminating...", strerror(errno));
  log("[MEM] UPH initialised");

  struct sigaction act {};
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_SIGINFO | SA_ONSTACK;
  act.sa_sigaction = handler;
  sigaction(SIGSEGV, &act, nullptr);
  log("[MEM] UPH signal handler installed");

  sem_post(&sigHandlerPendingEvent);
  sigHandlerGuardThread = std::make_unique<std::thread>([]() {
    log("[MEM]\tUPH guard thread started");
    while (true) {
      if (sem_wait(&sigHandlerPendingEvent) == -1) {
        std::abort();
      }
      if (sigHandlerTerminate) break;
      auto address = sigFaultAddress.load();
      if (address) handleFault(reinterpret_cast<void *>(address));
    }
    log("[MEM]\tUPH guard thread terminated");
  });
  log("[MEM] UPH initialised");
}

void terminateUserspacePagefaultHandling() {
  log("[MEM] UPH termination requested");
  std::unique_lock<std::shared_mutex> write(allocationLock);
  for (auto &[ptr, size] : allocations) {
    log("[MEM]\trelease: %p, %ld", ptr, size);
    if (mprotect(ptr, size, PROT_READ | PROT_WRITE) != 0) {
      log("[MEM]\tWARN: mprotect(%p, %ld, PROT_READ | PROT_WRITE) failed: %s", ptr, size, strerror(errno));
    }
  }
  sigHandlerTerminate = true;
  sem_post(&sigHandlerPendingEvent);
  if (sigHandlerGuardThread) sigHandlerGuardThread->join();
  log("[MEM] UPH terminated");
}

void registerPage(void *ptr, size_t size) {
  std::unique_lock<std::shared_mutex> write(allocationLock);
  log("[MEM] UPH register page (%p, %ld) total=%zu", ptr, size, allocations.size());
  auto [_, inserted] = allocations.emplace(ptr, size);
  if (inserted) {
    if (mprotect(ptr, size, PROT_NONE) != 0) {
      fatal("[MEM] mprotect failed, reason=%s, terminating...", strerror(errno));
    }
  } else {
    log("[MEM] UPH page already registered");
  }
}

void unregisterPage(void *ptr) {
  std::unique_lock<std::shared_mutex> write(allocationLock);
  log("[MEM] UPH unregister page (%p)", ptr);
  if (auto it = allocations.find(ptr); it != allocations.end()) {
    if (mprotect(it->first, it->second, PROT_READ | PROT_WRITE) != 0) {
      fatal("[MEM]\tmprotect(%p, %ld, PROT_READ | PROT_WRITE) failed: %s", it->first, it->second, strerror(errno));
    }
    allocations.erase(it);
  } else
    fatal("[MEM] UPH unregister nonexistent page (%p)", ptr);
}

std::optional<std::pair<void *, size_t>> lookupRegisteredPage(const void *ptr) {
  std::shared_lock<std::shared_mutex> read(allocationLock);
  for (auto &[addr, length] : allocations) {
    //    log("[MEM] \tUPH page (%p, %ld)", addr, length);

    auto signalAddrI = reinterpret_cast<uintptr_t>(ptr);
    auto allocAddrI = reinterpret_cast<uintptr_t>(addr);
    if (signalAddrI >= allocAddrI && signalAddrI < (allocAddrI + length)) {
      // std::fprintf(stderr, "offset from base=%d", (( ((uintptr_t) faultAddr)) - ( (uintptr_t) addr)));
      // size = length - (reinterpret_cast<uintptr_t>(faultAddr) - reinterpret_cast<uintptr_t>(addr));
      return std::pair{addr, length};
    }
  }
  return {};
}

size_t hostPageSize() { return pageSize; }

} // namespace utpx::fault
