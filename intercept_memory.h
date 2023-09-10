#pragma once

#include <list>
#include <mutex>
#include <shared_mutex>

namespace utpx::fault {

void initialiseUserspacePagefaultHandling();
void terminateUserspacePagefaultHandling();
void registerPage(void *ptr, size_t size);
void unregisterPage(void *ptr);
[[nodiscard]] std::optional<std::pair<void *, size_t>> lookupRegisteredPage(const void *ptr);
[[nodiscard]] size_t hostPageSize();

void handleUserspaceFault(void *faultAddr, void *allocAddr, size_t allocLength);

} // namespace utpx::fault
