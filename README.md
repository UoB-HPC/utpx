UTPX
----

UTPX (Userspace Transparent Paging Extension) is a `LD_PRELOAD` library that accelerates HIP
managed allocations (e.g `hipMallocManaged`) on systems without XNACK or with XNACK disabled.

Without XNACK, the GPU device cannot handle pagefaults and ROCm degrades the allocation to
host-resident memory where all access from the device must cross the host-device interconnect (e.g.,
PCIe).
In this degraded mode, no page migration occurs, so performance for memory-bandwidth bound
applications will be limited to interconnect performance.
For example, a MI100 operating at PCIe 4.0 x16 without XNACK will see application performance capped
at 31.5GB/s when device-resident memory should be capable of 1228.8 GB/s, a 40x difference.

UTPX solves this by shifting pagefault handling to the host where it can be easily accomplished in
the userspace.
The implementation performs userspace page migration using `mprotect` and signal handlers.

Paging is done at the granularity of the allocation itself using a *Mirror-on-Access* (MoA) scheme.
With MoA, initial allocations are resident on the host, and a separate device resident allocation is
made whenever a kernel is launched that has dependency on the allocation.
A device to host write-back is triggered by a `mprotect` induced pagefault, this happens if the host
mirror of the device-resident memory is accessed in any way (e.g., through pointer dereference).

## Status

UTPX requires certain HIP memory APIs to be intercepted for the mirrored allocation to work
correctly. A minimal set are implemented such
that [roc-stdpar](https://github.com/ROCmSoftwarePlatform/roc-stdpar) and HIP versions of HPC
mini-apps work:

* `<<<>>>`/`hipLaunchKernelGGL`
* `hipDeviceSynchronize`
* `hipMallocManaged`
* `hipFree`
* `hipMemcpy`
* `hipMemset`
* `hipPointerGetAttributes` (partial, only works in roc-stdpar)
* Any device query or stream API, those do not require special handling

## Usage

```shell
# On RadeonVII
# without UTPX:
> ./std-indices-stream
BabelStream
Version: 4.0
Implementation: STD (index-oriented)
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Backing storage typeid: Pd
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        21989.273   0.02442     0.02462     0.02450     
Mul         22020.178   0.02438     0.02460     0.02449     
Add         19225.037   0.04189     0.04197     0.04193     
Triad       19229.798   0.04188     0.04204     0.04193     
Dot         13468.631   0.03986     0.04003     0.03994   

# With UTPX:
> LD_PRELOAD="$HOME/utpx/build/libutpx.so" ./std-indices-stream
BabelStream
Version: 4.0
Implementation: STD (index-oriented)
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Backing storage typeid: Pd
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        845300.937  0.00064     0.00064     0.00064     
Mul         842076.480  0.00064     0.00070     0.00065     
Add         835445.048  0.00096     0.00103     0.00097     
Triad       833085.608  0.00097     0.00101     0.00097     
Dot         851026.490  0.00063     0.00065     0.00064 
```

## Building

### Dependencies

* CMake >= 3.14
* C++17-capable compiler
* Linux (Win32 is not supported)
* **A ROCm (HIP/HSA) installation is not required** and will not be used

```shell
cd utpx
cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
# library available at build/libutpx.so
```