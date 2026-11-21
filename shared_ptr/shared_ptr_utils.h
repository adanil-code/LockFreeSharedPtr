/*
* LockFreeSharedPtr
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

#pragma once

// ========================================================================================
// File Purpose & Architecture Overview
// ========================================================================================
// This file provides the cross-platform hardware, compiler, contention-backoff, and NUMA
// topology/memory-allocation foundation shared by AtomicSharedPtr and RundownSharedPtr
// across Windows, Linux, and macOS on x64 and ARM64 architectures.
//
// Core Responsibilities:
// 1. Hardware Geometry & Compiler Intrinsics:
//    Enforces 64-bit x64/ARM64 target compatibility, defines portable inlining macros
//    (FORCE_INLINE, NO_INLINE_ODR), maps architecture-native spin-wait instructions
//    (HARDWARE_PAUSE: _mm_pause on x64, __yield / yield on ARM64), and establishes
//    false-sharing alignment boundaries (CACHE_LINE_SIZE: 128 bytes on ARM64 / Apple
//    Silicon to match L2 spatial prefetcher pairs, 64 bytes on x64).
//
// 2. Multi-Stage Hybrid Lock-Free Contention Backoff (AdaptiveBackoff):
//    Implements a three-phase contention backoff state machine tuned for symmetric
//    multiprocessors, hybrid P-Core/E-Core topologies, and oversubscribed virtual
//    machines:
//      - Stage 0 (step < 4)   : Single hardware pause for ultra-low-latency retry.
//      - Stage 1 (step < 32)  : Bounded exponential hardware pause loop (up to 64 pauses)
//                               keeping contention resolution entirely in user mode.
//      - Stage 2 (step >= 32) : Bounded hardware pause loop paired with an OS scheduler
//                               yield (sched_yield / std::this_thread::yield) every 8th
//                               step to break oversubscription and priority-inversion stalls.
//
// 3. Zero-Allocation NUMA Topology Discovery & Amortized Node Tracking (NumaAllocator):
//    Discovers online physical NUMA nodes and logical-CPU-to-node mappings once into a
//    trivially destructible static POD table (TopologyState) with zero heap allocations:
//      - Windows : Queries processor groups and node masks via GetNumaHighestNodeNumber,
//                  GetNumaAvailableMemoryNodeEx, GetNumaNodeProcessorMaskEx, and
//                  GetNumaProcessorNodeEx.
//      - Linux   : Queries libnuma when <numa.h> is available, or parses sysfs node
//                  cpulist files (/sys/devices/system/node/node*/cpulist) when compiled
//                  without libnuma.
//      - macOS   : Resolves to a single UMA node (Node 0) with zero syscall overhead.
//    GetCurrentNode() packs the cached node ID and refresh tick into a single constant-
//    initialized thread_local uint64_t, refreshing every 64 calls via
//    GetCurrentProcessorNumberEx (Windows) or sched_getcpu (Linux).
//
// 4. Node-Affinitized Virtual / Heap Memory Allocation (Allocate / Free):
//    Allocates and releases page- or slab-granularity memory physically bound to a target
//    NUMA node via VirtualAllocExNuma / VirtualFree (Windows), numa_alloc_onnode /
//    numa_free or mmap + SYS_mbind / munmap (Linux), or 128-byte aligned ::operator new[]
//    / ::operator delete[] on single-node UMA and macOS systems.
// ========================================================================================

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <limits>
#include <new>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <sched.h>
#endif

#if defined(__linux__)
#if __has_include(<numa.h>)
#include <numa.h>
#define SHAREDPTR_HAS_LIBNUMA 1
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <cstdio>
#include <cstdlib>
#define SHAREDPTR_HAS_LIBNUMA 0
#endif
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// ========================================================================================
// Architecture Compatibility Verification
// ========================================================================================
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__) && !defined(_M_ARM64)
#error Unsupported architecture. Implementation requires x64 or ARM64.
#endif

// ========================================================================================
// Compiler Inlining Attributes
// ========================================================================================
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#define NO_INLINE_ODR __declspec(noinline) inline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#define NO_INLINE_ODR __attribute__((noinline)) inline
#else
#define FORCE_INLINE inline
#define NO_INLINE_ODR inline
#endif

// ========================================================================================
// Platform-Specific Intrinsic Hardware Pauses
// ========================================================================================
#ifndef HARDWARE_PAUSE
#if defined(_MSC_VER) && defined(_M_ARM64)
#define HARDWARE_PAUSE() __yield()
#elif defined(_MSC_VER)
#define HARDWARE_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define HARDWARE_PAUSE() _mm_pause()
#elif defined(__aarch64__)
// 'yield' maps conceptually to an explicit ARM spin-wait better than an ISB
#define HARDWARE_PAUSE() __asm__ volatile("yield" ::: "memory")
#else
#define HARDWARE_PAUSE() \
    do                   \
    {                    \
    } while (0)
#endif
#endif

// ========================================================================================
// Cache Line Geometry
// ========================================================================================
// Explicit architecture-native cache-line padding to prevent False Sharing while
// guaranteeing a stable cross-TU ABI (avoiding std::hardware_destructive_interference_size
// -mtune ABI drift and GCC -Winterference-size warnings in public headers):
//   - ARM64 (including Apple Silicon): 128 bytes (matches Apple L1/L2 lines and ARM64 L2
//     spatial prefetcher pairs).
//   - x64 (Windows, Linux, Intel macOS): 64 bytes.
#if defined(__aarch64__) || defined(_M_ARM64)
inline constexpr std::size_t CACHE_LINE_SIZE = 128;
#elif defined(__x86_64__) || defined(_M_X64)
inline constexpr std::size_t CACHE_LINE_SIZE = 64;
#else
#error Unsupported architecture. Implementation requires x64 or ARM64.
#endif

// ========================================================================================
// Multi-Stage Hybrid Lock-Free Backoff Policy
// ========================================================================================
// This backoff strategy resolves the contention differences between
// symmetric CPUs, hybrid architectures (P-Core/E-Core), and virtual machines.
// It uses a three-stage escalation to maximize throughput:
//   - Stage 0 (step < 4)   : Single hardware pause for low-latency retry.
//   - Stage 1 (step < 32)  : Bounded exponential hardware pause loop (up to 64 pauses).
//   - Stage 2 (step >= 32) : Bounded hardware pause loop coupled with an OS scheduler
//                            yield every 8th step to break oversubscription livelocks.
class AdaptiveBackoff
{
    int step = 0; // Current escalation step counter tracking consecutive contention retries

public:
    // ------------------------------------------------------------------------------------
    // operator()
    // Purpose:
    //   Executes the next backoff escalation step (pause or yield). Private logic
    //   handles escalation automatically based on counter progression.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    FORCE_INLINE void operator()() noexcept
    {
        if (step < 4)
        {
            // Fast path: single hardware pause for uncontended immediate retry without
            // starving SMT sibling threads.
            HARDWARE_PAUSE();
        }
#if defined(__linux__) || defined(__APPLE__)
        else if (step < 32)
        {
            // Bounded hardware exponential backoff for POSIX.
            int pauses = 1 << std::min<int>(step - 4, 6);
            for (int i = 0; i < pauses; ++i)
            {
                HARDWARE_PAUSE();
            }
        }
        else
        {
            // POSIX-only starvation release under heavy CPU oversubscription.
            int pauses = 1 << std::min<int>(step - 4, 6);
            for (int i = 0; i < pauses; ++i)
            {
                HARDWARE_PAUSE();
            }

            if ((step & 7) == 0)
            {
                sched_yield();
            }
        }
#else
        else if (step < 32)
        {
            // Phase 1: Pure hardware spin. 
            // Maintains peak MPMC throughput by keeping contention entirely in user-space.
            int pauses = 1 << std::min<int>(step - 4, 6);
            for (int i = 0; i < pauses; ++i)
            {
                HARDWARE_PAUSE();
            }
        }
        else
        {
            // Phase 2: Livelock & oversubscription priority-inversion breaker.
            // Continues hardware pausing, but yields to the OS scheduler every 8th step
            // once hardware pauses are exhausted so preempted threads can complete their window.
            int pauses = 1 << std::min<int>(step - 4, 6);
            for (int i = 0; i < pauses; ++i)
            {
                HARDWARE_PAUSE();
            }

            if ((step & 7) == 0)
            {
                std::this_thread::yield();
            }
        }
#endif
        // Prevent undefined behavior on long stalls, though practically unreachable
        if (step < (std::numeric_limits<int>::max)())
        {
            step++;
        }
    }
};

// ========================================================================================
// Cross-Platform User-Mode NUMA Topology & Memory Allocator (Windows / Linux / macOS)
// ========================================================================================
// Provides OS-level NUMA node discovery, fast amortized per-thread CPU-to-node resolution,
// and physical node-affinitized memory allocation:
//   - Windows : Uses GetNumaHighestNodeNumber, GetNumaAvailableMemoryNodeEx,
//               GetNumaNodeProcessorMaskEx, GetNumaProcessorNodeEx,
//               GetCurrentProcessorNumberEx, and VirtualAllocExNuma (with VirtualAlloc
//               fallback) and VirtualFree.
//   - Linux   : Uses libnuma (numa_available, numa_alloc_onnode, numa_alloc_local, numa_free)
//               when <numa.h> is present, or sysfs topology parsing + mmap/mbind/munmap
//               fallback, paired with sched_getcpu for amortized thread-local node tracking.
//   - macOS   : Gracefully resolves to a single UMA node (Node 0) backed by cache-aligned
//               ::operator new[] / ::operator delete[] with zero syscall overhead.
// ========================================================================================
struct NumaAllocator
{
    static constexpr uint32_t MAX_NUMA_NODES   = 64;          // Maximum number of physical NUMA nodes tracked in static lookup tables
    static constexpr uint32_t MAX_LOGICAL_CPUS = 1024;        // Maximum number of logical processors mapped in cpu_to_node
    static constexpr uint32_t ANY_NODE_OK      = 0xFFFFFFFFu; // Sentinel indicating caller prefers the current thread's local NUMA node

    // ------------------------------------------------------------------------------------
    // TopologyState
    // Trivially destructible, zero-heap-allocation POD table storing discovered NUMA nodes
    // and logical-CPU-to-node mappings. Because it owns no dynamic memory, it can never
    // throw std::bad_alloc during initialization and remains valid across static and
    // thread_local teardown at process exit.
    // ------------------------------------------------------------------------------------
    struct TopologyState
    {
        uint32_t nodes[MAX_NUMA_NODES]{0};          // Contiguous list of discovered active physical NUMA node IDs
        uint32_t node_count{1};                     // Total count of valid entries populated in the nodes[] array
        uint32_t max_node_id{0};                    // Highest valid physical NUMA node ID discovered on the system
        uint8_t  is_valid_node[MAX_NUMA_NODES]{1};  // O(1) boolean lookup table indicating whether a given node ID is online
        uint16_t cpu_to_node[MAX_LOGICAL_CPUS]{0};  // Maps flat logical processor index (0 .. MAX_LOGICAL_CPUS - 1) to its NUMA node ID
        bool     is_numa_available{false};          // True if multi-node NUMA topology or libnuma support is active

        // --------------------------------------------------------------------------------
        // begin
        // Purpose:
        //   Returns a const pointer to the first active NUMA node ID in the nodes array.
        // Parameters:
        //   None.
        // Return Value:
        //   Const pointer to the beginning of the active nodes sequence.
        // --------------------------------------------------------------------------------
        [[nodiscard]] constexpr const uint32_t* begin() const noexcept { return &nodes[0]; }

        // --------------------------------------------------------------------------------
        // end
        // Purpose:
        //   Returns a const pointer one past the last active NUMA node ID in the nodes array.
        // Parameters:
        //   None.
        // Return Value:
        //   Const pointer to the end of the active nodes sequence.
        // --------------------------------------------------------------------------------
        [[nodiscard]] constexpr const uint32_t* end() const noexcept { return &nodes[node_count]; }

        // --------------------------------------------------------------------------------
        // size
        // Purpose:
        //   Returns the number of active physical NUMA nodes discovered on the system.
        // Parameters:
        //   None.
        // Return Value:
        //   Count of active NUMA nodes (at least 1).
        // --------------------------------------------------------------------------------
        [[nodiscard]] constexpr size_t size() const noexcept { return node_count; }

        // --------------------------------------------------------------------------------
        // operator[]
        // Purpose:
        //   Retrieves the physical NUMA node ID at index i within the active nodes list.
        // Parameters:
        //   i - Zero-based index into the active nodes array (0 .. size() - 1).
        // Return Value:
        //   The physical NUMA node ID stored at index i.
        // --------------------------------------------------------------------------------
        [[nodiscard]] constexpr uint32_t operator[](size_t i) const noexcept { return nodes[i]; }

        // --------------------------------------------------------------------------------
        // IsValidNode
        // Purpose:
        //   Performs an O(1) bounds-checked lookup to verify whether the specified
        //   physical NUMA node ID is valid and online.
        // Parameters:
        //   node - Physical NUMA node ID to validate.
        // Return Value:
        //   true if node is within bounds (< MAX_NUMA_NODES) and online; false otherwise.
        // --------------------------------------------------------------------------------
        [[nodiscard]] FORCE_INLINE bool IsValidNode(uint32_t node) const noexcept
        {
            return node < MAX_NUMA_NODES && is_valid_node[node] != 0;
        }
    };

    // ------------------------------------------------------------------------------------
    // GetTopology
    // Purpose:
    //   Discovers active NUMA nodes and populates the static POD lookup tables once.
    //   Guaranteed noexcept with zero heap allocations.
    // Parameters:
    //   None.
    // Return Value:
    //   Const reference to the process-wide static TopologyState instance.
    // ------------------------------------------------------------------------------------
    static const TopologyState& GetTopology() noexcept
    {
        static const TopologyState state = []() noexcept
        {
            TopologyState topo{};
            topo.node_count        = 0;
            topo.max_node_id      = 0;
            topo.is_numa_available = false;
            topo.is_valid_node[0]  = 0;

#if defined(_WIN32)
            ULONG highestNode = 0;

            if (GetNumaHighestNodeNumber(&highestNode)) [[likely]]
            {
                const USHORT maxNode = static_cast<USHORT>(highestNode < (MAX_NUMA_NODES - 1) ? highestNode : (MAX_NUMA_NODES - 1));

                for (USHORT i = 0; i <= maxNode; ++i)
                {
                    ULONGLONG      availableMemory = 0;
                    GROUP_AFFINITY processorMask{};
                    if (GetNumaAvailableMemoryNodeEx(i, &availableMemory) ||
                        (GetNumaNodeProcessorMaskEx(i, &processorMask) && processorMask.Mask != 0))
                    {
                        topo.nodes[topo.node_count++] = static_cast<uint32_t>(i);
                        topo.is_valid_node[i]         = 1;
                        topo.max_node_id              = static_cast<uint32_t>(i);
                    }
                }

                // Map all active logical processors across processor groups using each
                // group's actual processor count (supporting groups with < 64 logical CPUs).
                const WORD groupCount = GetActiveProcessorGroupCount();
                for (WORD group = 0; group < groupCount; ++group)
                {
                    const DWORD procsInGroup = GetActiveProcessorCount(group);
                    for (DWORD number = 0; number < procsInGroup && number < 64u; ++number)
                    {
                        const uint32_t flatIdx = static_cast<uint32_t>(group) * 64u + static_cast<uint32_t>(number);
                        if (flatIdx >= MAX_LOGICAL_CPUS)
                        {
                            break;
                        }

                        PROCESSOR_NUMBER procNum{};
                        procNum.Group     = group;
                        procNum.Number    = static_cast<BYTE>(number);
                        USHORT nodeNumber = 0;
                        if (GetNumaProcessorNodeEx(&procNum, &nodeNumber) &&
                            nodeNumber < MAX_NUMA_NODES &&
                            topo.is_valid_node[nodeNumber])
                        {
                            topo.cpu_to_node[flatIdx] = static_cast<uint16_t>(nodeNumber);
                        }
                    }
                }
            }
#elif defined(__linux__) && SHAREDPTR_HAS_LIBNUMA
            if (numa_available() >= 0) [[likely]]
            {
                // Do not set topo.is_numa_available = true here because numa_available()
                // returns 0 even on single-socket (1-node UMA) Linux kernels with CONFIG_NUMA=y.
                // The unified (topo.node_count > 1) check below enables NUMA only on multi-node systems.
                const int highestNode  = numa_max_node();
                const int maxNode      = highestNode < static_cast<int>(MAX_NUMA_NODES - 1) ? highestNode : static_cast<int>(MAX_NUMA_NODES - 1);

                for (int i = 0; i <= maxNode; ++i)
                {
                    if (numa_bitmask_isbitset(numa_all_nodes_ptr, i))
                    {
                        topo.nodes[topo.node_count++] = static_cast<uint32_t>(i);
                        topo.is_valid_node[i]         = 1;
                        topo.max_node_id              = static_cast<uint32_t>(i);
                    }
                }

                for (uint32_t cpu = 0; cpu < MAX_LOGICAL_CPUS; ++cpu)
                {
                    const int node = numa_node_of_cpu(static_cast<int>(cpu));
                    if (node >= 0 && node < static_cast<int>(MAX_NUMA_NODES) && topo.is_valid_node[node])
                    {
                        topo.cpu_to_node[cpu] = static_cast<uint16_t>(node);
                    }
                }
            }
#elif defined(__linux__)
            // Fallback sysfs topology discovery when compiling on Linux without <numa.h>
            for (uint32_t node = 0; node < MAX_NUMA_NODES; ++node)
            {
                char path[128];
                std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpulist", node);
                FILE* fp = std::fopen(path, "r");
                if (!fp)
                {
                    continue;
                }

                // 4096-byte buffer accommodates worst-case interleaved CPU lists up to MAX_LOGICAL_CPUS
                char buf[4096]{};
                if (std::fgets(buf, sizeof(buf), fp))
                {
                    topo.nodes[topo.node_count++] = node;
                    topo.is_valid_node[node]      = 1;
                    topo.max_node_id              = node;

                    const char* p = buf;
                    while (*p)
                    {
                        char*         end_p     = nullptr;
                        unsigned long start_cpu = std::strtoul(p, &end_p, 10);
                        if (end_p == p)
                        {
                            break;
                        }
                        unsigned long end_cpu = start_cpu;
                        p = end_p;

                        if (*p == '-')
                        {
                            ++p;
                            end_cpu = std::strtoul(p, &end_p, 10);
                            p       = end_p;
                        }

                        for (unsigned long cpu = start_cpu; cpu <= end_cpu && cpu < MAX_LOGICAL_CPUS; ++cpu)
                        {
                            topo.cpu_to_node[cpu] = static_cast<uint16_t>(node);
                        }

                        if (*p == ',')
                        {
                            ++p;
                        }
                    }
                }
                std::fclose(fp);
            }
#endif

            if (topo.node_count == 0) [[unlikely]]
            {
                topo.nodes[0]         = 0;
                topo.node_count       = 1;
                topo.max_node_id      = 0;
                topo.is_valid_node[0] = 1;
            }

            if (topo.node_count > 1)
            {
                topo.is_numa_available = true;
            }

            return topo;
        }();

        return state;
    }

    // ------------------------------------------------------------------------------------
    // GetValidNodes
    // Purpose:
    //   Returns a std::vector<uint32_t> of valid physical NUMA node IDs on the system.
    //   Pre-allocates the single-node fallback vector outside the try-block so the catch
    //   handler never allocates heap memory inside a noexcept function.
    // Parameters:
    //   None.
    // Return Value:
    //   Const reference to a static vector containing all online physical NUMA node IDs.
    // ------------------------------------------------------------------------------------
    static const std::vector<uint32_t>& GetValidNodes() noexcept
    {
        // Pre-allocated fallback ensures OOM inside validNodes lambda never calls std::terminate
        static const std::vector<uint32_t> fallback{ 0 };

        static const std::vector<uint32_t> validNodes = []() noexcept -> std::vector<uint32_t>
        {
            try
            {
                const TopologyState&  topo = GetTopology();
                std::vector<uint32_t> nodes(topo.begin(), topo.end());
                nodes.shrink_to_fit();
                return nodes;
            }
            catch (...)
            {
                // Default-constructed vector is constexpr noexcept and performs 0 allocations
                return std::vector<uint32_t>();
            }
        }();

        return validNodes.empty() ? fallback : validNodes;
    }

    // ------------------------------------------------------------------------------------
    // GetCurrentNode
    // Purpose:
    //   Resolves the physical NUMA node of the processor currently executing the calling
    //   thread. On single-node UMA systems and macOS, returns Node 0 immediately without
    //   OS calls. On multi-socket NUMA systems, packs the cached node ID and tick counter
    //   into a single constant-initialized thread_local uint64_t (avoiding dynamic TLS
    //   initialization guards) and refreshes every 64 operations.
    // Parameters:
    //   None.
    // Return Value:
    //   The 32-bit physical NUMA node ID for the calling processor.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] static FORCE_INLINE uint32_t GetCurrentNode() noexcept
    {
        const TopologyState& topo = GetTopology();
        if (topo.node_count <= 1) [[likely]]
        {
            return topo.nodes[0];
        }

        // Upper 32 bits store the cached NUMA node ID; lower 32 bits store the tick counter.
        // Constant zero-initialization avoids C++ dynamic TLS guard checks on every call,
        // and (tick == 0) forces an immediate hardware lookup on the thread's first touch.
        thread_local uint64_t tls_state = 0;

        const uint32_t tick = static_cast<uint32_t>(tls_state);
        uint32_t       node = static_cast<uint32_t>(tls_state >> 32);

        if ((tick & 63u) == 0u) [[unlikely]]
        {
            node = topo.nodes[0];
#if defined(_WIN32)
            PROCESSOR_NUMBER procNum{};
            GetCurrentProcessorNumberEx(&procNum);
            const uint32_t flatIdx = static_cast<uint32_t>(procNum.Group) * 64u +
                                     static_cast<uint32_t>(procNum.Number);
            if (flatIdx < MAX_LOGICAL_CPUS) [[likely]]
            {
                node = topo.cpu_to_node[flatIdx];
            }
#elif defined(__linux__)
            const int cpu = sched_getcpu();
            if (cpu >= 0 && static_cast<uint32_t>(cpu) < MAX_LOGICAL_CPUS) [[likely]]
            {
                node = topo.cpu_to_node[static_cast<uint32_t>(cpu)];
            }
#endif
        }

        tls_state = (static_cast<uint64_t>(node) << 32) | static_cast<uint64_t>(tick + 1u);
        return node;
    }

    // ------------------------------------------------------------------------------------
    // Allocate
    // Purpose:
    //   Allocates page/slab-granularity memory physically affinitized to the specified NUMA
    //   node, falling back to local/default OS allocation if node-specific allocation fails.
    // Parameters:
    //   size - Number of bytes to allocate.
    //   node - Target physical NUMA node ID, or ANY_NODE_OK to use GetCurrentNode().
    // Return Value:
    //   Pointer to the allocated memory region, or nullptr if size is 0 or allocation fails.
    // ------------------------------------------------------------------------------------
    static inline void* Allocate(size_t   size,
                                 uint32_t node = ANY_NODE_OK) noexcept
    {
        if (size == 0) [[unlikely]]
        {
            return nullptr;
        }

        const TopologyState& topo       = GetTopology();
        const uint32_t       targetNode = (node != ANY_NODE_OK && topo.IsValidNode(node)) ? node : GetCurrentNode();

#if defined(_WIN32)
        void* ptr = VirtualAllocExNuma(GetCurrentProcess(),
                                       NULL,
                                       size,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_READWRITE,
                                       targetNode);

        if (!ptr) [[unlikely]]
        {
            ptr = VirtualAlloc(NULL,
                               size,
                               MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
        }

        return ptr;
#elif defined(__linux__) && SHAREDPTR_HAS_LIBNUMA
        if (topo.is_numa_available) [[likely]]
        {
            void* ptr = numa_alloc_onnode(size, static_cast<int>(targetNode));
            if (!ptr) [[unlikely]]
            {
                ptr = numa_alloc_local(size);
            }

            return ptr;
        }

        return ::operator new[](size, std::align_val_t{ 128 }, std::nothrow);
#elif defined(__linux__)
        if (topo.is_numa_available) [[likely]]
        {
            void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr != MAP_FAILED)
            {
#if defined(SYS_mbind)
                if (targetNode < 64)
                {
                    constexpr int MPOL_PREFERRED_MODE = 1;
                    // Pass 65UL with a 2-word mask because Linux's get_nodes() decrements maxnode by 1
                    const unsigned long nodemask[2] = { (1UL << targetNode), 0UL };
                    (void)::syscall(SYS_mbind, ptr, size, MPOL_PREFERRED_MODE, nodemask, 65UL, 0U);
                }
#endif
                return ptr;
            }

            return nullptr;
        }

        return ::operator new[](size, std::align_val_t{ 128 }, std::nothrow);
#else
        // macOS (__APPLE__) and other non-NUMA platforms: allocate 128-byte aligned heap block
        (void)topo;
        (void)targetNode;
        return ::operator new[](size, std::align_val_t{ 128 }, std::nothrow);
#endif
    }

    // ------------------------------------------------------------------------------------
    // Free
    // Purpose:
    //   Releases memory previously allocated by NumaAllocator::Allocate back to the OS
    //   virtual memory manager or aligned heap.
    // Parameters:
    //   ptr  - Pointer to the memory block to release (no-op if nullptr).
    //   size - Byte size of the allocation as originally passed to Allocate().
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    static inline void Free(void*  ptr,
                            size_t size) noexcept
    {
        if (!ptr) [[unlikely]]
        {
            return;
        }

#if defined(_WIN32)
        (void)size;
        VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(__linux__) && SHAREDPTR_HAS_LIBNUMA
        if (GetTopology().is_numa_available) [[likely]]
        {
            numa_free(ptr, size);
        }
        else
        {
            ::operator delete[](ptr, std::align_val_t{ 128 });
        }
#elif defined(__linux__)
        if (GetTopology().is_numa_available) [[likely]]
        {
            ::munmap(ptr, size);
        }
        else
        {
            ::operator delete[](ptr, std::align_val_t{ 128 });
        }
#else
        (void)size;
        ::operator delete[](ptr, std::align_val_t{ 128 });
#endif
    }
};