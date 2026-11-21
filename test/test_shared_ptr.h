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

// ============================================================================
// File Purpose & Architecture Overview
// ============================================================================
// This file implements the cross-platform User-Mode verification, stress-soak,
// tail-latency, and multi-core scaling benchmark harness comparing three
// shared-pointer architectures:
//   - Baseline : std::atomic<std::shared_ptr<TrackedObject>>
//   - Atomic   : AtomicSharedPtr<TrackedObject> (128-bit CAS Split-Count)
//   - Rundown  : RundownSharedPtr<TrackedObject> (Epoch-Based Cache-Aware Rundown)
//
// Suite Structure (Phases 0 - 19):
//
// 1. Functional & API Correctness (Phases 0 - 2b):
//    Validates C++ smart-pointer handle semantics, split ref-count folding,
//    l-value/r-value CAS rollback guarantees, spurious failure masking under
//    reader contention, and execute-around/persistent promotion with explicit
//    GC barrier flushes for Rundown anchors.
//
// 2. Contention & Oversubscription Soak (Phases 3 - 8):
//    Measures sustained MPMC throughput (75% readers / 25% writers) at 1x
//    hardware concurrency and stresses scheduler preemption resilience under
//    6x CPU thread oversubscription (up to 256 concurrent user-mode threads).
//
// 3. Reclamation Safety & Chaos Testing (Phases 9 - 15):
//    Verifies use-after-free immunity via 25-step in-flight payload snapshots
//    during continuous writer displacement, 100% atomic exchange() rotation,
//    and forced scheduler quantum yields (std::this_thread::yield) to trigger
//    core migrations across lock-free hazard windows.
//
// 4. Tail Latency & Multi-Core Scaling Analytics (Phases 16 - 19):
//    - Tail Latency (Phases 16 - 18): Captures per-read hardware cycle deltas
//      in 1M-entry per-reader ring buffers under paced 3us writer mutation,
//      computing P50, P90, P99.9, and P99.99 latencies in nanoseconds.
//    - Thread Scaling Sweep (Phase 19): Sweeps 1 to N hardware threads across
//      80/20, 99/1, and 99.9/0.1 read/write ratios using pre-allocated payload
//      ring buffers, staggered read/write phases, and explicit inter-step GC
//      flushes to isolate synchronization scaling from heap lock contention.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <latch>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
#endif

#include "atomic_shared_ptr.h"
#include "rundown_shared_ptr.h"

// ============================================================================
// Test Suite Constants & Global State
// ============================================================================
constexpr int          MAX_TEST_THREADS              = 256;     // Maximum concurrent worker threads spawned in any test phase
constexpr unsigned int MAX_SCALING_POINTS            = 40;      // Maximum thread-count steps recorded during Phase 19 scaling sweeps
constexpr size_t       LATENCY_RING_SIZE             = 1048576; // Per-reader power-of-two sample ring capacity (2^20 entries)

// High-watermark and periodic wake interval matching the KM benchmark suite (65,535 entries / 2s)
constexpr int32_t      GC_HIGH_WATERMARK             = 65535;   // Retired block depth threshold that triggers an asynchronous GC wake
constexpr uint32_t     GC_PERIODIC_INTERVAL_MS       = 2000;    // Background GC worker periodic timeout interval in milliseconds

// Scaling read-to-write ratios:
//   - 80.0% Read / 20.0% Write :   4 reads per 1 write (  4 /    5)
//   - 99.0% Read /  1.0% Write :  99 reads per 1 write ( 99 /  100)
//   - 99.9% Read /  0.1% Write : 999 reads per 1 write (999 / 1000)
constexpr int          SCALING_READS_PER_WRITE_80_20  = 4;      // Read batch count per write for the 80% Read / 20% Write sweep
constexpr int          SCALING_READS_PER_WRITE_99_1   = 99;     // Read batch count per write for the 99% Read / 1% Write sweep
constexpr int          SCALING_READS_PER_WRITE_999_01 = 999;    // Read batch count per write for the 99.9% Read / 0.1% Write sweep

// Cache-line isolated instance counter to prevent false sharing with control flags during GC sweeps
alignas(CACHE_LINE_SIZE) static std::atomic<int32_t> g_AbortRequested{0};  // Global teardown flag set upon assertion failure or Ctrl-C
static std::atomic<int32_t>                          g_TestFailed{0};      // Non-zero if any assertion or fatal handler fired during the run
static std::atomic<int32_t>                          g_PausePeriodicGc{0}; // Suppresses periodic background GC timer wakes during Phase 19 steps
static uint32_t                                      g_SoakDurationMs = 10000; // Execution duration per soak benchmark phase in milliseconds
static uint64_t                                      g_CooldownMs     = 0;     // Computed thermal/GC cooldown delay between heavy benchmark phases
static double                                        g_CyclesPerUs    = 0.0;   // Calibrated hardware timestamp counter cycles per microsecond
static double                                        g_CyclesPerPause = 1.0;   // Calibrated average hardware cycles elapsed per HARDWARE_PAUSE()

// Optional callback hook invoked before and after each test phase to synchronously drain deferred GC queues
inline void (*g_TestTeardownHook)() = nullptr;

// ============================================================================
// Timing, Concurrency & Output Utilities
// ============================================================================

// ----------------------------------------------------------------------------
// Formats and prints a string to stdout and flushes the output stream immediately
// ----------------------------------------------------------------------------
inline void Print(const char* format, 
                  ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    fflush(stdout);
}

// ----------------------------------------------------------------------------
// Returns the current monotonic wall-clock timestamp in milliseconds
// ----------------------------------------------------------------------------
inline uint64_t GetTimeMs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// ----------------------------------------------------------------------------
// Returns the current monotonic wall-clock timestamp in nanoseconds
// ----------------------------------------------------------------------------
inline uint64_t GetTimeNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// ----------------------------------------------------------------------------
// Reads the processor's invariant hardware cycle counter (RDTSC on x64,
// CNTVCT / monotonic raw clock on ARM64) for sub-nanosecond interval timing
// ----------------------------------------------------------------------------
FORCE_INLINE uint64_t GetHardwareCycles()
{
#if defined(_MSC_VER)
#if defined(_M_ARM64)
    return _ReadStatusReg(ARM64_CNTVCT);
#else
    return __rdtsc();
#endif
#elif defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#elif defined(__aarch64__) && defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#elif defined(__aarch64__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL) + static_cast<uint64_t>(ts.tv_nsec);
#else
    return static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
#endif
}

// ----------------------------------------------------------------------------
// Queries the number of logical hardware threads available on the host system
// ----------------------------------------------------------------------------
inline unsigned int GetHardwareThreads()
{
    unsigned int hw = std::thread::hardware_concurrency();
    
    if (hw == 0)
    {
        return 4;
    }
    
    return hw;
}

// Assertion macro that halts execution gracefully and signals a suite abort
#define ASSERT_TRUE(condition)                                                               \
    do                                                                                       \
    {                                                                                        \
        if (!(condition))                                                                    \
        {                                                                                    \
            g_TestFailed.store(1, std::memory_order_relaxed);                                \
            if (g_AbortRequested.exchange(1, std::memory_order_acq_rel) == 0)                \
            {                                                                                \
                Print("\n[FAILED] Assertion failed: %s at line %d\n", #condition, __LINE__); \
                std::this_thread::sleep_for(std::chrono::milliseconds(500));                 \
            }                                                                                \
            return;                                                                          \
        }                                                                                    \
    } while (false)

// ============================================================================
// Global Fatal Error Handler (User-Mode Only)
// ============================================================================
// Replaces std::terminate/std::abort with a safe abort signal that prevents 
// infinite loops while preserving test suite teardown visibility.
inline void AtomicPtrFatalHandler(const char* reason) noexcept
{
    g_TestFailed.store(1, std::memory_order_relaxed);
    if (g_AbortRequested.exchange(1, std::memory_order_acq_rel) == 0)
    {
        Print("\n[FATAL ERROR] AtomicSharedPtr panic: %s\n", reason);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ============================================================================
// Cache-Isolated Sharded Instance Tracker
// ============================================================================
// Eliminates artificial test-harness cache-line ping-ponging between writer
// threads constructing TrackedObject instances and background GC worker threads
// destroying retired batches. Each thread mutates its own cache-aligned shard,
// while load() sums all shards to verify exact zero-leak invariants at phase bounds.
class ShardedInstanceCounter
{
    static constexpr size_t SHARD_COUNT = 64; // Number of cache-isolated counter shards (must be a power of two)

    // Cache-line padded atomic counter shard preventing false sharing between threads
    struct alignas(CACHE_LINE_SIZE) CounterShard
    {
        std::atomic<int64_t> value{0}; // Net count (+allocations / -deallocations) recorded on this shard
    };

    CounterShard                        shards[SHARD_COUNT]{};      // Contiguous array of cache-isolated counter shards
    inline static std::atomic<uint32_t> next_shard_index{0};        // Monotonic ticket counter assigning threads to shards round-robin

    // Resolves a stable, collision-free shard index for the calling thread using a
    // constant-initialized POD thread_local that remains valid across late TLS teardown.
    [[nodiscard]] static FORCE_INLINE size_t get_thread_shard() noexcept
    {
        thread_local uint32_t tls_shard = 0xFFFFFFFFu;
        if (tls_shard == 0xFFFFFFFFu) [[unlikely]]
        {
            tls_shard = next_shard_index.fetch_add(1u, std::memory_order_relaxed) &
                        static_cast<uint32_t>(SHARD_COUNT - 1u);
        }

        return static_cast<size_t>(tls_shard);
    }

public:
    // Atomically adds delta to the calling thread's assigned counter shard
    FORCE_INLINE void fetch_add(int64_t           delta,
                                std::memory_order order = std::memory_order_seq_cst) noexcept
    {
        shards[get_thread_shard()].value.fetch_add(delta, order);
    }

    // Atomically subtracts delta from the calling thread's assigned counter shard
    FORCE_INLINE void fetch_sub(int64_t           delta,
                                std::memory_order order = std::memory_order_seq_cst) noexcept
    {
        shards[get_thread_shard()].value.fetch_sub(delta, order);
    }

    // Computes the aggregate live instance count by summing all shards
    [[nodiscard]] int64_t load(std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        int64_t total = 0;
        for (size_t i = 0; i < SHARD_COUNT; ++i)
        {
            total += shards[i].value.load(order);
        }

        return total;
    }
};

// Global sharded instance counter tracking all live TrackedObject allocations across the test suite
static ShardedInstanceCounter g_ActiveInstances{};

// ----------------------------------------------------------------------------
// Formats the test name, executes the test, and perfectly aligns the output
// ----------------------------------------------------------------------------
inline void RunTest(const char* testName, 
                    void      (*testFunc)())
{
    // If an abort is signaled (e.g., Ctrl-C), skip remaining tests
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0) 
    {
        return;
    }

    char   nameBuf[128];
    size_t i = 0;
    
    // Copy the test name into the buffer
    while (testName[i] != '\0' && i < 55)
    {
        nameBuf[i] = testName[i];
        i++;
    }
    
    // Add exact trailing dots
    nameBuf[i++] = '.';
    nameBuf[i++] = '.';
    nameBuf[i++] = '.';
    
    // Pad to exactly 58 characters to guarantee vertical alignment of throughput
    while (i < 58)
    {
        nameBuf[i++] = ' ';
    }
    
    nameBuf[i] = '\0';

    // Print exactly 11 characters of prefix followed by the padded name
    Print("[RUNNING]  %s", nameBuf);
    
    if (g_TestTeardownHook) 
    {
        g_TestTeardownHook();
    }
    
    ASSERT_TRUE(g_ActiveInstances.load(std::memory_order_acquire) == 0);
    
    testFunc();
    
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0) 
    {
        Print("\n[ABORTED]  %s\n", nameBuf);
        return;
    }

    if (g_TestTeardownHook) 
    {
        g_TestTeardownHook();
    }
    
    ASSERT_TRUE(g_ActiveInstances.load(std::memory_order_acquire) == 0);
    
    // The \r overwrites the 11-character prefix and drops down a line,
    // leaving the Phase string and Throughput perfectly intact.
    Print("\r[FINISHED] \n");
}

// ----------------------------------------------------------------------------
// Calculates and prints the fixed-width throughput string 
// ----------------------------------------------------------------------------
inline void PrintThroughput(uint64_t totalOps, 
                            uint64_t elapsedMs)
{
    if (elapsedMs == 0)
    {
        elapsedMs = 1;
    }
    
    double opsPerSec = (static_cast<double>(totalOps) * 1000.0) / static_cast<double>(elapsedMs);
    double mops      = opsPerSec / 1000000.0;
    
    Print(" [ %5.2f Million Ops/sec]", mops);
}

// ----------------------------------------------------------------------------
// Intercepts long blocking sleep times to enable instant teardown.
// Checks the abort signal every 50ms and returns early if requested.
// ----------------------------------------------------------------------------
inline void AbortableSleepMs(uint32_t totalMs)
{
    uint32_t waited = 0;

    while (waited < totalMs && g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        uint32_t sleepTime = (totalMs - waited > 50) ? 50 : (totalMs - waited);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        waited += sleepTime;
    }
}

// ============================================================================
// Hardware Timer Calibration & Scaling Utilities
// ============================================================================

// ----------------------------------------------------------------------------
// Calibrates the CPU's cycle counter against the OS wall-clock timer
// and measures the hardware cycle cost of a single HARDWARE_PAUSE()
// ----------------------------------------------------------------------------
inline void CalibrateHardwareTimer()
{
    uint64_t t0 = GetTimeNs();
    uint64_t c0 = GetHardwareCycles();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    uint64_t t1 = GetTimeNs();
    uint64_t c1 = GetHardwareCycles();
    
    if (t1 > t0)
    {
        g_CyclesPerUs = (static_cast<double>(c1 - c0) * 1000.0) / static_cast<double>(t1 - t0);
    }
    
    if (g_CyclesPerUs <= 0.0)
    {
        g_CyclesPerUs = 1.0;
    }

    // Calibrate average cycles per HARDWARE_PAUSE() over 1024 iterations
    // so hold/spin loops can burn exact cycle durations without polling __rdtsc().
    uint64_t p0 = GetHardwareCycles();
    for (int i = 0; i < 1024; ++i)
    {
        HARDWARE_PAUSE();
    }

    uint64_t p1 = GetHardwareCycles();

    g_CyclesPerPause = (p1 > p0) ? (static_cast<double>(p1 - p0) / 1024.0) : 16.0;
    if (g_CyclesPerPause <= 0.0)
    {
        g_CyclesPerPause = 1.0;
    }
}

// ----------------------------------------------------------------------------
// Burns the specified number of hardware cycles using calibrated steps
// without stalling the CPU front-end on microcoded __rdtsc() polling loops.
// Uses HARDWARE_PAUSE() matching the KM benchmark suite so SMT sibling threads
// (including background GC sweeps) are not starved on ALU multiplier ports.
// ----------------------------------------------------------------------------
FORCE_INLINE void BurnCycles(uint64_t targetCycles) noexcept
{
    if (targetCycles == 0)
    {
        return;
    }

    uint64_t start = GetHardwareCycles();

    // Pace cycle delay using HARDWARE_PAUSE() between invariant cycle counter
    // reads so SMT sibling cores retain full ALU execution bandwidth.
    while ((GetHardwareCycles() - start) < targetCycles &&
           g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        HARDWARE_PAUSE();
    }
}

// ----------------------------------------------------------------------------
// Partitions total threads into Readers and Writers so that the Read/Write
// ratio scales proportionally with CPU core count (O(N / log2(N))).
// Guarantees (outReaders + outWriters == totalThreads) with zero truncation
// loss while preserving multi-writer contention (outWriters >= 2 for N >= 8).
// Matches the KM suite's exact 75% Reader / 25% Writer standard split and
// 67% Reader / 33% Writer aggressive split (uHw readers + uHw/2 writers).
// ----------------------------------------------------------------------------
inline void CalculateCoreProportionalRWThreads(unsigned int totalThreads,
                                               int&         outReaders,
                                               int&         outWriters,
                                               bool         aggressiveWriteContention = false) noexcept
{
    if (totalThreads <= 2)
    {
        outWriters = 1;
        outReaders = (totalThreads > 1) ? static_cast<int>(totalThreads - 1) : 1;
        return;
    }

    // Standard R/W scaling: 25% writers / 75% readers (matching KM uHw / 4 split).
    int writers = (totalThreads / 4) > 1 ? static_cast<int>(totalThreads / 4) : 1;

    if (aggressiveWriteContention)
    {
        // Chaos and Replacement phases apply a 1/3 writer split (matching KM uHw readers + uHw/2 writers)
        // to stress writer-writer handoffs without starving reader cores.
        writers = (totalThreads / 3) > 1 ? static_cast<int>(totalThreads / 3) : 1;
        if (writers < 2)
        {
            writers = 2;
        }
    }

    if (writers >= static_cast<int>(totalThreads))
    {
        writers = static_cast<int>(totalThreads) - 1;
    }

    outWriters = writers;
    outReaders = static_cast<int>(totalThreads) - writers;
}

// ----------------------------------------------------------------------------
// Calculates the cool-down time dynamically using logarithmic scaling
// ----------------------------------------------------------------------------
inline uint64_t CalculateCooldownTimeMs(uint64_t valueMs)
{
    double valueT = static_cast<double>(valueMs) / 1000.0;
    
    if (valueT <= 0.0)
    {
        return 0;
    }

    double cooldownSec = std::ceil(1.0 + (1.5 * std::log2(valueT)));
    return static_cast<uint64_t>(cooldownSec) * 1000ULL;
}

// ============================================================================
// Global Test Tracked Payload
// ============================================================================
// Simulates the primary data structure being securely transferred between threads.
struct alignas(CACHE_LINE_SIZE) TrackedObject
{
    int data; // Payload value verified by reader threads
    
    // Constructs the tracked payload and increments the sharded active-instance counter
    explicit TrackedObject(int val) 
        : data(val)
    {
        g_ActiveInstances.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Destroys the tracked payload and decrements the sharded active-instance counter
    ~TrackedObject()
    {
        g_ActiveInstances.fetch_sub(1, std::memory_order_release);
    }

    // Allocates cache-line aligned storage to prevent false sharing across test payloads
    void* operator new(size_t size)
    {
        return ::operator new(size, std::align_val_t{CACHE_LINE_SIZE});
    }

    // Releases cache-line aligned storage back to the heap
    void operator delete(void* p) noexcept
    {
        if (p)
        {
            ::operator delete(p, std::align_val_t{CACHE_LINE_SIZE});
        }
    }
};

// ============================================================================
// General Test Context & Factory Anchor
// ============================================================================

// Primary template specialized per smart-pointer handle type (std::shared_ptr,
// LocalSharedPtr, PersistentSharedPtr) to provide uniform Make() and MakeNull() factories.
template <typename TLocalType>
struct HandleFactory 
{
    // Instantiated directly by the individual test harnesses 
};

// Structures for storing tabular benchmark output
struct LatencyResult
{
    uint64_t p50{0};   // 50th percentile (median) read latency in nanoseconds
    uint64_t p90{0};   // 90th percentile read latency in nanoseconds
    uint64_t p999{0};  // 99.9th percentile tail latency in nanoseconds
    uint64_t p9999{0}; // 99.99th percentile extreme tail latency in nanoseconds
};

// Common Context block shared among the various test threads
struct TestContext 
{
    std::latch*           StartLatch{nullptr};     // Barrier synchronizing simultaneous worker thread start
    std::atomic<int32_t>* KeepRunning{nullptr};    // Pointer to phase active flag (1 = run, 0 = stop)
    std::atomic<int32_t>* WarmupPhase{nullptr};    // Optional pointer to warmup flag (1 = warmup, 0 = timed window)
    std::atomic<int64_t>* TotalReads{nullptr};     // Accumulator for completed read (or combined scaling) operations
    std::atomic<int64_t>* TotalWrites{nullptr};    // Accumulator for completed write/exchange operations
    void*                 SharedWrapper{nullptr};  // Type-erased pointer to the shared anchor under test
    uint64_t*             LatencyBuffer{nullptr};  // Per-reader ring buffer recording cycle deltas
    size_t                SamplesRecorded{0};      // Count of latency samples captured by this reader
    int                   ThreadId{0};             // Logical worker thread index [0 .. totalThreads - 1]
    bool                  IsWriter{false};         // True if thread is assigned to mutate the shared anchor
    int                   ReadsPerWrite{0};        // Number of batched reads per write in scaling sweep
    uint64_t              HoldTimeCycles{0};       // Simulated critical-section hold duration in CPU cycles
};

// ----------------------------------------------------------------------------
// Safe Heap-Allocated Suite Tracker
// Dynamically provisions the thread and context structures on the heap.
// ----------------------------------------------------------------------------
struct TestHarnessState 
{
    std::vector<std::thread> threads;  // Heap-allocated vector of worker thread objects
    std::vector<TestContext> contexts; // Heap-allocated vector of per-thread execution contexts

    // Pre-sizes the worker thread and context vectors for up to maxThreads workers
    explicit TestHarnessState(size_t maxThreads = MAX_TEST_THREADS) : threads(maxThreads),
                                                                      contexts(maxThreads)
    {
    }

    // Joins any remaining joinable worker threads upon scope exit
    ~TestHarnessState()
    {
        for (auto& t : threads)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
};

// Stores throughput metrics (in Million Ops/sec) for a single thread-count step
// during the Phase 19 scaling sweep across 80/20, 99/1, and 99.9/0.1 read/write ratios.
struct ScalingPoint
{
    unsigned int threads{0};             // Number of concurrent worker threads for this scaling data point
    double       base_mops_80_20{0.0};   // Baseline throughput at 80% Read / 20% Write (Mops/sec)
    double       test_mops_80_20{0.0};   // AtomicSharedPtr throughput at 80% Read / 20% Write (Mops/sec)
    double       rundown_mops_80_20{0.0};// RundownSharedPtr throughput at 80% Read / 20% Write (Mops/sec)
    double       base_mops_99_1{0.0};    // Baseline throughput at 99% Read / 1% Write (Mops/sec)
    double       test_mops_99_1{0.0};    // AtomicSharedPtr throughput at 99% Read / 1% Write (Mops/sec)
    double       rundown_mops_99_1{0.0}; // RundownSharedPtr throughput at 99% Read / 1% Write (Mops/sec)
    double       base_mops_999_01{0.0};  // Baseline throughput at 99.9% Read / 0.1% Write (Mops/sec)
    double       test_mops_999_01{0.0};  // AtomicSharedPtr throughput at 99.9% Read / 0.1% Write (Mops/sec)
    double       rundown_mops_999_01{0.0}; // RundownSharedPtr throughput at 99.9% Read / 0.1% Write (Mops/sec)
};

// Global analytics storage populated during Phases 16-19 and printed at suite completion
static LatencyResult g_BaseLatency;                          // Recorded tail-latency percentiles for Baseline
static LatencyResult g_TestLatency;                          // Recorded tail-latency percentiles for AtomicSharedPtr
static LatencyResult g_RundownLatency;                       // Recorded tail-latency percentiles for RundownSharedPtr
static ScalingPoint  g_ScalingResults[MAX_SCALING_POINTS];   // Recorded throughput table rows across swept thread counts
static unsigned int  g_ScalingResultsCount = 0;              // Number of valid populated entries in g_ScalingResults

// ============================================================================
// Generic Throughput Test Runner
// ============================================================================
// Centralizes boilerplate thread spawning, synchronization, and cleanup.
template <typename TAtomicType, typename TLocalType>
void RunStandardThroughputPhase(int  totalThreads, 
                                int  numReaders, 
                                void (*workerFunc)(void*))
{
    if (totalThreads > MAX_TEST_THREADS) 
    {
        totalThreads = MAX_TEST_THREADS;
    }

    // Ensure any prior GC state is completely drained before starting the phase
    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    std::latch           startLatch(totalThreads + 1);
    std::atomic<int32_t> keepRunning{1};
    std::atomic<int64_t> totalReads{0};
    std::atomic<int64_t> totalWrites{0};

    // Pre-populate the anchor on the coordinator thread before spawning workers
    // so readers never observe an uninitialized (nullptr) anchor at startup.
    TAtomicType shared(HandleFactory<TLocalType>::Make(1));
    if constexpr (requires { shared.load(); })
    {
        TLocalType initialObj = shared.load();
        ASSERT_TRUE(initialObj && initialObj->data == 1);
    }
    else if constexpr (requires { shared.load_persistent(); })
    {
        TLocalType initialObj = shared.load_persistent();
        ASSERT_TRUE(initialObj && initialObj->data == 1);
    }

    TestHarnessState state(static_cast<size_t>(totalThreads));

    // Interleave writer and reader assignments evenly across thread creation order
    // so oversubscribed run queues schedule writers and readers symmetrically.
    const int numWriters      = totalThreads - numReaders;
    const int writerStride    = (numWriters > 0 && totalThreads >= numWriters)
                                ? (totalThreads / numWriters)
                                : 0;
    int       writersAssigned = 0;

    for (int i = 0; i < totalThreads; ++i)
    {
        bool assignWriter = false;
        if (writersAssigned < numWriters)
        {
            if (numReaders == 0 ||
                (writerStride > 0 && (i % writerStride) == 0) ||
                (totalThreads - i) <= (numWriters - writersAssigned))
            {
                assignWriter = true;
                writersAssigned++;
            }
        }

        state.contexts[i].StartLatch    = &startLatch;
        state.contexts[i].KeepRunning   = &keepRunning;
        state.contexts[i].TotalReads    = &totalReads;
        state.contexts[i].TotalWrites   = &totalWrites;
        state.contexts[i].SharedWrapper = &shared;
        state.contexts[i].ThreadId      = i;
        state.contexts[i].IsWriter      = assignWriter;
        
        state.threads[i] = std::thread(workerFunc, &state.contexts[i]);
    }

    startLatch.arrive_and_wait();
    
    // Capture time immediately after latch release. The 50ms scheduler 
    // settling time is now mathematically accounted for in the divisor,
    // and endTime is sampled immediately before signaling stop and joining.
    uint64_t startTime = GetTimeMs();
    
    AbortableSleepMs(50 + g_SoakDurationMs);
    uint64_t endTime = GetTimeMs();
    keepRunning.store(0, std::memory_order_relaxed);

    for (int i = 0; i < totalThreads; ++i) 
    {
        if (state.threads[i].joinable())
        {
            state.threads[i].join();
        }
    }
    
    PrintThroughput(
        static_cast<uint64_t>(totalReads.load(std::memory_order_relaxed) + totalWrites.load(std::memory_order_relaxed)),
        endTime - startTime);
    
    shared.store(HandleFactory<TLocalType>::MakeNull());

    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }
}

// ----------------------------------------------------------------------------
// Shared Helper for 1.5x Oversubscribed Reader/Writer Phases
// Sizes hw readers and hw/2 writers (clamped to MAX_TEST_THREADS) and invokes
// RunStandardThroughputPhase with the caller-supplied worker routine.
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void RunOnePointFiveOversubPhase(void (*workerFunc)(void*))
{
    unsigned int hw           = GetHardwareThreads();
    int          numWriters   = (hw / 2) > 1 ? static_cast<int>(hw / 2) : 1;
    int          numReaders   = hw > 2 ? static_cast<int>(hw) : 2;
    int          totalThreads = numReaders + numWriters;

    if (totalThreads > MAX_TEST_THREADS)
    {
        totalThreads = MAX_TEST_THREADS;
        numWriters   = totalThreads / 3;
        numReaders   = totalThreads - numWriters;
    }

    RunStandardThroughputPhase<TAtomicType, TLocalType>(totalThreads,
                                                        numReaders,
                                                        workerFunc);
}

// ============================================================================
// Phase 0: Local Handle Semantics
// ============================================================================
// Verifies that constructors, destructors, and moves properly govern the 
// underlying reference count mechanics without triggering use-after-free conditions.
template <typename TAtomicType, typename TLocalType>
void TestLocalHandleSemantics()
{
    // Verify default empty initialization invariants
    TLocalType empty_handle;
    ASSERT_TRUE(!empty_handle);
    ASSERT_TRUE(empty_handle.get() == nullptr);
    ASSERT_TRUE(empty_handle.use_count() == 0);

    TAtomicType empty_shared;
    TLocalType  empty_loaded = empty_shared.load();
    ASSERT_TRUE(!empty_loaded);
    ASSERT_TRUE(empty_loaded.get() == nullptr);
    ASSERT_TRUE(empty_loaded.use_count() == 0);

    TAtomicType shared(HandleFactory<TLocalType>::Make(100));

    TLocalType p1 = shared.load();
    ASSERT_TRUE(p1 && p1->data == 100);

    TLocalType p2(p1);
    ASSERT_TRUE(p2 && p2->data == 100);

    TLocalType p3; 
    p3 = p1;
    ASSERT_TRUE(p3 && p3->data == 100);

    TLocalType p4(std::move(p1));
    ASSERT_TRUE(p4 && p4->data == 100);
    ASSERT_TRUE(!p1);

    TLocalType p5; 
    p5 = std::move(p2);
    ASSERT_TRUE(p5 && p5->data == 100);
    ASSERT_TRUE(!p2);

    // Verify copy and move self-assignment safety without refcount drift
    TLocalType* p4_addr = &p4;
    p4 = *p4_addr;
    ASSERT_TRUE(p4 && p4->data == 100);

    p4 = std::move(*p4_addr);
    ASSERT_TRUE(p4 && p4->data == 100);

    // Verify explicit reset releases local ownership cleanly
    p3.reset();
    ASSERT_TRUE(!p3);
    ASSERT_TRUE(p3.use_count() == 0);

    ASSERT_TRUE(g_ActiveInstances.load(std::memory_order_acquire) == 1);

    shared.store(HandleFactory<TLocalType>::MakeNull());
}

// ============================================================================
// Phase 1: Extended Handle & API Parity
// ============================================================================
// Verifies use_count correctness, swap semantics, and object scope lifetimes 
// when the global anchor is modified concurrently.
template <typename TAtomicType, typename TLocalType>
void TestExtendedLocalHandleSemantics()
{
    TAtomicType atomic_ptr;
    
    TLocalType p1 = HandleFactory<TLocalType>::MakeNull();
    ASSERT_TRUE(!p1);
    
    TLocalType p2 = HandleFactory<TLocalType>::Make(100);
    TLocalType p3 = HandleFactory<TLocalType>::Make(200);
    p2.swap(p3);
    ASSERT_TRUE(p2->data == 200 && p3->data == 100);

    TLocalType p4 = HandleFactory<TLocalType>::Make(300);
    ASSERT_TRUE(p4.use_count() == 1);
    
    TLocalType p5 = p4;
    ASSERT_TRUE(p4.use_count() == 2);
    ASSERT_TRUE(p5.use_count() == 2);

    atomic_ptr.store(p4);
    
    // Regression verification: repeated load() + scope exit cycles while anchored
    // must never drive use_count() to zero or negative values via negative c.external.
    for (int i = 0; i < 64; ++i)
    {
        TLocalType transient_checkout = atomic_ptr.load();
        ASSERT_TRUE(transient_checkout && transient_checkout->data == 300);
        ASSERT_TRUE(transient_checkout.use_count() == 4);
    }

    TLocalType p6 = atomic_ptr.load();
    ASSERT_TRUE(p6 && p6->data == 300);
    ASSERT_TRUE(p4.use_count() == 4);
    ASSERT_TRUE(p6.use_count() == 4);
    
    // Scavenge the transient external counts held globally before asserting refs
    atomic_ptr.store(HandleFactory<TLocalType>::MakeNull());
    
    // Remaining refs: p4, p5, p6
    ASSERT_TRUE(p4.use_count() == 3);

    // Verify solo-anchored handle use_count stability across repeated checkouts
    atomic_ptr.store(HandleFactory<TLocalType>::Make(400));
    for (int i = 0; i < 64; ++i)
    {
        TLocalType solo_checkout = atomic_ptr.load();
        ASSERT_TRUE(solo_checkout && solo_checkout->data == 400);
        ASSERT_TRUE(solo_checkout.use_count() == 2);
    }
    atomic_ptr.store(HandleFactory<TLocalType>::MakeNull());
}

// ============================================================================
// Phase 2: Compare Exchange Semantics
// ============================================================================
// Strictly evaluates compare_exchange correctness by unleashing maximum 
// thread contention on a single shared index counter, alongside deterministic
// rollback, rvalue move-only-on-success, and spurious-failure immunity checks.

// ----------------------------------------------------------------------------
// Background reader worker that continuously calls load() to mutate the
// anchor's embedded external_count while a writer tests compare_exchange_strong
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void CasSpuriousReaderWorker(void* ctxRaw)
{
    TestContext* ctx    = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared = static_cast<TAtomicType*>(ctx->SharedWrapper);
    ctx->StartLatch->arrive_and_wait();

    while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
           g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        TLocalType local = shared->load();
        if (local)
        {
            int volatile v = local->data;
            (void)v;
        }
    }
}

// ----------------------------------------------------------------------------
// Deterministically verifies single-threaded failure rollback and ownership
// preservation for weak/strong and l-value/r-value compare_exchange overloads
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void VerifyCompareExchangeRollbackSemantics()
{
    // Verify lvalue and rvalue compare_exchange failure rollback & ownership preservation
    TAtomicType shared(HandleFactory<TLocalType>::Make(10));
    TLocalType  wrong_expected = HandleFactory<TLocalType>::Make(20);
    TLocalType  desired_lval   = HandleFactory<TLocalType>::Make(30);

    ASSERT_TRUE(desired_lval.use_count() == 1);

    // Lvalue compare_exchange_weak failure must update expected and leave desired_lval at refcount 1
    ASSERT_TRUE(!shared.compare_exchange_weak(wrong_expected, desired_lval));
    ASSERT_TRUE(wrong_expected && wrong_expected->data == 10);
    ASSERT_TRUE(desired_lval && desired_lval->data == 30);
    ASSERT_TRUE(desired_lval.use_count() == 1);

    // Lvalue compare_exchange_strong failure must update expected and leave desired_lval at refcount 1
    wrong_expected = HandleFactory<TLocalType>::Make(25);
    ASSERT_TRUE(!shared.compare_exchange_strong(wrong_expected, desired_lval));
    ASSERT_TRUE(wrong_expected && wrong_expected->data == 10);
    ASSERT_TRUE(desired_lval && desired_lval->data == 30);
    ASSERT_TRUE(desired_lval.use_count() == 1);

    // Rvalue compare_exchange_weak failure must NOT steal ownership of desired_rval
    TLocalType desired_rval = HandleFactory<TLocalType>::Make(40);
    wrong_expected          = HandleFactory<TLocalType>::Make(25);
    ASSERT_TRUE(!shared.compare_exchange_weak(wrong_expected, std::move(desired_rval)));
    ASSERT_TRUE(wrong_expected && wrong_expected->data == 10);
    ASSERT_TRUE(desired_rval && desired_rval->data == 40);
    ASSERT_TRUE(desired_rval.use_count() == 1);

    // Rvalue compare_exchange_strong failure must NOT steal ownership of desired_rval
    wrong_expected = HandleFactory<TLocalType>::Make(25);
    ASSERT_TRUE(!shared.compare_exchange_strong(wrong_expected, std::move(desired_rval)));
    ASSERT_TRUE(wrong_expected && wrong_expected->data == 10);
    ASSERT_TRUE(desired_rval && desired_rval->data == 40);
    ASSERT_TRUE(desired_rval.use_count() == 1);

    // Rvalue compare_exchange_strong success MUST consume desired_rval
    ASSERT_TRUE(shared.compare_exchange_strong(wrong_expected, std::move(desired_rval)));
    ASSERT_TRUE(!desired_rval);

    TLocalType verify_loaded = shared.load();
    ASSERT_TRUE(verify_loaded && verify_loaded->data == 40);

    // Verify mismatch against an empty (nullptr) anchor resets expected cleanly
    shared.store(HandleFactory<TLocalType>::MakeNull());
    ASSERT_TRUE(!shared.compare_exchange_weak(verify_loaded, desired_lval));
    ASSERT_TRUE(!verify_loaded);
    ASSERT_TRUE(desired_lval.use_count() == 1);

    verify_loaded = HandleFactory<TLocalType>::Make(99);
    ASSERT_TRUE(!shared.compare_exchange_strong(verify_loaded, desired_lval));
    ASSERT_TRUE(!verify_loaded);
    ASSERT_TRUE(desired_lval.use_count() == 1);
}

// ----------------------------------------------------------------------------
// Verifies that compare_exchange_strong never fails spuriously when a sole
// writer advances the anchor while N-1 readers hammer external_count via load()
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void VerifyCompareExchangeStrongUnderReaderContention()
{
    // Verify compare_exchange_strong immunity to spurious external_count contention:
    // Concurrent readers hammer shared.load() (mutating external_count), while a single
    // writer performs compare_exchange_strong with the known current expected handle.
    unsigned int hw         = GetHardwareThreads();
    int          numReaders = (hw > 1) ? static_cast<int>(hw - 1) : 1;
    if (numReaders > MAX_TEST_THREADS)
    {
        numReaders = MAX_TEST_THREADS;
    }

    std::latch           latch(numReaders + 1);
    std::atomic<int32_t> keepRunning{1};
    TAtomicType          contended_shared(HandleFactory<TLocalType>::Make(1));
    TestHarnessState     reader_state(static_cast<size_t>(numReaders));

    for (int i = 0; i < numReaders; ++i)
    {
        reader_state.contexts[i].SharedWrapper = &contended_shared;
        reader_state.contexts[i].KeepRunning   = &keepRunning;
        reader_state.contexts[i].StartLatch    = &latch;
        reader_state.threads[i] = std::thread(CasSpuriousReaderWorker<TAtomicType, TLocalType>, &reader_state.contexts[i]);
    }

    TLocalType expected_handle = contended_shared.load();
    latch.arrive_and_wait();

    for (int step = 2; step <= 500 && g_AbortRequested.load(std::memory_order_relaxed) == 0; ++step)
    {
        TLocalType next_handle = HandleFactory<TLocalType>::Make(step);
        if ((step & 1) == 0)
        {
            // Lvalue compare_exchange_strong must never fail spuriously due to readers
            bool ok = contended_shared.compare_exchange_strong(expected_handle, next_handle);
            ASSERT_TRUE(ok);
            expected_handle = std::move(next_handle);
        }
        else
        {
            // Rvalue compare_exchange_strong must never fail spuriously due to readers
            TLocalType next_copy = next_handle;
            bool ok = contended_shared.compare_exchange_strong(expected_handle, std::move(next_handle));
            ASSERT_TRUE(ok);
            ASSERT_TRUE(!next_handle);
            expected_handle = std::move(next_copy);
        }
    }

    keepRunning.store(0, std::memory_order_relaxed);
    for (int i = 0; i < numReaders; ++i)
    {
        if (reader_state.threads[i].joinable())
        {
            reader_state.threads[i].join();
        }
    }

    expected_handle.reset();
    contended_shared.store(HandleFactory<TLocalType>::MakeNull());
}

// ----------------------------------------------------------------------------
// Multi-writer CAS contention worker that races all hardware threads to advance
// the shared payload's integer counter from 0 to 10,000 across all 4 CAS overloads
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void CasWorker(void* ctxRaw)
{
    TestContext* ctx    = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared = static_cast<TAtomicType*>(ctx->SharedWrapper);
    ctx->StartLatch->arrive_and_wait();

    TLocalType current = shared->load();
    int        mode    = 0;
    
    while (current && current->data < 10000 && g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        int        target_val = current->data + 1;
        TLocalType next       = HandleFactory<TLocalType>::Make(target_val);
        int        op         = (mode++) & 3;
        
        if (op == 0)
        {
            if (shared->compare_exchange_weak(current, next))
            {
                current = std::move(next);
            }
        }
        else if (op == 1)
        {
            TLocalType next_copy = next;
            if (shared->compare_exchange_weak(current, std::move(next)))
            {
                ASSERT_TRUE(!next);
                current = std::move(next_copy);
            }
            else
            {
                ASSERT_TRUE(next && next->data == target_val);
            }
        }
        else if (op == 2)
        {
            if (shared->compare_exchange_strong(current, next))
            {
                current = std::move(next);
            }
        }
        else
        {
            TLocalType next_copy = next;
            if (shared->compare_exchange_strong(current, std::move(next)))
            {
                ASSERT_TRUE(!next);
                current = std::move(next_copy);
            }
            else
            {
                ASSERT_TRUE(next && next->data == target_val);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Phase 2 Coordinator: Executes CAS rollback checks, reader-contention immunity,
// and an all-core multi-writer CAS race advancing a counter to >= 10,000
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void TestCompareExchangeSemantics()
{
    VerifyCompareExchangeRollbackSemantics<TAtomicType, TLocalType>();
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0)
    {
        return;
    }
    ASSERT_TRUE(g_ActiveInstances.load(std::memory_order_acquire) == 0);

    VerifyCompareExchangeStrongUnderReaderContention<TAtomicType, TLocalType>();
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0)
    {
        return;
    }
    ASSERT_TRUE(g_ActiveInstances.load(std::memory_order_acquire) == 0);

    TAtomicType          shared(HandleFactory<TLocalType>::Make(0));
    unsigned int         hw = GetHardwareThreads();
    if (hw > static_cast<unsigned int>(MAX_TEST_THREADS))
    {
        hw = static_cast<unsigned int>(MAX_TEST_THREADS);
    }

    std::latch       latch(hw + 1);
    TestHarnessState state(hw);

    for (unsigned int i = 0; i < hw; ++i)
    {
        state.contexts[i].SharedWrapper = &shared;
        state.contexts[i].StartLatch    = &latch;
        
        state.threads[i] = std::thread(CasWorker<TAtomicType, TLocalType>, &state.contexts[i]);
    }

    latch.arrive_and_wait();
    
    for (unsigned int i = 0; i < hw; ++i) 
    {
        if (state.threads[i].joinable())
        {
            state.threads[i].join();
        }
    }

    if (g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        TLocalType final_ptr = shared.load();
        ASSERT_TRUE(final_ptr && final_ptr->data >= 10000);
    }
    
    shared.store(HandleFactory<TLocalType>::MakeNull());
}

// ============================================================================
// Phase 3, 4, 5: MPMC Benchmark (Atomic / Baseline)
// ============================================================================
// Evaluates general Multi-Producer/Multi-Consumer scaling. Threads load or 
// store completely unrestrained to saturate the atomic boundary.

// ----------------------------------------------------------------------------
// Worker routine for the MPMC benchmark where writers continuously publish new
// objects via store() and readers continuously acquire and read the payload
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void MPMCWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared    = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();

    if (ctx->IsWriter)
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            TLocalType new_obj = HandleFactory<TLocalType>::Make(ctx->ThreadId);
            if (new_obj) [[likely]]
            {
                shared->store(std::move(new_obj));
            }
            local_ops++;
        }
        
        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if constexpr (requires { shared->load(); })
            {
                TLocalType local = shared->load();
                
                if (local)
                {
                    int volatile read_val = local->data;
                    (void)read_val;
                }
            }
            else
            {
                shared->read([](TrackedObject* local)
                {
                    if (local)
                    {
                        int volatile read_val = local->data;
                        (void)read_val;
                    }
                });
            }
            
            local_ops++;
        }
        
        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Configures a 75% reader / 25% writer thread split and executes the MPMC phase
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void RunMPMCBenchmark(int oversubFactor = 1)
{
    unsigned int hw           = GetHardwareThreads() * oversubFactor;
    int          totalThreads = (hw > 3) ? static_cast<int>(hw) : 3;
    int          numReaders   = 0;
    int          numWriters   = 0;
    
    if (totalThreads > MAX_TEST_THREADS)
    {
        totalThreads = MAX_TEST_THREADS;
    }

    CalculateCoreProportionalRWThreads(static_cast<unsigned int>(totalThreads),
                                       numReaders,
                                       numWriters,
                                       false);

    RunStandardThroughputPhase<TAtomicType, TLocalType>(totalThreads, numReaders, MPMCWorker<TAtomicType, TLocalType>);
}

// ============================================================================
// Phase 6, 7, 8: Heavy CPU Oversubscription
// ============================================================================
// Spawns 6x the logical hardware thread count to force aggressive OS preemptions, 
// ensuring the implementation never deadlocks.

// ----------------------------------------------------------------------------
// Worker routine for the 6x CPU oversubscription test; writers alternate
// between store() and exchange() while yielding every 65,536 operations
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void OversubscriptionWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared    = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();
    
    if (ctx->IsWriter)
    {
        int counter = ctx->ThreadId * 10000 + 1;
        
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if ((local_ops & 1) == 0)
            {
                TLocalType new_obj = HandleFactory<TLocalType>::Make(++counter);
                if (new_obj) [[likely]]
                {
                    shared->store(std::move(new_obj));
                }
            }
            else
            {
                TLocalType new_obj = HandleFactory<TLocalType>::Make(++counter);
                if (new_obj) [[likely]]
                {
                    if constexpr (requires { shared->exchange(std::move(new_obj)); })
                    {
                        TLocalType old_val = shared->exchange(std::move(new_obj));
                        
                        if (old_val)
                        {
                            int volatile v = old_val->data;
                            (void)v;
                        }
                    }
                    else
                    {
                        shared->store(std::move(new_obj));
                    }
                }
            }
            
            local_ops++;
            if ((local_ops & 0xFFFF) == 0) [[unlikely]]
            {
                std::this_thread::yield();
            }
        }
        
        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if constexpr (requires { shared->load(); })
            {
                TLocalType local = shared->load();
                
                if (local)
                {
                    int volatile val = local->data;
                    (void)val;
                }
            }
            else
            {
                shared->read([](TrackedObject* local)
                {
                    if (local)
                    {
                        int volatile val = local->data;
                        (void)val;
                    }
                });
            }
            
            local_ops++;
            if ((local_ops & 0xFFFF) == 0) [[unlikely]]
            {
                std::this_thread::yield();
            }
        }
        
        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Spawns 6x hardware concurrency threads (capped at MAX_TEST_THREADS) to test
// lock-free progress and preemption resilience under heavy oversubscription
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void TestHeavyOversubscription()
{
    unsigned int hw           = GetHardwareThreads();
    int          totalThreads = hw * 6;
    int          numReaders   = 0;
    int          numWriters   = 0;
    
    if (totalThreads > MAX_TEST_THREADS) 
    {
        totalThreads = MAX_TEST_THREADS;
    }
    
    CalculateCoreProportionalRWThreads(static_cast<unsigned int>(totalThreads),
                                       numReaders,
                                       numWriters,
                                       false);

    if (numWriters < 4 && totalThreads > 8)
    {
        numWriters = 4;
        numReaders = totalThreads - numWriters;
    }

    RunStandardThroughputPhase<TAtomicType, TLocalType>(totalThreads, numReaders, OversubscriptionWorker<TAtomicType, TLocalType>);
}

// ============================================================================
// Phase 9, 10, 11: Concurrent Replacement (In Use Snapshotting)
// ============================================================================
// Critical Test: Readers hold the snapshot while writers concurrently mutate 
// the anchor. Validates the split-ref counter protects live payloads.

// ----------------------------------------------------------------------------
// Worker routine where readers verify payload immutability across a 25-step
// read accumulation loop while writers continuously displace the active anchor
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void ReplacementWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared    = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();
    
    if (ctx->IsWriter)
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            TLocalType new_obj = HandleFactory<TLocalType>::Make(ctx->ThreadId);
            if (new_obj) [[likely]]
            {
                if ((local_ops & 1) == 0) 
                {
                    shared->store(std::move(new_obj));
                }
                else 
                {
                    if constexpr (requires { shared->exchange(std::move(new_obj)); })
                    {
                        TLocalType old_ptr = shared->exchange(std::move(new_obj));
                        
                        if (old_ptr) 
                        { 
                            int volatile v = old_ptr->data; 
                            (void)v; 
                        }
                    }
                    else
                    {
                        shared->store(std::move(new_obj));
                    }
                }
            }
            
            local_ops++;
        }
        
        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if constexpr (requires { shared->load(); })
            {
                TLocalType local = shared->load();
                
                if (local)
                {
                    int initial_val = local->data;
                    int acc         = 0;
                    
                    for (int step = 0; step < 25; ++step) 
                    {
                        acc += local->data;
                    }
                    
                    (void)acc;
                    ASSERT_TRUE(local->data == initial_val);
                }
            }
            else
            {
                shared->read([](TrackedObject* local)
                {
                    if (local)
                    {
                        int initial_val = local->data;
                        int acc         = 0;
                        
                        for (int step = 0; step < 25; ++step) 
                        {
                            acc += local->data;
                        }
                        
                        (void)acc;
                        ASSERT_TRUE(local->data == initial_val);
                    }
                });
            }
            
            local_ops++;
        }
        
        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Runs the concurrent replacement soak phase at 1.5x thread oversubscription
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void TestConcurrentReplacementWhileInUse()
{
    RunOnePointFiveOversubPhase<TAtomicType, TLocalType>(ReplacementWorker<TAtomicType, TLocalType>);
}

// ============================================================================
// Phase 12: Heavy Exchange Contention
// ============================================================================
// Isolates the `exchange()` mechanics, forcing massive orphaned transient reads 
// that mandate continuous dynamic lock-free scavenging.

// ----------------------------------------------------------------------------
// Worker routine where every thread prepares a handle with reconciled external
// counts via a temporary anchor and swaps it into the global anchor via exchange()
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void HeavyExchangeWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared    = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();

    while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
           g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        TAtomicType temp(HandleFactory<TLocalType>::Make(ctx->ThreadId));
        TLocalType  my_new = temp.load();
        
        temp.store(HandleFactory<TLocalType>::MakeNull());

        if (my_new) [[likely]]
        {
            TLocalType old = shared->exchange(std::move(my_new));
            
            if (old)
            {
                int volatile val = old->data;
                (void)val;
            }
        }
        
        local_ops++;
    }
    
    ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------------
// Runs 100% concurrent exchange() contention across all logical hardware threads
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void TestHeavyExchange()
{
    unsigned int hw           = GetHardwareThreads();
    int          totalThreads = hw > 2 ? static_cast<int>(hw) : 2;
    
    if (totalThreads > MAX_TEST_THREADS)
    {
        totalThreads = MAX_TEST_THREADS;
    }

    // Passing totalThreads for numReaders explicitly prevents any threads from evaluating IsWriter to true
    RunStandardThroughputPhase<TAtomicType, TLocalType>(totalThreads, totalThreads, HeavyExchangeWorker<TAtomicType, TLocalType>);
}

// ============================================================================
// Phase 13, 14, 15: Hyper-Aggressive Chaos
// ============================================================================
// Forces the OS scheduler to unpredictably suspend executing threads during 
// sensitive instruction windows by yielding inside tight loops.

// ----------------------------------------------------------------------------
// Worker routine that injects OS scheduler yields every 50 writes and every
// 100 reads to trigger thread migrations across lock-free hazard windows
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void ChaosWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared    = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();

    if (ctx->IsWriter)
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if (local_ops % 50 == 0) 
            {
                std::this_thread::yield();
            }
            
            TLocalType new_obj = HandleFactory<TLocalType>::Make(ctx->ThreadId);
            if (new_obj) [[likely]]
            {
                shared->store(std::move(new_obj));
            }

            local_ops++;
        }

        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            if (local_ops % 100 == 0) 
            {
                std::this_thread::yield();
            }
            
            if constexpr (requires { shared->load(); })
            {
                TLocalType local = shared->load();
                
                if (local) 
                { 
                    int volatile read_val = local->data; 
                    (void)read_val; 
                }
            }
            else
            {
                shared->read([](TrackedObject* local)
                {
                    if (local)
                    {
                        int volatile read_val = local->data;
                        (void)read_val;
                    }
                });
            }
            
            local_ops++;
        }

        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Executes the hyper-aggressive scheduler yield chaos phase at 1.5x concurrency
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void TestHyperAggressiveChaos()
{
    RunOnePointFiveOversubPhase<TAtomicType, TLocalType>(ChaosWorker<TAtomicType, TLocalType>);
}

// ============================================================================
// Phase 16, 17, 18: Tail Latency (Atomic / Baseline)
// ============================================================================
// Exclusively tracks the worst-case delays experienced by Reader threads. 
// Uses pre-allocated Ring Buffers to bypass dynamic allocation overhead.

// ----------------------------------------------------------------------------
// Worker routine that records per-read hardware cycle deltas into a thread-local
// 1M-entry ring buffer while writers publish new payloads every ~3 microseconds
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void LatencyWorker(void* ctxRaw)
{
    TestContext* ctx    = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared = static_cast<TAtomicType*>(ctx->SharedWrapper);
    
    ctx->StartLatch->arrive_and_wait();

    if (ctx->IsWriter)
    {
        uint64_t spin_cycles = (ctx->HoldTimeCycles > 0) ? ctx->HoldTimeCycles : static_cast<uint64_t>(std::round(g_CyclesPerUs * 3.0));
        
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            TLocalType new_obj = HandleFactory<TLocalType>::Make(ctx->ThreadId);
            if (new_obj) [[likely]]
            {
                shared->store(std::move(new_obj));
            }

            BurnCycles(spin_cycles);
        }
    }
    else
    {
        size_t idx   = 0;
        size_t total = 0;
        
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            uint64_t start = GetHardwareCycles();
            std::atomic_signal_fence(std::memory_order_acq_rel);
            if constexpr (requires { shared->load(); })
            {
                TLocalType local = shared->load();
                
                if (local) 
                { 
                    int volatile v = local->data; 
                    (void)v; 
                }
            }
            else
            {
                shared->read([](TrackedObject* local)
                {
                    if (local)
                    {
                        int volatile v = local->data;
                        (void)v;
                    }
                });
            }

            std::atomic_signal_fence(std::memory_order_acq_rel);
            uint64_t end = GetHardwareCycles();

            ctx->LatencyBuffer[idx] = (end - start);
            idx                     = (idx + 1) & (LATENCY_RING_SIZE - 1);
            total++;
        }
        
        ctx->SamplesRecorded = std::min<size_t>(total, LATENCY_RING_SIZE);
    }
}

// ----------------------------------------------------------------------------
// Runs the tail-latency benchmark, consolidates and sorts all reader ring
// buffers, and computes P50, P90, P99.9, and P99.99 read latencies in ns
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void RunLatencyBenchmark(int targetType)
{
    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    AbortableSleepMs(static_cast<uint32_t>(g_CooldownMs));
    
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0) 
    {
        return;
    }

    unsigned int hw           = GetHardwareThreads();
    int          totalThreads = (hw > 3) ? static_cast<int>(hw) : 3;
    int          numReaders   = 0;
    int          numWriters   = 0;
    
    if (totalThreads > MAX_TEST_THREADS) 
    {
        totalThreads = MAX_TEST_THREADS;
    }

    CalculateCoreProportionalRWThreads(static_cast<unsigned int>(totalThreads),
                                       numReaders,
                                       numWriters,
                                       false);

    // Writers publish a new object every ~3 microseconds to match the KM latency benchmark.
    uint64_t writerHoldCycles = static_cast<uint64_t>(std::round(g_CyclesPerUs * 3.0));
    if (writerHoldCycles == 0)
    {
        writerHoldCycles = 1;
    }

    std::latch           startLatch(totalThreads + 1);
    std::atomic<int32_t> keepRunning{1};

    TAtomicType      shared(HandleFactory<TLocalType>::Make(0));
    TestHarnessState state(static_cast<size_t>(totalThreads));

    std::vector<uint64_t> globalBuffer;
    try
    {
        globalBuffer.resize(static_cast<size_t>(numReaders) * LATENCY_RING_SIZE);
    }
    catch (const std::bad_alloc&)
    {
        Print(" [OOM: Cannot allocate latency buffers]\n");
        shared.store(HandleFactory<TLocalType>::MakeNull());
        if (g_TestTeardownHook)
        {
            g_TestTeardownHook();
        }
        return;
    }

    for (int i = 0; i < totalThreads; ++i)
    {
        state.contexts[i].StartLatch     = &startLatch;
        state.contexts[i].KeepRunning    = &keepRunning;
        state.contexts[i].SharedWrapper  = &shared;
        state.contexts[i].IsWriter       = (i >= numReaders);
        state.contexts[i].ThreadId       = i;
        state.contexts[i].HoldTimeCycles = writerHoldCycles;
        
        if (!state.contexts[i].IsWriter) 
        {
            state.contexts[i].LatencyBuffer = globalBuffer.data() + (static_cast<size_t>(i) * LATENCY_RING_SIZE);
        }
        
        state.threads[i] = std::thread(LatencyWorker<TAtomicType, TLocalType>, &state.contexts[i]);
    }

    startLatch.arrive_and_wait();
    AbortableSleepMs(g_SoakDurationMs);
    keepRunning.store(0, std::memory_order_relaxed);

    for (int i = 0; i < totalThreads; ++i) 
    {
        if (state.threads[i].joinable())
        {
            state.threads[i].join();
        }
    }
    
    shared.store(HandleFactory<TLocalType>::MakeNull());

    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    if (g_AbortRequested.load(std::memory_order_relaxed) != 0)
    {
        return;
    }

    size_t totalSamples = 0;
    
    for (int i = 0; i < numReaders; ++i) 
    {
        totalSamples += state.contexts[i].SamplesRecorded;
    }

    if (totalSamples == 0)
    {
        return;
    }
    
    std::vector<uint64_t> flatBuffer;
    try
    {
        flatBuffer.reserve(totalSamples);
        for (int i = 0; i < numReaders; ++i)
        {
            const uint64_t* src   = state.contexts[i].LatencyBuffer;
            const size_t    count = state.contexts[i].SamplesRecorded;
            flatBuffer.insert(flatBuffer.end(), src, src + count);
        }
    }
    catch (const std::bad_alloc&)
    {
        return;
    }

    std::sort(flatBuffer.begin(), flatBuffer.end());
    
    // Nearest-rank percentile mapping rounding (P / 100) * (N - 1)
    size_t max_idx  = totalSamples - 1;
    size_t p50_idx  = static_cast<size_t>(std::round(0.5000 * static_cast<double>(max_idx)));
    size_t p90_idx  = static_cast<size_t>(std::round(0.9000 * static_cast<double>(max_idx)));
    size_t p99_idx  = static_cast<size_t>(std::round(0.9990 * static_cast<double>(max_idx)));
    size_t p999_idx = static_cast<size_t>(std::round(0.9999 * static_cast<double>(max_idx)));
    
    uint64_t p50_cycles  = flatBuffer[p50_idx];
    uint64_t p90_cycles  = flatBuffer[p90_idx];
    uint64_t p99_cycles  = flatBuffer[p99_idx];
    uint64_t p999_cycles = flatBuffer[p999_idx];
    
    uint64_t p50  = static_cast<uint64_t>(std::round((static_cast<double>(p50_cycles) * 1000.0) / g_CyclesPerUs));
    uint64_t p90  = static_cast<uint64_t>(std::round((static_cast<double>(p90_cycles) * 1000.0) / g_CyclesPerUs));
    uint64_t p99  = static_cast<uint64_t>(std::round((static_cast<double>(p99_cycles) * 1000.0) / g_CyclesPerUs));
    uint64_t p999 = static_cast<uint64_t>(std::round((static_cast<double>(p999_cycles) * 1000.0) / g_CyclesPerUs));
    
    if (targetType == 0)
    {
        g_BaseLatency.p50   = p50;
        g_BaseLatency.p90   = p90;
        g_BaseLatency.p999  = p99;
        g_BaseLatency.p9999 = p999;
    }
    else if (targetType == 1)
    {
        g_TestLatency.p50   = p50;
        g_TestLatency.p90   = p90;
        g_TestLatency.p999  = p99;
        g_TestLatency.p9999 = p999;
    }
    else if (targetType == 2)
    {
        g_RundownLatency.p50   = p50;
        g_RundownLatency.p90   = p90;
        g_RundownLatency.p999  = p99;
        g_RundownLatency.p9999 = p999;
    }
}

// ============================================================================
// Phase 19: Thread Scaling Sweep (Generic)
// ============================================================================
// Ramps thread saturation from 1 to Max, charting precise scaling behavior 
// without accumulating metrics via global atomics.

// ----------------------------------------------------------------------------
// Homogeneous scaling worker executing readsPerWrite reads followed by 1 write
// from a thread-local 4-element preallocated ring with a simulated hold delay
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
void ScalingWorker(void* ctxRaw)
{
    TestContext* ctx           = static_cast<TestContext*>(ctxRaw);
    TAtomicType* shared        = static_cast<TAtomicType*>(ctx->SharedWrapper);
    int          readsPerWrite = ctx->ReadsPerWrite;
    uint64_t     holdCycles    = ctx->HoldTimeCycles;

    // Pre-allocated per-thread to isolate pointer mechanics from malloc locks
    TLocalType preallocated[4] = 
    {
        HandleFactory<TLocalType>::Make(ctx->ThreadId * 10 + 1),
        HandleFactory<TLocalType>::Make(ctx->ThreadId * 10 + 2),
        HandleFactory<TLocalType>::Make(ctx->ThreadId * 10 + 3),
        HandleFactory<TLocalType>::Make(ctx->ThreadId * 10 + 4)
    };
    
    int     writeIdx  = 0;
    int64_t local_ops = 0;

    // Stagger each thread's initial read cycle offset using a coprime stride so threads
    // do not fire synchronized write waves every (readsPerWrite + 1) operations.
    int currentReadsTarget = (readsPerWrite > 0) ? ((ctx->ThreadId * 131) % (readsPerWrite + 1)) : 0;
    
    ctx->StartLatch->arrive_and_wait();

    while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
           g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        if constexpr (requires { shared->load(); })
        {
            for (int r = 0; r < currentReadsTarget; ++r)
            {
                if (ctx->KeepRunning->load(std::memory_order_relaxed) == 0 ||
                    g_AbortRequested.load(std::memory_order_relaxed) != 0) [[unlikely]]
                {
                    break;
                }

                // Lock-free checkout
                TLocalType local = shared->load();
                
                if (local)
                {
                    int volatile v = local->data;
                    (void)v;

                    // Emulate real-world payload processing hold time.
                    // Holding the reference stabilizes the internal counts and 
                    // spaces out the 128-bit CAS traffic on the memory bus.
                    if (holdCycles > 0)
                    {
                        BurnCycles(holdCycles);
                    }
                }

                // `local` drops out of scope here, relinquishing the transient external ref
                local_ops++;
            }
        }
        else
        {
            for (int r = 0; r < currentReadsTarget; ++r)
            {
                if (ctx->KeepRunning->load(std::memory_order_relaxed) == 0 ||
                    g_AbortRequested.load(std::memory_order_relaxed) != 0) [[unlikely]]
                {
                    break;
                }

                shared->read([holdCycles](TrackedObject* local)
                {
                    if (local)
                    {
                        int volatile v = local->data;
                        (void)v;

                        if (holdCycles > 0)
                        {
                            BurnCycles(holdCycles);
                        }
                    }
                });

                local_ops++;
            }
        }

        // Subsequent iterations execute the full readsPerWrite batch
        currentReadsTarget = readsPerWrite;

        if (ctx->KeepRunning->load(std::memory_order_relaxed) == 0 ||
            g_AbortRequested.load(std::memory_order_relaxed) != 0) [[unlikely]]
        {
            break;
        }

        // Execute the write operation using the pre-allocated handle ring (matching KM
        // StorePreallocated) and apply the identical simulated work delay (holdCycles)
        // so that 80/20 (20% writes), 99/1 (1% writes), and 99.9/0.1 (0.1% writes) have
        // symmetric per-op work weighting without measuring OS heap lock contention.
        if constexpr (requires { shared->load(); })
        {
            shared->store(preallocated[writeIdx]);
        }
        else
        {
            shared->store(std::move(preallocated[writeIdx]));
            preallocated[writeIdx] = HandleFactory<TLocalType>::Make(ctx->ThreadId * 10 + writeIdx + 1);
        }

        if (holdCycles > 0)
        {
            BurnCycles(holdCycles);
        }
        
        writeIdx = (writeIdx + 1) & 3;
        local_ops++;
    }

    ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------------
// Runs a single thread-count step (threadCount workers for 50 + stepDuration ms)
// with deterministic pre/post GC flushes and returns throughput in Mops/sec
// ----------------------------------------------------------------------------
template <typename TAtomicType, typename TLocalType>
double MeasureThroughput(unsigned int threadCount, 
                         uint32_t     stepDuration,
                         int          readsPerWrite,
                         uint64_t     holdTimeCycles)
{
    if (threadCount > static_cast<unsigned int>(MAX_TEST_THREADS))
    {
        threadCount = static_cast<unsigned int>(MAX_TEST_THREADS);
    }

    // Ensure GC queues are completely empty before starting this scaling step
    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    std::latch           startLatch(threadCount + 1);
    std::atomic<int32_t> keepRunning{1};
    std::atomic<int64_t> totalOps{0};
    uint64_t             startTime = 0;
    uint64_t             endTime   = 0;

    // Scope shared and state inside an inner block so all worker threads join
    // and destroy their stack-allocated preallocated[4] handles before
    // g_TestTeardownHook() drains the GC pipeline.
    {
        TAtomicType      shared(HandleFactory<TLocalType>::Make(0));
        TestHarnessState state(threadCount);

        for (unsigned int i = 0; i < threadCount; ++i)
        {
            state.contexts[i].StartLatch     = &startLatch;
            state.contexts[i].KeepRunning    = &keepRunning;
            state.contexts[i].TotalReads     = &totalOps;
            state.contexts[i].SharedWrapper  = &shared;
            state.contexts[i].ThreadId       = static_cast<int>(i);
            state.contexts[i].ReadsPerWrite  = readsPerWrite;
            state.contexts[i].HoldTimeCycles = holdTimeCycles;
            
            state.threads[i] = std::thread(ScalingWorker<TAtomicType, TLocalType>, &state.contexts[i]);
        }

        startLatch.arrive_and_wait();
        
        // Begin tracking time immediately after latch release and include the 50ms
        // scheduler settling window in the timed duration (matching KM MeasureThroughput)
        // so all operations and GC work are symmetrically accounted for.
        startTime = GetTimeMs();
        AbortableSleepMs(50 + stepDuration); 
        
        // Capture end time BEFORE signaling stop to eliminate thread teardown latency
        endTime = GetTimeMs();
        
        keepRunning.store(0, std::memory_order_relaxed);

        for (unsigned int i = 0; i < threadCount; ++i)
        {
            if (state.threads[i].joinable())
            {
                state.threads[i].join();
            }
        }
        
        shared.store(HandleFactory<TLocalType>::MakeNull());
    }

    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    if (g_AbortRequested.load(std::memory_order_relaxed) == 0 &&
        g_ActiveInstances.load(std::memory_order_acquire) != 0)
    {
        Print("\n[FAILED] GC list not empty after scaling step (%u threads): %lld active instances\n",
              threadCount,
              static_cast<long long>(g_ActiveInstances.load(std::memory_order_acquire)));

        g_TestFailed.store(1, std::memory_order_relaxed);
        g_AbortRequested.store(1, std::memory_order_relaxed);
    }

    uint64_t elapsedMs = endTime - startTime;
    
    if (elapsedMs == 0) 
    {
        elapsedMs = 1;
    }

    double opsPerSec = (static_cast<double>(totalOps.load(std::memory_order_relaxed)) * 1000.0) / static_cast<double>(elapsedMs);
    
    return opsPerSec / 1000000.0;
}