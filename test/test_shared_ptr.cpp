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

// ============================================================================
// File Purpose & Architecture Overview
// ============================================================================
// This file implements the cross-platform User-Mode host harness for the
// shared-pointer verification, stress-soak, tail-latency, and multi-core
// scaling benchmark suite defined in test_shared_ptr_common.h.
//
// Core Responsibilities:
// 1. Hardware & OS Environment Configuration:
//    Queries the processor brand string across x86/x64 (CPUID), Apple Silicon
//    (sysctlbyname), and Linux (/proc/cpuinfo), elevates process/thread
//    scheduling priority to minimize OS background jitter, and manages POSIX
//    terminal echo state and Ctrl-C / SIGTERM abort handlers.
//
// 2. Handle Factory Specializations:
//    Binds uniform HandleFactory<T> traits for the three evaluated user-mode
//    architectures:
//      - Baseline : std::atomic<std::shared_ptr<TrackedObject>>
//      - Atomic   : AtomicSharedPtr<TrackedObject> / LocalSharedPtr<TrackedObject>
//      - Rundown  : RundownSharedPtr<TrackedObject> / PersistentSharedPtr<TrackedObject>
//
// 3. Asynchronous Rundown Garbage Collector Worker Thread:
//    Runs a dedicated background GC worker thread woken either by a 2-second
//    periodic timeout or a 65,535-entry high-watermark callback (matching the
//    KM benchmark driver), while honoring g_PausePeriodicGc during Phase 19
//    scaling sweeps so inter-step synchronous flushes remain deterministic.
//
// 4. Rundown Contention, Tail-Latency & Amortized Batch Scaling Workers:
//    Implements execute-around reader/writer workers for MPMC, 6x CPU
//    oversubscription, concurrent replacement, and chaos phases, plus an
//    amortized 32-read BatchedRundownGuard scaling worker that mirrors the KM
//    CRundownBatchGuard::Checkpoint() design across 80/20, 99/1, and 99.9/0.1
//    read/write ratios.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <new>
#include <string>
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
#include <unistd.h>
#include <sys/resource.h>
#include <termios.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#endif

#include "test_shared_ptr.h"

// ----------------------------------------------------------------------------
// Returns CPU namestring for informational purposes
// ----------------------------------------------------------------------------
static std::string cpu_name()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    char brand[0x40];
    std::memset(brand, 0, sizeof(brand));

#if defined(_MSC_VER)
    int cpuInfo[4] = { 0 };
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = static_cast<unsigned int>(cpuInfo[0]);

    if (nExIds >= 0x80000004)
    {
        __cpuid(cpuInfo, 0x80000002);
        std::memcpy(brand, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000003);
        std::memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000004);
        std::memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    }
#else
    unsigned int eax;

    __asm__ __volatile__("cpuid" : "=a"(eax) : "a"(0x80000000) : "ebx", "ecx", "edx");

    if (eax >= 0x80000004)
    {
        unsigned int data[12];

        for (unsigned int i = 0; i < 3; ++i)
        {
            __asm__ __volatile__("cpuid"
                : "=a"(data[i * 4 + 0]),
                "=b"(data[i * 4 + 1]),
                "=c"(data[i * 4 + 2]),
                "=d"(data[i * 4 + 3])
                : "a"(0x80000002 + i));
        }

        std::memcpy(brand, data, sizeof(data));
    }
#endif
    return std::string(brand);

#elif defined(__APPLE__)
    char   buf[256];
    size_t size = sizeof(buf);

    if (sysctlbyname("machdep.cpu.brand_string", &buf, &size, nullptr, 0) == 0)
    {
        return std::string(buf);
    }

    size = sizeof(buf);
    if (sysctlbyname("hw.model", &buf, &size, nullptr, 0) == 0)
    {
        return std::string(buf);
    }

    return "Unknown CPU";

#elif defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string   line;

    while (std::getline(f, line))
    {
        if (line.find("model name") != std::string::npos ||
            line.find("Hardware") != std::string::npos)
        {
            std::size_t pos = line.find(':');
            if (pos != std::string::npos)
            {
                return line.substr(pos + 2);
            }
        }
    }

    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

// ----------------------------------------------------------------------------
// Elevates process and main coordinator thread scheduling priority to reduce
// OS background preemption jitter during latency and throughput measurements.
// ----------------------------------------------------------------------------
void set_process_high_priority()
{
#if defined(_WIN32) || defined(_WIN64)
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
    (void)setpriority(PRIO_PROCESS, 0, -10);
#endif
}

// ============================================================================
// Terminal State Management
// ============================================================================
// Disables canonical line buffering and input echo on POSIX terminals while
// the benchmark suite is running, restoring the original termios state on exit.
#if defined(_WIN32) || defined(_WIN64)
struct TerminalState
{
    TerminalState() = default;
    ~TerminalState() = default;
};
#else
struct TerminalState
{
    struct termios OriginalTermios{};
    bool           IsTty{false};

    TerminalState()
    {
        IsTty = isatty(STDIN_FILENO);        
        if (IsTty)
        {
            tcgetattr(STDIN_FILENO, &OriginalTermios);
            
            struct termios modifiedTermios = OriginalTermios;
            modifiedTermios.c_lflag       &= ~static_cast<tcflag_t>(ECHO | ICANON);
            
            tcsetattr(STDIN_FILENO, TCSANOW, &modifiedTermios);
        }
    }

    ~TerminalState()
    {
        if (IsTty)
        {
            tcsetattr(STDIN_FILENO, TCSANOW, &OriginalTermios);
        }
    }
};
#endif

// ============================================================================
// Signal Hooks for Terminal Interrupts
// ============================================================================
// Intercepts Ctrl-C / Ctrl-Break (Windows) and SIGINT / SIGTERM (POSIX) to
// set g_AbortRequested so active 50ms sleep slices unwind cleanly.
#if defined(_WIN32) || defined(_WIN64)
BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType)
{
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT)
    {
        g_AbortRequested.store(1, std::memory_order_relaxed);
        return TRUE; 
    }
    
    return FALSE;
}
#endif

void PosixSignalHandler(int signum)
{
    (void)signum;
    g_AbortRequested.store(1, std::memory_order_relaxed);
}

// ============================================================================
// Global Rundown Fatal Error Handler (User-Mode Only)
// ============================================================================
// Mirrors AtomicPtrFatalHandler in test_shared_ptr_common.h: logs the fatal
// RundownSharedPtr invariant failure reason, flags suite abort/failure state,
// and pauses briefly so diagnostic output flushes prior to process termination.
inline void RundownPtrFatalHandler(const char* reason) noexcept
{
    Print("\n[FATAL ERROR] RundownSharedPtr panic: %s\n", reason);
    g_TestFailed.store(1, std::memory_order_relaxed);
    g_AbortRequested.store(1, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

// ============================================================================
// Inject UM Factories
// ============================================================================
// Specializes HandleFactory<T> for each user-mode smart-pointer handle type
// so the templated phases in test_shared_ptr_common.h can construct and reset
// payloads uniformly.
template <> 
struct HandleFactory<std::shared_ptr<TrackedObject>> 
{
    static std::shared_ptr<TrackedObject> Make(int val) 
    { 
        return std::make_shared<TrackedObject>(val); 
    }
    
    static std::shared_ptr<TrackedObject> MakeNull() 
    { 
        return nullptr; 
    }
};

template <> 
struct HandleFactory<LocalSharedPtr<TrackedObject>> 
{
    static LocalSharedPtr<TrackedObject> Make(int val) 
    { 
        return MakeLocalShared<TrackedObject>(val); 
    }
    
    static LocalSharedPtr<TrackedObject> MakeNull() 
    { 
        return nullptr; 
    }
};

template <> 
struct HandleFactory<PersistentSharedPtr<TrackedObject>> 
{
    static PersistentSharedPtr<TrackedObject> Make(int val) 
    { 
        return MakeRundownShared<TrackedObject>(val); 
    }
    
    static PersistentSharedPtr<TrackedObject> MakeNull() 
    { 
        return nullptr; 
    }
};

// ============================================================================
// Rundown Garbage Collector Thread
// ============================================================================
// Coordinates asynchronous epoch advancement and batch reclamation on a
// dedicated background worker thread. Uses an atomic transition guard
// (g_GcWakePending) so writer threads invoking GCWakeCallback() only acquire
// g_GcMutex on the 0 -> 1 edge, matching the KM SynchronizationEvent design.
static std::mutex              g_GcMutex;
static std::condition_variable g_GcCondVar;
static std::atomic<int32_t>    g_GcWakePending{0};
static std::thread             g_GcThread;

// Notification callback injected into SharedPtrGarbageCollector wake_callback
void GCWakeCallback()
{
    if (g_GcWakePending.exchange(1, std::memory_order_acq_rel) == 0)
    {
        {
            std::lock_guard<std::mutex> lock(g_GcMutex);
        }

        g_GcCondVar.notify_one();
    }
}

// Dedicated background worker thread servicing high-watermark and periodic GC sweeps
void GCWorkerThread()
{
    while (g_AbortRequested.load(std::memory_order_relaxed) == 0)
    {
        bool wokeBySignal = false;
        {
            std::unique_lock<std::mutex> lock(g_GcMutex);
            wokeBySignal = g_GcCondVar.wait_for(
                lock,
                std::chrono::milliseconds(GC_PERIODIC_INTERVAL_MS),
                []()
                {
                    return g_AbortRequested.load(std::memory_order_relaxed) != 0 ||
                           g_GcWakePending.load(std::memory_order_acquire) != 0;
                });

            g_GcWakePending.store(0, std::memory_order_release);
        }

        if (g_AbortRequested.load(std::memory_order_relaxed) != 0)
        {
            break;
        }

        // During Phase 19 (TestThreadScaling), MeasureThroughput and MeasureRundownThroughput
        // deterministically flush GC between every 500ms step; skip periodic timeout wakes
        // so a random timer tick cannot fire mid-step (matching KM RunGcWorkerLoop).
        if (!wokeBySignal && g_PausePeriodicGc.load(std::memory_order_acquire) != 0)
        {
            continue;
        }

        SharedPtrGarbageCollector::get_instance().process_garbage_pipeline(false);
    }

    // Synchronous final flush on teardown
    SharedPtrGarbageCollector::get_instance().process_garbage_pipeline(true);
}

// ============================================================================
// Rundown Benchmark Workers & Suite
// ============================================================================

// ----------------------------------------------------------------------------
// Phase 5 Helper: RundownMPMCWorker
// Worker routine for the Rundown Multi-Producer Multi-Consumer (MPMC) phase.
// Writers continuously publish newly allocated payloads via store() while
// readers execute scoped execute-around lambdas via read().
// ----------------------------------------------------------------------------
void RundownMPMCWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    auto*        shared    = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();

    if (ctx->IsWriter)
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            shared->store(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId));
            local_ops++;
        }
        
        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            shared->read([](TrackedObject* obj) 
            {
                if (obj)
                {
                    int volatile read_val = obj->data;
                    (void)read_val;
                }
            });
            
            local_ops++;
        }
        
        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Phase 8 Helper: RundownOversubscriptionWorker
// Worker routine for the 6x CPU oversubscription stress test on RundownSharedPtr.
// Periodically yields every 65,536 operations to rotate oversubscribed threads
// across logical cores, matching OversubscriptionWorker in test_shared_ptr_common.h.
// ----------------------------------------------------------------------------
void RundownOversubscriptionWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    auto*        shared    = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();
    
    if (ctx->IsWriter)
    {
        int counter = ctx->ThreadId * 10000;
        
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            // Rundown operates purely on stores since execute-around guarantees stability
            shared->store(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(++counter));
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
            shared->read([](TrackedObject* obj) 
            {
                if (obj)
                {
                    int volatile val = obj->data;
                    (void)val;
                }
            });
            
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
// Phase 11 Helper: RundownReplacementWorker
// Verifies that in-flight execute-around read() scopes never observe premature
// payload reclamation or overwrite across a 25-step read loop while writers
// continuously displace the global anchor.
// ----------------------------------------------------------------------------
void RundownReplacementWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    auto*        shared    = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
    int64_t      local_ops = 0;
    
    ctx->StartLatch->arrive_and_wait();
    
    if (ctx->IsWriter)
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            shared->store(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId));
            local_ops++;
        }
        
        ctx->TotalWrites->fetch_add(local_ops, std::memory_order_relaxed);
    }
    else
    {
        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            shared->read([](TrackedObject* obj) 
            {
                if (obj)
                {
                    int initial_val = obj->data;
                    int acc         = 0;
                    
                    for (int step = 0; step < 25; ++step) 
                    {
                        acc += obj->data;
                    }
                    
                    (void)acc;
                    ASSERT_TRUE(obj->data == initial_val);
                }
            });
            
            local_ops++;
        }
        
        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Phase 15 Helper: RundownChaosWorker
// Injects frequent voluntary OS scheduler yields (every 50 writes and every
// 100 reads) under 1.5x thread oversubscription to stress preemption across
// epoch acquisition and SList retirement windows.
// ----------------------------------------------------------------------------
void RundownChaosWorker(void* ctxRaw)
{
    TestContext* ctx       = static_cast<TestContext*>(ctxRaw);
    auto*        shared    = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
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
            
            shared->store(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId));
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
            
            shared->read([](TrackedObject* obj) 
            {
                if (obj) 
                { 
                    int volatile read_val = obj->data; 
                    (void)read_val; 
                }
            });
            
            local_ops++;
        }

        ctx->TotalReads->fetch_add(local_ops, std::memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Phase 18 Helper: RundownLatencyWorker
// Captures per-read hardware cycle deltas into a 1M-entry ring buffer while
// writers publish a new payload every ~3 microseconds.
// ----------------------------------------------------------------------------
void RundownLatencyWorker(void* ctxRaw)
{
    TestContext* ctx    = static_cast<TestContext*>(ctxRaw);
    auto*        shared = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
    
    ctx->StartLatch->arrive_and_wait();
    
    if (ctx->IsWriter)
    {
        uint64_t spin_cycles = (ctx->HoldTimeCycles > 0) ? ctx->HoldTimeCycles : 
                                                           static_cast<uint64_t>(std::round(g_CyclesPerUs * 3.0));

        while (ctx->KeepRunning->load(std::memory_order_relaxed) != 0 &&
               g_AbortRequested.load(std::memory_order_relaxed) == 0)
        {
            shared->store(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId));
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
            shared->read([](TrackedObject* obj) 
            {
                if (obj) 
                { 
                    int volatile v = obj->data; 
                    (void)v; 
                }
            });

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
// Phase 18: RunRundownLatencyBenchmark
// Measures P50, P90, P99.9, and P99.99 read latencies in nanoseconds for
// RundownSharedPtr under the same 3us writer pacing as RunLatencyBenchmark.
// ----------------------------------------------------------------------------
void RunRundownLatencyBenchmark()
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

    // Writers publish a new object every ~3 microseconds to match the KM and UM latency benchmarks.
    uint64_t writerHoldCycles = static_cast<uint64_t>(std::round(g_CyclesPerUs * 3.0));
    if (writerHoldCycles == 0)
    {
        writerHoldCycles = 1;
    }

    std::latch           startLatch(totalThreads + 1);
    std::atomic<int32_t> keepRunning{1};

    RundownSharedPtr<TrackedObject> shared(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(0));
    TestHarnessState                state(static_cast<size_t>(totalThreads));

    std::vector<uint64_t> globalBuffer;
    try
    {
        globalBuffer.resize(static_cast<size_t>(numReaders) * LATENCY_RING_SIZE);
    }
    catch (const std::bad_alloc&)
    {
        Print(" [OOM: Cannot allocate latency buffers]\n");
        shared.store(HandleFactory<PersistentSharedPtr<TrackedObject>>::MakeNull());
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
        
        state.threads[i] = std::thread(RundownLatencyWorker, &state.contexts[i]);
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
    
    shared.store(HandleFactory<PersistentSharedPtr<TrackedObject>>::MakeNull());

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
    
    size_t max_idx  = totalSamples - 1;
    size_t p50_idx  = static_cast<size_t>(std::round(0.5000 * static_cast<double>(max_idx)));
    size_t p90_idx  = static_cast<size_t>(std::round(0.9000 * static_cast<double>(max_idx)));
    size_t p99_idx  = static_cast<size_t>(std::round(0.9990 * static_cast<double>(max_idx)));
    size_t p999_idx = static_cast<size_t>(std::round(0.9999 * static_cast<double>(max_idx)));
    
    uint64_t p50_cycles  = flatBuffer[p50_idx];
    uint64_t p90_cycles  = flatBuffer[p90_idx];
    uint64_t p99_cycles  = flatBuffer[p99_idx];
    uint64_t p999_cycles = flatBuffer[p999_idx];
    
    g_RundownLatency.p50   = static_cast<uint64_t>(std::round((static_cast<double>(p50_cycles) * 1000.0) / g_CyclesPerUs));
    g_RundownLatency.p90   = static_cast<uint64_t>(std::round((static_cast<double>(p90_cycles) * 1000.0) / g_CyclesPerUs));
    g_RundownLatency.p999  = static_cast<uint64_t>(std::round((static_cast<double>(p99_cycles) * 1000.0) / g_CyclesPerUs));
    g_RundownLatency.p9999 = static_cast<uint64_t>(std::round((static_cast<double>(p999_cycles) * 1000.0) / g_CyclesPerUs));
}

// ----------------------------------------------------------------------------
// Phase 19 Helper: RundownScalingWorker
// Executes interleaved read/write batches for the Phase 19 scaling sweep.
// Amortizes epoch rundown slot acquisition across 32-read chunks via
// acquire_batched() (matching KM's CRundownBatchGuard::Checkpoint() every 32
// reads) so 99/1 and 99.9/0.1 read bursts execute with zero atomic instructions
// per inner read while releasing the epoch guard every 32 reads so background
// GC sweeps never stall.
// ----------------------------------------------------------------------------
void RundownScalingWorker(void* ctxRaw)
{
    constexpr int RUNDOWN_BATCH_CHECKPOINT = 32;

    TestContext* ctx           = static_cast<TestContext*>(ctxRaw);
    auto*        shared        = static_cast<RundownSharedPtr<TrackedObject>*>(ctx->SharedWrapper);
    int          readsPerWrite = ctx->ReadsPerWrite;
    uint64_t     holdCycles    = ctx->HoldTimeCycles;

    PersistentSharedPtr<TrackedObject> preallocated[4] = 
    {
        HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId * 10 + 1),
        HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId * 10 + 2),
        HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId * 10 + 3),
        HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId * 10 + 4)
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
        int r = 0;
        while (r < currentReadsTarget)
        {
            if (ctx->KeepRunning->load(std::memory_order_relaxed) == 0 ||
                g_AbortRequested.load(std::memory_order_relaxed) != 0) [[unlikely]]
            {
                break;
            }

            const int chunkEnd = std::min<int>(currentReadsTarget, r + RUNDOWN_BATCH_CHECKPOINT);

            // Pin the active epoch slot once for up to 32 consecutive reads,
            // matching KM's CRundownBatchGuard + 32-read Checkpoint() cadence.
            BatchedRundownGuard<TrackedObject> batchGuard = shared->acquire_batched();

            for (; r < chunkEnd; ++r)
            {
                if (ctx->KeepRunning->load(std::memory_order_relaxed) == 0 ||
                    g_AbortRequested.load(std::memory_order_relaxed) != 0) [[unlikely]]
                {
                    break;
                }

                batchGuard.read([holdCycles](TrackedObject* obj) 
                {
                    if (obj)
                    {
                        int volatile v = obj->data;
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

        shared->store(std::move(preallocated[writeIdx]));
        preallocated[writeIdx] = HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(ctx->ThreadId * 10 + writeIdx + 1);
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
// Phase 19 Helper: MeasureRundownThroughput
// Runs a single thread-count step for RundownSharedPtr over (50 + stepDuration)
// milliseconds with deterministic pre-step and post-step GC flushes.
// ----------------------------------------------------------------------------
double MeasureRundownThroughput(unsigned int threadCount, 
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
        RundownSharedPtr<TrackedObject> shared(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(0));
        TestHarnessState                state(threadCount);

        for (unsigned int i = 0; i < threadCount; ++i)
        {
            state.contexts[i].StartLatch     = &startLatch;
            state.contexts[i].KeepRunning    = &keepRunning;
            state.contexts[i].TotalReads     = &totalOps;
            state.contexts[i].SharedWrapper  = &shared;
            state.contexts[i].ThreadId       = static_cast<int>(i);
            state.contexts[i].ReadsPerWrite  = readsPerWrite;
            state.contexts[i].HoldTimeCycles = holdTimeCycles;
            
            state.threads[i] = std::thread(RundownScalingWorker, &state.contexts[i]);
        }

        startLatch.arrive_and_wait();
        
        // Begin tracking time immediately after latch release and include the 50ms
        // scheduler settling window in the timed duration (matching KM MeasureThroughput).
        startTime = GetTimeMs();
        AbortableSleepMs(50 + stepDuration); 
        endTime = GetTimeMs();
        keepRunning.store(0, std::memory_order_relaxed);

        for (unsigned int i = 0; i < threadCount; ++i)
        {
            if (state.threads[i].joinable())
            {
                state.threads[i].join();
            }
        }
        
        shared.store(HandleFactory<PersistentSharedPtr<TrackedObject>>::MakeNull());
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

// ============================================================================
// Suite Entry Coordinator & Table Printer
// ============================================================================

// ----------------------------------------------------------------------------
// Phase 19: TestThreadScaling
// Sweeps worker thread counts from 1 up to GetHardwareThreads() across 80/20,
// 99/1, and 99.9/0.1 read/write ratios while suppressing periodic background
// GC timer ticks (g_PausePeriodicGc = 1) and deterministically flushing GC
// between steps.
// ----------------------------------------------------------------------------
template <typename TBaseAtomic, typename TBaseLocal, typename TTestAtomic, typename TTestLocal>
void TestThreadScaling()
{
    // Enforce at least a 5-second inter-pass thermal/GC cooldown (even when a short
    // CLI soak duration is supplied) and a 100ms inter-measurement settling pause
    // so OS thread teardown and CPU turbo governors stabilize between steps.
    const uint32_t     pass_cooldown_ms = static_cast<uint32_t>(std::max<uint64_t>(g_CooldownMs, 5000ULL));
    constexpr uint32_t step_settle_ms   = 100;

    AbortableSleepMs(pass_cooldown_ms);
    
    if (g_AbortRequested.load(std::memory_order_relaxed) != 0) 
    {
        return;
    }

    if (g_TestTeardownHook)
    {
        g_TestTeardownHook();
    }

    // Suppress the periodic background GC timer during Phase 19 so a random timer
    // tick cannot fire mid-step while MeasureThroughput / MeasureRundownThroughput
    // deterministically flushes GC between every step (matching KM g_lPausePeriodicGc).
    g_PausePeriodicGc.store(1, std::memory_order_release);

    unsigned int hw            = GetHardwareThreads();
    uint32_t     step_duration = 500;
    
    if (hw > static_cast<unsigned int>(MAX_TEST_THREADS))
    {
        hw = static_cast<unsigned int>(MAX_TEST_THREADS);
    }

    // Calculate ~100 nanoseconds of hardware cycle burn.
    // E.g., on a 4.0 GHz CPU (4000 cycles/us), this results in 400 cycles.
    uint64_t hold_cycles = static_cast<uint64_t>(std::round(g_CyclesPerUs * 0.1));
    
    unsigned int thread_counts[MAX_SCALING_POINTS];
    unsigned int tc_idx = 0;

    if (hw <= 16)
    {
        for (unsigned int i = 1; i <= hw; ++i)
        {
            thread_counts[tc_idx++] = i;
        }
    }
    else
    {
        thread_counts[tc_idx++] = 1;
        thread_counts[tc_idx++] = 2;
        thread_counts[tc_idx++] = 4;
        thread_counts[tc_idx++] = 8;
        thread_counts[tc_idx++] = 12;
        thread_counts[tc_idx++] = 16;
        
        for (unsigned int i = 24; i <= hw && tc_idx < (MAX_SCALING_POINTS - 1); i += 8)
        {
            thread_counts[tc_idx++] = i;
        }

        if (thread_counts[tc_idx - 1] != hw && tc_idx < MAX_SCALING_POINTS)
        {
            thread_counts[tc_idx++] = hw;
        }
    }

    g_ScalingResultsCount = tc_idx;

    // Pass 1: 80% Read / 20% Write (4 reads per 1 write)
    for (unsigned int i = 0; i < tc_idx && g_AbortRequested.load(std::memory_order_relaxed) == 0; ++i)
    {
        unsigned int tc = thread_counts[i];
        g_ScalingResults[i].threads            = tc;
        g_ScalingResults[i].base_mops_80_20    = MeasureThroughput<TBaseAtomic, TBaseLocal>(tc, step_duration, SCALING_READS_PER_WRITE_80_20, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].test_mops_80_20    = MeasureThroughput<TTestAtomic, TTestLocal>(tc, step_duration, SCALING_READS_PER_WRITE_80_20, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].rundown_mops_80_20 = MeasureRundownThroughput(tc, step_duration, SCALING_READS_PER_WRITE_80_20, hold_cycles);
        AbortableSleepMs(step_settle_ms);
    }

    AbortableSleepMs(pass_cooldown_ms);

    // Pass 2: 99% Read / 1% Write (99 reads per 1 write)
    for (unsigned int i = 0; i < tc_idx && g_AbortRequested.load(std::memory_order_relaxed) == 0; ++i)
    {
        unsigned int tc = thread_counts[i];
        g_ScalingResults[i].base_mops_99_1    = MeasureThroughput<TBaseAtomic, TBaseLocal>(tc, step_duration, SCALING_READS_PER_WRITE_99_1, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].test_mops_99_1    = MeasureThroughput<TTestAtomic, TTestLocal>(tc, step_duration, SCALING_READS_PER_WRITE_99_1, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].rundown_mops_99_1 = MeasureRundownThroughput(tc, step_duration, SCALING_READS_PER_WRITE_99_1, hold_cycles);
        AbortableSleepMs(step_settle_ms);
    }

    AbortableSleepMs(pass_cooldown_ms);

    // Pass 3: 99.9% Read / 0.1% Write (999 reads per 1 write)
    for (unsigned int i = 0; i < tc_idx && g_AbortRequested.load(std::memory_order_relaxed) == 0; ++i)
    {
        unsigned int tc = thread_counts[i];
        g_ScalingResults[i].base_mops_999_01    = MeasureThroughput<TBaseAtomic, TBaseLocal>(tc, step_duration, SCALING_READS_PER_WRITE_999_01, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].test_mops_999_01    = MeasureThroughput<TTestAtomic, TTestLocal>(tc, step_duration, SCALING_READS_PER_WRITE_999_01, hold_cycles);
        AbortableSleepMs(step_settle_ms);
        g_ScalingResults[i].rundown_mops_999_01 = MeasureRundownThroughput(tc, step_duration, SCALING_READS_PER_WRITE_999_01, hold_cycles);
        AbortableSleepMs(step_settle_ms);
    }

    g_PausePeriodicGc.store(0, std::memory_order_release);
}

// ----------------------------------------------------------------------------
// Formats and prints a single read/write ratio scaling table (80/20, 99/1, or 99.9/0.1)
// ----------------------------------------------------------------------------
void PrintScalingTable(const char* title, 
                       int         ratioMode)
{
    Print("\n===========================================================================================\n");
    Print(" %s\n", title);
    Print("===========================================================================================\n");
    Print("Threads      Baseline (Base)    Atomic (LockFree)    Rundown (Lambda)     \n");
    Print("-------------------------------------------------------------------------------------------\n");

    for (unsigned int i = 0; i < g_ScalingResultsCount; ++i)
    {
        double base_mops    = 0.0;
        double test_mops    = 0.0;
        double rundown_mops = 0.0;

        if (ratioMode == 0)
        {
            base_mops    = g_ScalingResults[i].base_mops_80_20;
            test_mops    = g_ScalingResults[i].test_mops_80_20;
            rundown_mops = g_ScalingResults[i].rundown_mops_80_20;
        }
        else if (ratioMode == 1)
        {
            base_mops    = g_ScalingResults[i].base_mops_99_1;
            test_mops    = g_ScalingResults[i].test_mops_99_1;
            rundown_mops = g_ScalingResults[i].rundown_mops_99_1;
        }
        else
        {
            base_mops    = g_ScalingResults[i].base_mops_999_01;
            test_mops    = g_ScalingResults[i].test_mops_999_01;
            rundown_mops = g_ScalingResults[i].rundown_mops_999_01;
        }

        Print("%-10u %18.2f %23.2f %23.2f\n", g_ScalingResults[i].threads, base_mops, test_mops, rundown_mops);
    }
}

// ----------------------------------------------------------------------------
// Prints all three thread-scaling tables and the tail-latency percentile table
// ----------------------------------------------------------------------------
void PrintResultsTables(const char* baseName, 
                        const char* testName)
{
    (void)baseName;
    (void)testName;

    PrintScalingTable("Thread Scaling Results (80% Read / 20% Write - Million Ops/sec)", 0);
    PrintScalingTable("Thread Scaling Results (99% Read / 1% Write - Million Ops/sec)", 1);
    PrintScalingTable("Thread Scaling Results (99.9% Read / 0.1% Write - Million Ops/sec)", 2);

    Print("\n===========================================================================================\n");
    Print(" Tail Latency Results (nanoseconds)\n");
    Print("===========================================================================================\n");
    Print("Implementation                 P50      P90      P99.9      P99.99\n");
    Print("-------------------------------------------------------------------------------------------\n");
    
    Print("%-28s %6" PRIu64 "   %6" PRIu64 "    %6" PRIu64 "     %6" PRIu64 "\n", 
          "Baseline", g_BaseLatency.p50, g_BaseLatency.p90, g_BaseLatency.p999, g_BaseLatency.p9999);
                  
    Print("%-28s %6" PRIu64 "   %6" PRIu64 "    %6" PRIu64 "     %6" PRIu64 "\n", 
          "Atomic", g_TestLatency.p50, g_TestLatency.p90, g_BaseLatency.p999 ? g_TestLatency.p999 : g_TestLatency.p999, g_TestLatency.p9999);

    Print("%-28s %6" PRIu64 "   %6" PRIu64 "    %6" PRIu64 "     %6" PRIu64 "\n", 
          "Rundown", g_RundownLatency.p50, g_RundownLatency.p90, g_RundownLatency.p999, g_RundownLatency.p9999);
                  
    Print("===========================================================================================\n");
}

// ----------------------------------------------------------------------------
// RunAllPhases (Master Test Suite Coordinator)
// Calibrates the hardware cycle timer, computes the inter-phase cooldown
// interval, and sequentially executes all verification, stress-soak,
// tail-latency, and thread-scaling phases with strict zero-leak assertions.
// ----------------------------------------------------------------------------
template <typename TBaseAtomic, typename TBaseLocal, typename TTestAtomic, typename TTestLocal>
void RunAllPhases(const char* baseName, 
                  const char* testName)
{
    SetAtomicPtrErrorHandler(AtomicPtrFatalHandler);
    SetRundownPtrErrorHandler(RundownPtrFatalHandler);

    CalibrateHardwareTimer();

    g_CooldownMs = CalculateCooldownTimeMs(g_SoakDurationMs);

    std::atomic<std::shared_ptr<TrackedObject>> dummy_shared_atomic;

    Print("\n===========================================================\n");    
    Print(" Detected Hardware Concurrency : %u logical threads\n", GetHardwareThreads());
    Print(" Soak Duration Per Phase       : %u seconds\n", g_SoakDurationMs / 1000);
    Print(" std::shared_ptr is_lock_free(): %s\n", dummy_shared_atomic.is_lock_free() ? "true" : "false");
    Print("===========================================================\n\n");

    RunTest("Phase 0: Local Handle Semantics (Atomic)", TestLocalHandleSemantics<TTestAtomic, TTestLocal>);
    RunTest("Phase 1: Extended Handle API Parity (Atomic)", TestExtendedLocalHandleSemantics<TTestAtomic, TTestLocal>);
    RunTest("Phase 2: Compare & Exchange Correctness (Atomic)", TestCompareExchangeSemantics<TTestAtomic, TTestLocal>);
    
    RunTest("Phase 2b: Execute-Around Correctness (Rundown)", []() 
    {
        // Verify custom Rundown fatal error handler registration and restoration
        ASSERT_TRUE(g_rundown_fatal_error_handler.load(std::memory_order_acquire) == RundownPtrFatalHandler);
        SetRundownPtrErrorHandler(nullptr);
        ASSERT_TRUE(g_rundown_fatal_error_handler.load(std::memory_order_acquire) == nullptr);
        SetRundownPtrErrorHandler(RundownPtrFatalHandler);
        ASSERT_TRUE(g_rundown_fatal_error_handler.load(std::memory_order_acquire) == RundownPtrFatalHandler);

        RundownSharedPtr<TrackedObject> shared(HandleFactory<PersistentSharedPtr<TrackedObject>>::Make(100));
        shared.read([](TrackedObject* obj) 
        {
            ASSERT_TRUE(obj != nullptr);
            ASSERT_TRUE(obj->data == 100);
        });

        PersistentSharedPtr<TrackedObject> persistent = shared.load_persistent();
        ASSERT_TRUE(persistent && persistent->data == 100);
        shared.store(HandleFactory<PersistentSharedPtr<TrackedObject>>::MakeNull());
    });

    Print("\n--- Baseline Contention (%s) ---\n", baseName);
    
    RunTest("Phase 3: MPMC Benchmark (Baseline)", []() 
    { 
        RunMPMCBenchmark<TBaseAtomic, TBaseLocal>(); 
    });
    
    RunTest("Phase 4: MPMC Benchmark (Atomic)", []() 
    { 
        RunMPMCBenchmark<TTestAtomic, TTestLocal>(); 
    });

    RunTest("Phase 5: MPMC Benchmark (Rundown)", []() 
    {
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

        RunStandardThroughputPhase<RundownSharedPtr<TrackedObject>, 
                                   PersistentSharedPtr<TrackedObject>>(totalThreads, numReaders, RundownMPMCWorker);
    });

    Print("\n--- Extreme Oversubscription Performance ------------------\n");
    
    RunTest("Phase 6: Heavy CPU Oversub (Baseline)", []() 
    { 
        TestHeavyOversubscription<TBaseAtomic, TBaseLocal>();
    });
    
    RunTest("Phase 7: Heavy CPU Oversub (Atomic)", []() 
    { 
        TestHeavyOversubscription<TTestAtomic, TTestLocal>(); 
    });

    RunTest("Phase 8: Heavy CPU Oversub (Rundown)", []() 
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

        RunStandardThroughputPhase<RundownSharedPtr<TrackedObject>, PersistentSharedPtr<TrackedObject>>(
            totalThreads, numReaders, RundownOversubscriptionWorker);
    });

    Print("\n--- Correctness & Chaos Soak Path -------------------------\n");
    
    RunTest("Phase 9: Concurrent Replacement (Baseline)", TestConcurrentReplacementWhileInUse<TBaseAtomic, TBaseLocal>);
    RunTest("Phase 10: Concurrent Replacement (Atomic)", TestConcurrentReplacementWhileInUse<TTestAtomic, TTestLocal>);
    
    RunTest("Phase 11: Concurrent Replacement (Rundown)", []() 
    {
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
                                           true);

        RunStandardThroughputPhase<RundownSharedPtr<TrackedObject>, 
                                   PersistentSharedPtr<TrackedObject>>(totalThreads, numReaders, RundownReplacementWorker);
    });
    
    RunTest("Phase 12: Heavy Exchange Contention (Atomic)", TestHeavyExchange<TTestAtomic, TTestLocal>);
    
    RunTest("Phase 13: Hyper-Aggressive Chaos (Baseline)", TestHyperAggressiveChaos<TBaseAtomic, TBaseLocal>);
    RunTest("Phase 14: Hyper-Aggressive Chaos (Atomic)", TestHyperAggressiveChaos<TTestAtomic, TTestLocal>);
    
    RunTest("Phase 15: Hyper-Aggressive Chaos (Rundown)", []() 
    {
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
                                           true);

        RunStandardThroughputPhase<RundownSharedPtr<TrackedObject>, 
                                   PersistentSharedPtr<TrackedObject>>(totalThreads, numReaders, RundownChaosWorker);
    });

    Print("\n--- Contention Metrics & Analytics ------------------------\n");
    
    RunTest("Phase 16: Tail Latency (Baseline)", []() 
    { 
        RunLatencyBenchmark<TBaseAtomic, TBaseLocal>(0); 
    });
    
    RunTest("Phase 17: Tail Latency (Atomic)", []() 
    { 
        RunLatencyBenchmark<TTestAtomic, TTestLocal>(1); 
    });

    RunTest("Phase 18: Tail Latency (Rundown)", []() 
    { 
        RunRundownLatencyBenchmark(); 
    });

    RunTest("Phase 19: Thread Scaling Sweep", []() 
    { 
        TestThreadScaling<TBaseAtomic, TBaseLocal, TTestAtomic, TTestLocal>(); 
    });

    if (g_AbortRequested.load(std::memory_order_relaxed) != 0)
    {
        Print("\n-------------------------------------------------------------------\n");
        Print("Suite aborted early by user or due to test failure.\n");
        PrintResultsTables(baseName, testName);
        return;
    }

    PrintResultsTables(baseName, testName);
    
    Print("\n-------------------------------------------------------------------\n");
    Print("All multi-threaded soak phases completed successfully!\n");
    Print("Zero Use-After-Free crashes. Zero memory leaks detected.\n");
}

// ----------------------------------------------------------------------------
// Application Entry Point
// Configures OS signal handlers, parses optional soak duration CLI arguments,
// initializes the background Rundown GC worker with a 65,535-entry watermark,
// and executes the full benchmark suite.
// ----------------------------------------------------------------------------
int main(int   argc, 
         char* argv[])
{
#if defined(_WIN32) || defined(_WIN64)
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
#else
    std::signal(SIGINT, PosixSignalHandler);
    std::signal(SIGTERM, PosixSignalHandler);
#endif

    [[maybe_unused]] TerminalState termState;

    if (argc > 1)
    {
        int parsed_duration = std::atoi(argv[1]);
        
        if (parsed_duration > 0)
        {
            g_SoakDurationMs = static_cast<uint32_t>(parsed_duration) * 1000U;
        }
    }
    else
    {
        Print("Usage: %s [phase_duration_in_seconds]\n", argv[0]);
    }

    std::cout << "\n===========================================================\n";
    std::cout << "    CPU: " << cpu_name() << "\n";    
    std::cout << "===========================================================\n";


    set_process_high_priority();

    // Hook tests to enforce sequential synchronous garbage collection pipeline flushes
    // natively at the bounds of each Execute-Around (Rundown) sequence, and clear any
    // pending high-watermark wake signal so GCWorkerThread never fires a stale wake
    // at the start of the subsequent measurement step.
    g_TestTeardownHook = []() 
    {
        SharedPtrGarbageCollector::get_instance().process_garbage_pipeline(true);
        g_GcWakePending.store(0, std::memory_order_release);
    };

    // Initialize GC thread notification via wake callback using KM-aligned 65,535 watermark
    SharedPtrGarbageCollector::get_instance().set_wake_callback(GCWakeCallback, GC_HIGH_WATERMARK);
    g_GcThread = std::thread(GCWorkerThread);

    RunAllPhases<std::atomic<std::shared_ptr<TrackedObject>>, 
                 std::shared_ptr<TrackedObject>, 
                 AtomicSharedPtr<TrackedObject>, 
                 LocalSharedPtr<TrackedObject>>("std::atomic<std::shared_ptr>", 
                                                "UmAtomicSharedPtr");

    // Trigger GC teardown explicitly
    g_AbortRequested.store(1, std::memory_order_relaxed);
    GCWakeCallback();
    
    if (g_GcThread.joinable())
    {
        g_GcThread.join();
    }

    return (g_TestFailed.load(std::memory_order_relaxed) != 0) ? 1 : 0;
}