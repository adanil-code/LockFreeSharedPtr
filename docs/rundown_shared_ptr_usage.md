# `RundownSharedPtr` Usage Guide & Patterns

`rundown_shared_ptr.h` provides an epoch-based, NUMA- and cache-aware smart pointer for high-throughput user-mode applications on Windows, Linux, and macOS (x64 and ARM64). Short-lived readers scale wait-free across cores without mutating per-object reference counts, while writers publish updates lock-free and defer reclamation to an $O(1)$ two-phase epoch garbage collector.

---

## 1. Basic Rundown Usage (Fast-Path `read` & Lock-Free `store`)

Fast-path reads use the execute-around `read()` lambda pattern. The calling thread acquires wait-free rundown protection on its thread-local epoch slot, resolves the payload pointer via constant compile-time offset arithmetic (never touching the control block's mutable reference-count cache line), invokes your lambda, and releases the slot with a single-writer non-RMW store on scope exit.

```cpp
#include <cstdint>
#include <utility>
#include "rundown_shared_ptr.h"

// 1. Define the shared data structure
struct NetworkConfig
{
    uint32_t max_connections;
    uint32_t timeout_ms;
    bool     jumbo_frames;

    NetworkConfig(uint32_t max_conn,
                  uint32_t timeout,
                  bool     jumbo) noexcept : max_connections(max_conn),
                                             timeout_ms(timeout),
                                             jumbo_frames(jumbo)
    {
    }
};

// 2. Declare a global or shared rundown-protected anchor
RundownSharedPtr<NetworkConfig> g_NetConfig;

// ============================================================================
// Reader Routine (Wait-Free, 0 Per-Object Atomic RMWs)
// ============================================================================
bool ShouldAcceptConnection(uint32_t active_connections) noexcept
{
    // read() binds lifetime protection strictly to the lambda scope and
    // forwards the lambda's by-value return result.
    // Returning raw pointers or references from read() is blocked at compile time.
    return g_NetConfig.read([active_connections](const NetworkConfig* cfg) noexcept -> bool
    {
        if (!cfg)
        {
            return false;
        }

        return active_connections < cfg->max_connections;
    });
}

// ============================================================================
// Writer Routine (Lock-Free Publication)
// ============================================================================
bool UpdateNetworkConfig(uint32_t new_max_conn,
                         uint32_t new_timeout_ms,
                         bool     new_jumbo) noexcept
{
    // 1. Allocate control block + cache-aligned payload in a single block
    PersistentSharedPtr<NetworkConfig> new_cfg = MakeRundownShared<NetworkConfig>(new_max_conn,
                                                                                  new_timeout_ms,
                                                                                  new_jumbo);
    if (!new_cfg)
    {
        return false; // Allocation failed
    }

    // 2. Atomically replace the active pointer and stage the displaced object
    //    into the calling processor's NUMA-local GC queue.
    return g_NetConfig.store(std::move(new_cfg));
}
```

---

## 2. High-Throughput Burst Reads with `acquire_batched()` & `BatchedRundownGuard<T>`

When a thread needs to read the anchor repeatedly inside a tight loop (for example, filtering a batch of packets), calling `acquire_batched()` acquires the epoch rundown slot **once** and returns a stack-bound `BatchedRundownGuard<T>`. Each subsequent `guard.read(...)` executes with **zero atomic operations**.

For very large loops, acquire the guard in chunks (e.g., 32 iterations per batch) so the reader periodically refreshes its epoch slot and never delays background garbage collection.

```cpp
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "rundown_shared_ptr.h"

extern RundownSharedPtr<NetworkConfig> g_NetConfig;

size_t CountAllowedPackets(const std::vector<uint32_t>& packet_conn_counts) noexcept
{
    constexpr size_t BATCH_CHECKPOINT = 32;
    size_t           allowed_count    = 0;
    size_t           i                = 0;

    while (i < packet_conn_counts.size())
    {
        const size_t chunk_end = std::min(packet_conn_counts.size(), i + BATCH_CHECKPOINT);

        // Pin the epoch slot and control block once for up to 32 consecutive reads
        BatchedRundownGuard<NetworkConfig> batch_guard = g_NetConfig.acquire_batched();

        for (; i < chunk_end; ++i)
        {
            const uint32_t conn_count = packet_conn_counts[i];

            // Zero atomic instructions per iteration inside batch_guard.read()
            bool allowed = batch_guard.read([conn_count](const NetworkConfig* cfg) noexcept -> bool
            {
                return cfg && (conn_count < cfg->max_connections);
            });

            if (allowed)
            {
                ++allowed_count;
            }
        }
    } // batch_guard leaves scope here, refreshing the epoch slot every 32 items

    return allowed_count;
}
```

---

## 3. Extended Blocking & Out-of-Scope Access via `load_persistent()`

Since `read()` and `BatchedRundownGuard<T>` hold an active slot on the current epoch shard, you should not perform long sleeps or blocking I/O inside a `read()` lambda.

When you need to hold the object across blocking calls or outside the immediate scope, call `load_persistent()`. It increments the control block's `persistent_ref_count` under transient epoch protection, **immediately releases the epoch rundown slot**, and returns a standard reference-counted `PersistentSharedPtr<T>`.

```cpp
#include <chrono>
#include <thread>
#include "rundown_shared_ptr.h"

extern RundownSharedPtr<NetworkConfig> g_NetConfig;

void PerformBlockingHealthCheck() noexcept
{
    // 1. Promote the current object to a persistent reference-counted handle.
    //    The transient epoch rundown slot is released before load_persistent() returns.
    PersistentSharedPtr<NetworkConfig> pinned_cfg = g_NetConfig.load_persistent();

    // 2. Check validity via operator bool()
    if (!pinned_cfg)
    {
        return;
    }

    // 3. Dereference directly via operator->, operator*, or get() outside any lambda.
    //    Blocking here never stalls background GC epoch advancement.
    const uint32_t wait_ms = pinned_cfg->timeout_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));

    // 4. Optional handle management: swap() or explicit reset()
    PersistentSharedPtr<NetworkConfig> backup_cfg;
    backup_cfg.swap(pinned_cfg); // Transfers ownership to backup_cfg without atomic RMWs
    backup_cfg.reset();          // Explicitly drops the persistent reference early
}
```

---

## 4. Asynchronous Task & Work-Queue Caching

To guarantee that an asynchronous task, thread-pool job, or stream context uses a consistent configuration snapshot from the moment it was created, store a `PersistentSharedPtr<T>` inside the task object:

```cpp
#include <cstdint>
#include <queue>
#include <mutex>
#include <utility>
#include "rundown_shared_ptr.h"

struct PacketTask
{
    PersistentSharedPtr<NetworkConfig> config_snapshot;
    uint32_t                           packet_id;

    PacketTask(PersistentSharedPtr<NetworkConfig> cfg,
               uint32_t                           id) noexcept : config_snapshot(std::move(cfg)),
                                                                 packet_id(id)
    {
    }
};

class TaskQueue
{
    std::mutex             mtx_;
    std::queue<PacketTask> queue_;

public:
    // Producer: captures the active configuration snapshot at enqueue time
    bool Enqueue(const RundownSharedPtr<NetworkConfig>& global_cfg,
                 uint32_t                               packet_id)
    {
        PersistentSharedPtr<NetworkConfig> snapshot = global_cfg.load_persistent();
        if (!snapshot)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(mtx_);
        queue_.emplace(std::move(snapshot), packet_id);
        return true;
    }

    // Consumer: processes the task later using its captured snapshot
    bool TryProcessNext()
    {
        PacketTask task(nullptr, 0);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (queue_.empty())
            {
                return false;
            }

            task = std::move(queue_.front());
            queue_.pop();
        }

        // Safe access even if global_cfg was updated multiple times since Enqueue()
        if (task.config_snapshot && task.config_snapshot->jumbo_frames)
        {
            // Process jumbo packet...
        }

        return true; // task destructor automatically releases config_snapshot
    }
};
```

---

## 5. NUMA-Aware Allocation (`MakeRundownSharedNuma` & `NumaAllocator`)

On multi-socket servers, you can query the discovered NUMA topology via `NumaAllocator` and allocate objects directly on a specific physical NUMA node with `MakeRundownSharedNuma<T>(preferred_node, ...)`:

```cpp
#include <cstdio>
#include <vector>
#include "rundown_shared_ptr.h"

// One cache-aligned rundown anchor per physical NUMA node
RundownSharedPtr<NetworkConfig> g_NodeConfigs[NumaAllocator::MAX_NUMA_NODES];

void InitializePerNodeConfigs()
{
    // 1. Query discovered online physical NUMA nodes
    const std::vector<uint32_t>& valid_nodes = NumaAllocator::GetValidNodes();

    for (uint32_t node_id : valid_nodes)
    {
        // 2. Allocate control block + payload physically affinitized to node_id
        PersistentSharedPtr<NetworkConfig> node_cfg = MakeRundownSharedNuma<NetworkConfig>(node_id,
                                                                                           10000u, // max_connections
                                                                                           500u,   // timeout_ms
                                                                                           true);  // jumbo_frames

        g_NodeConfigs[node_id].store(std::move(node_cfg));
    }
}

uint32_t ReadLocalNodeTimeout() noexcept
{
    // 3. Resolve the calling processor's local NUMA node (amortized TLS lookup)
    const uint32_t local_node = NumaAllocator::GetCurrentNode();

    return g_NodeConfigs[local_node].read([](const NetworkConfig* cfg) noexcept -> uint32_t
    {
        return cfg ? cfg->timeout_ms : 0u;
    });
}
```

---

## 6. Garbage Collection & Application Teardown (`SharedPtrGarbageCollector`)

Displaced objects retired by `store()` are reclaimed by the process-wide `SharedPtrGarbageCollector` singleton. At process exit, `~SharedPtrGarbageCollector()` automatically flushes all remaining retired objects and block caches. During normal execution, choose one of the two patterns below.

### Option A: Periodic Maintenance or Manual Flush (Simple Applications)

If your application updates pointers infrequently or already has an existing background maintenance loop, call `process_garbage_pipeline()` directly:

```cpp
#include "rundown_shared_ptr.h"

// Call periodically from an existing timer or maintenance thread:
void OnMaintenanceTick() noexcept
{
    // is_flush = false: non-blocking two-phase epoch advancement and batch free.
    // Returns immediately if a reader is still active on the old epoch.
    SharedPtrGarbageCollector::get_instance().process_garbage_pipeline(false);
}

// Optional: synchronous drain + control-block cache release (e.g., after a big reload):
void ForceFullGarbageFlush() noexcept
{
    // is_flush = true: waits until all pending epochs drain and releases cached blocks.
    // Never call with is_flush = true from inside a read() lambda!
    SharedPtrGarbageCollector::get_instance().process_garbage_pipeline(true);
}
```

### Option B: Event-Driven Background GC Worker Thread (High-Frequency Updates)

For write-heavy workloads, register a high-watermark callback via `set_wake_callback()` to wake a dedicated background thread whenever the retired queue depth reaches your threshold:

```cpp
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "rundown_shared_ptr.h"

class RundownGcService
{
    std::mutex              mtx_;
    std::condition_variable cv_;
    std::thread             worker_;

    inline static std::atomic<int32_t> wake_pending_{0};
    inline static std::atomic<int32_t> stop_requested_{0};
    inline static RundownGcService*    active_service_{nullptr};

    // High-watermark callback invoked from store() when queue depth hits watermark.
    // Uses an atomic 0 -> 1 transition check so writers only lock mtx_ once per cycle.
    static void OnGcWakeCallback() noexcept
    {
        if (wake_pending_.exchange(1, std::memory_order_acq_rel) == 0)
        {
            if (RundownGcService* svc = active_service_)
            {
                {
                    std::lock_guard<std::mutex> lock(svc->mtx_);
                }
                svc->cv_.notify_one();
            }
        }
    }

    void WorkerLoop() noexcept
    {
        SharedPtrGarbageCollector& gc = SharedPtrGarbageCollector::get_instance();

        while (stop_requested_.load(std::memory_order_acquire) == 0)
        {
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(1000), []() noexcept
                {
                    return stop_requested_.load(std::memory_order_acquire) != 0 ||
                           wake_pending_.load(std::memory_order_acquire) != 0;
                });

                wake_pending_.store(0, std::memory_order_release);
            }

            if (stop_requested_.load(std::memory_order_acquire) != 0)
            {
                break;
            }

            // Non-blocking epoch advancement and batch reclamation
            gc.process_garbage_pipeline(false);
        }

        // Final synchronous flush before worker exits
        gc.process_garbage_pipeline(true);
    }

public:
    void Start(int32_t high_watermark = 4096)
    {
        stop_requested_.store(0, std::memory_order_relaxed);
        wake_pending_.store(0, std::memory_order_relaxed);
        active_service_ = this;

        SharedPtrGarbageCollector::get_instance().set_wake_callback(OnGcWakeCallback,
                                                                    high_watermark);
        worker_ = std::thread(&RundownGcService::WorkerLoop, this);
    }

    void Stop() noexcept
    {
        // 1. Unregister callback before stopping the worker thread
        SharedPtrGarbageCollector::get_instance().set_wake_callback(nullptr);

        // 2. Signal stop and wake the worker
        stop_requested_.store(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mtx_);
        }
        
        cv_.notify_one();

        // 3. Join background thread
        if (worker_.joinable())
        {
            worker_.join();
        }

        active_service_ = nullptr;
    }

    ~RundownGcService() noexcept
    {
        Stop();
    }
};
```

---

## 7. Pluggable Fatal Error & Invariant Handling (`SetRundownPtrErrorHandler`)

`rundown_shared_ptr.h` continuously verifies internal reference-counting and allocator invariants (32-bit signed `persistent_ref_count` overflow at `INT32_MAX`, double-free/underflow if `persistent_ref_count` drops below `0`, 64 KB NUMA slab `active_blocks` underflow, and shared reader slot `overflow_slot_refs` underflow).

Before terminating the process via `std::abort()`, `rundown_fatal_panic()` invokes a user-configurable callback registered via `SetRundownPtrErrorHandler()`. You can install a custom handler at startup to route fatal invariant diagnostics to your application logger or crash reporter:

```cpp
#include <cstdio>
#include "rundown_shared_ptr.h"

// Custom noexcept fatal error callback invoked prior to std::abort()
void OnRundownSharedPtrFatalError(const char* reason) noexcept
{
    std::fprintf(stderr, "[CRITICAL] RundownSharedPtr invariant violation: %s\n", reason);
    std::fflush(stderr);
}

void InitializeApplication() noexcept
{
    // Install custom fatal error handler (pass nullptr to restore default std::cerr output)
    SetRundownPtrErrorHandler(&OnRundownSharedPtrFatalError);
}
```