# `AtomicSharedPtr` Usage Guide & Patterns

`atomic_shared_ptr.h` provides a lock-free, thread-safe smart pointer system for high-throughput user-mode applications on Windows, Linux, and macOS (x64 and ARM64). Using a 128-bit atomic publication anchor (`AtomicSharedPtr<T>`) paired with a 64-bit packed split-reference control block (`ControlBlock<T>`) and a zero-indirection thread-local handle (`LocalSharedPtr<T>`), it delivers full `std::atomic<std::shared_ptr<T>>` semantic parity—including lock-free `exchange` and `compare_exchange_*`—with deterministic, immediate memory reclamation and no background garbage collector thread.

---

## 1. Basic Lock-Free Usage (Snapshot `load` & Atomic `store`)

Readers call `load()` on a shared `AtomicSharedPtr<T>` anchor to check out an independent, reference-counted `LocalSharedPtr<T>` snapshot. Since `LocalSharedPtr<T>` caches the resolved `T*` payload pointer directly inside the handle, subsequent `operator->()`, `operator*()`, and `get()` calls execute with **zero control-block indirection**.

When an anchor is overwritten via `store()` and all checked-out `LocalSharedPtr<T>` handles leave scope, the displaced object is destroyed and freed immediately without waiting for a background GC cycle.

```cpp
#include <cstdint>
#include <utility>
#include "atomic_shared_ptr.h"

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

// 2. Declare a global or shared lock-free anchor (automatically cache-line aligned)
AtomicSharedPtr<NetworkConfig> g_NetConfig;

// ============================================================================
// Reader Routine (Lock-Free 128-Bit Checkout, Zero-Indirection Dereference)
// ============================================================================
bool ShouldAcceptConnection(uint32_t active_connections) noexcept
{
    // Atomically checks out a thread-local handle pinning the current snapshot.
    // The snapshot remains valid for the lifetime of 'cfg' even if writers
    // concurrently replace g_NetConfig.
    LocalSharedPtr<NetworkConfig> cfg = g_NetConfig.load();
    if (!cfg)
    {
        return false;
    }

    // Zero control-block indirection: cfg-> accesses the cached T* pointer directly
    return active_connections < cfg->max_connections;
} // 'cfg' leaves scope here; if it holds the final reference, the object is freed immediately

// ============================================================================
// Writer Routine (Single-Allocation Construction & Lock-Free Publication)
// ============================================================================
void UpdateNetworkConfig(uint32_t new_max_conn,
                         uint32_t new_timeout_ms,
                         bool     new_jumbo)
{
    // 1. Allocate ControlBlock + cache-isolated payload T in a single aligned block
    LocalSharedPtr<NetworkConfig> new_cfg = MakeLocalShared<NetworkConfig>(new_max_conn,
                                                                           new_timeout_ms,
                                                                           new_jumbo);

    // 2. Atomically replace the active anchor and reconcile the displaced object's
    //    split reference counts.
    g_NetConfig.store(std::move(new_cfg));
}
```

---

## 2. Lock-Free Copy-On-Write (COW) with `compare_exchange_strong`

Unlike epoch-rundown anchors, `AtomicSharedPtr<T>` supports true lock-free Read-Modify-Write (RMW) state machines via `compare_exchange_weak()` and `compare_exchange_strong()`.

* **Spurious-Failure Immunity:** Concurrent readers calling `load()` mutate the 128-bit anchor's `external_count`. `compare_exchange_strong()` automatically intercepts and retries those benign reader collisions in a tight loop while holding the cache line in Modified (M) state, returning `false` **only** if another writer genuinely replaced the underlying `ControlBlock<T>*` pointer.
* **Rvalue Move-Only-On-Success:** Passing `std::move(desired)` to the rvalue overload transfers ownership into the anchor on CAS success without an extra atomic increment, and leaves `desired` untouched on CAS failure while automatically updating `expected` to the newest snapshot.

```cpp
#include <cstdint>
#include <utility>
#include <vector>
#include "atomic_shared_ptr.h"

struct RoutingTable
{
    uint64_t              version;
    std::vector<uint32_t> active_routes;

    RoutingTable(uint64_t                    ver,
                 const std::vector<uint32_t>& routes) : version(ver),
                                                        active_routes(routes)
    {
    }
};

extern AtomicSharedPtr<RoutingTable> g_RoutingTable;

// Atomically appends a route using a lock-free Copy-On-Write CAS loop
uint64_t AppendRouteLockFree(uint32_t new_route_id)
{
    // 1. Check out the initial expected snapshot
    LocalSharedPtr<RoutingTable> expected = g_RoutingTable.load();

    while (true)
    {
        const uint64_t next_version = expected ? (expected->version + 1ULL) : 1ULL;

        std::vector<uint32_t> updated_routes;
        if (expected)
        {
            updated_routes = expected->active_routes;
        }
        updated_routes.push_back(new_route_id);

        // 2. Construct the candidate replacement snapshot
        LocalSharedPtr<RoutingTable> desired = MakeLocalShared<RoutingTable>(next_version,
                                                                             updated_routes);

        // 3. Attempt atomic commit.
        //    - If another reader called load(), compare_exchange_strong retries internally.
        //    - If another writer updated g_RoutingTable, compare_exchange_strong returns
        //      false and automatically updates 'expected' to the new snapshot.
        if (g_RoutingTable.compare_exchange_strong(expected, std::move(desired)))
        {
            return next_version;
        }
    }
}
```

### Using `compare_exchange_weak` for Low-Cost Speculative Updates

When a failed CAS does not require reconstructing `desired` from scratch (for example, installing a pre-built fallback object only if the anchor is still empty), you can use `compare_exchange_weak()` with an rvalue handle. If `compare_exchange_weak()` fails spuriously, `desired` retains ownership so it can be reused directly on the next iteration:

```cpp
#include <utility>
#include "atomic_shared_ptr.h"

extern AtomicSharedPtr<NetworkConfig> g_NetConfig;

bool TryInstallDefaultConfigIfEmpty()
{
    LocalSharedPtr<NetworkConfig> expected = nullptr;
    LocalSharedPtr<NetworkConfig> fallback = MakeLocalShared<NetworkConfig>(1024u, 250u, false);

    // Retry only while the anchor remains empty (expected == nullptr).
    // On spurious failure, 'fallback' is NOT moved from and remains valid for the next iteration.
    while (!expected)
    {
        if (g_NetConfig.compare_exchange_weak(expected, std::move(fallback)))
        {
            return true; // Default config installed; 'fallback' is now empty
        }
    }

    return false; // Another thread already populated g_NetConfig; 'expected' holds that snapshot
}
```

---

## 3. Atomic Swap & Batch Draining with `exchange()`

`exchange()` atomically replaces the anchor with a new `LocalSharedPtr<T>` and returns the displaced object as a `LocalSharedPtr<T>`, transferring the anchor's baseline internal reference directly to the caller and reconciling any accumulated reader `external_count` deltas.

This pattern is ideal for lock-free double-buffering or periodic telemetry/log batch harvesting:

```cpp
#include <atomic>
#include <cstdint>
#include <utility>
#include "atomic_shared_ptr.h"

struct TelemetryBucket
{
    std::atomic<uint64_t> request_count{0};
    std::atomic<uint64_t> error_count{0};
};

AtomicSharedPtr<TelemetryBucket> g_ActiveTelemetry(MakeLocalShared<TelemetryBucket>());

// ============================================================================
// Worker Threads: Record metrics into the currently active bucket
// ============================================================================
void RecordRequestOutcome(bool is_error) noexcept
{
    LocalSharedPtr<TelemetryBucket> bucket = g_ActiveTelemetry.load();
    if (bucket)
    {
        bucket->request_count.fetch_add(1, std::memory_order_relaxed);
        if (is_error)
        {
            bucket->error_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// ============================================================================
// Harvester Thread: Atomically swaps in a fresh bucket and reads the old one
// ============================================================================
void HarvestTelemetrySnapshot(uint64_t& out_requests,
                              uint64_t& out_errors)
{
    LocalSharedPtr<TelemetryBucket> fresh_bucket = MakeLocalShared<TelemetryBucket>();

    // Atomically swap in 'fresh_bucket' and take ownership of the displaced bucket
    LocalSharedPtr<TelemetryBucket> old_bucket = g_ActiveTelemetry.exchange(std::move(fresh_bucket));

    if (old_bucket)
    {
        out_requests = old_bucket->request_count.load(std::memory_order_acquire);
        out_errors   = old_bucket->error_count.load(std::memory_order_acquire);
    }
    else
    {
        out_requests = 0;
        out_errors   = 0;
    }
} // old_bucket is destroyed once any in-flight RecordRequestOutcome() readers finish
```

---

## 4. Extended Blocking, Async Tasks & Handle Manipulation (`LocalSharedPtr<T>`)

Since `LocalSharedPtr<T>` is a standalone reference-counted handle completely decoupled from the `AtomicSharedPtr<T>` anchor once checked out, you can hold it across arbitrary blocking I/O, sleeps, thread boundaries, or work queues without delaying writers or other readers.

```cpp
#include <chrono>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include "atomic_shared_ptr.h"

extern AtomicSharedPtr<NetworkConfig> g_NetConfig;

// ============================================================================
// Holding a Snapshot Across Blocking I/O & Local Handle Operations
// ============================================================================
void ExecuteBlockingOperation()
{
    // 1. Check out a local handle (or use implicit conversion: LocalSharedPtr<NetworkConfig> cfg = g_NetConfig;)
    LocalSharedPtr<NetworkConfig> cfg = g_NetConfig.load();
    if (!cfg)
    {
        return;
    }

    // 2. Blocking or sleeping while holding 'cfg' never blocks writers from
    //    updating g_NetConfig or reclaiming newer/older snapshots.
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg->timeout_ms));

    // 3. Standard smart-pointer handle operations:
    LocalSharedPtr<NetworkConfig> copy_cfg = cfg; // Wait-free 64-bit fetch_add (+1 internal)
    long approx_refs = copy_cfg.use_count();      // Approximate combined reference count
    (void)approx_refs;

    LocalSharedPtr<NetworkConfig> moved_cfg;
    moved_cfg.swap(copy_cfg);                     // 0 atomic operations
    moved_cfg.reset();                            // Explicitly drops reference early

    // 4. Clearing an anchor globally:
    // g_NetConfig.store(nullptr);
}

// ============================================================================
// Passing Snapshots into an Asynchronous Work Queue
// ============================================================================
struct WorkItem
{
    LocalSharedPtr<NetworkConfig> config_snapshot;
    uint32_t                      job_id;

    WorkItem(LocalSharedPtr<NetworkConfig> cfg,
             uint32_t                      id) noexcept : config_snapshot(std::move(cfg)),
                                                          job_id(id)
    {
    }
};

class AsyncJobProcessor
{
    std::mutex           mtx_;
    std::queue<WorkItem> queue_;

public:
    bool SubmitJob(uint32_t job_id)
    {
        LocalSharedPtr<NetworkConfig> snapshot = g_NetConfig.load();
        if (!snapshot)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(mtx_);
        queue_.emplace(std::move(snapshot), job_id);
        return true;
    }

    bool RunNextJob()
    {
        WorkItem item(nullptr, 0);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (queue_.empty())
            {
                return false;
            }

            item = std::move(queue_.front());
            queue_.pop();
        }

        if (item.config_snapshot && item.config_snapshot->jumbo_frames)
        {
            // Execute job with guaranteed-consistent config_snapshot...
        }

        return true;
    }
};
```

---

## 5. Custom Memory Ordering & Hardware Capability Queries

By default, `AtomicSharedPtr<T>` uses `std::memory_order_acquire` on `load()`, `std::memory_order_release` on `store()`, and `std::memory_order_acq_rel` on `exchange()` and `compare_exchange_*()` to avoid unnecessary `seq_cst` fence overhead while guaranteeing safe cross-thread publication and reclamation.

You can also query hardware 128-bit lock-free support and pass explicit `std::memory_order` tags when required:

```cpp
#include <atomic>
#include <cassert>
#include "atomic_shared_ptr.h"

extern AtomicSharedPtr<NetworkConfig> g_NetConfig;

void MemoryOrderingExamples()
{
    // 1. Verify hardware 128-bit lock-free execution (CMPXCHG16B on x64, CASP/LDXP+STXP on ARM64)
    assert(g_NetConfig.is_lock_free());

    // 2. Sequentially consistent store and load
    LocalSharedPtr<NetworkConfig> next_cfg = MakeLocalShared<NetworkConfig>(4096u, 100u, true);
    g_NetConfig.store(std::move(next_cfg), std::memory_order_seq_cst);

    LocalSharedPtr<NetworkConfig> current = g_NetConfig.load(std::memory_order_seq_cst);

    // 3. Dual-order compare_exchange_strong
    LocalSharedPtr<NetworkConfig> replacement = MakeLocalShared<NetworkConfig>(8192u, 100u, true);
    bool swapped = g_NetConfig.compare_exchange_strong(current,
                                                       std::move(replacement),
                                                       std::memory_order_acq_rel, // Success order
                                                       std::memory_order_acquire); // Failure load order
    (void)swapped;
}
```

> **Note on Memory Order Safety:**
> * Passing `std::memory_order_relaxed` to `load()` or as a CAS failure order is automatically promoted to `std::memory_order_acquire` so control-block pointer dereferences never race with concurrent construction.
> * Passing invalid load orders (`std::memory_order_release` or `std::memory_order_acq_rel`) to `load()` triggers the fatal error handler (`fatal_panic("Invalid load memory order")`).

---

## 6. Pluggable Fatal Error & Invariant Handling (`SetAtomicPtrErrorHandler`)

`atomic_shared_ptr.h` continuously verifies internal split-count invariants (32-bit signed counter overflow/underflow, double-free detection if `internal < 0`, reader saturation at `INT32_MAX`, and invalid memory orders).

Before terminating the process via `std::abort()`, `fatal_panic()` invokes a user-configurable callback registered via `SetAtomicPtrErrorHandler()`. You can install a custom handler at startup to route fatal invariant diagnostics to your application logger or crash reporter:

```cpp
#include <cstdio>
#include "atomic_shared_ptr.h"

// Custom noexcept fatal error callback invoked prior to std::abort()
void OnAtomicSharedPtrFatalError(const char* reason) noexcept
{
    std::fprintf(stderr, "[CRITICAL] AtomicSharedPtr invariant violation: %s\n", reason);
    std::fflush(stderr);
}

void InitializeApplication()
{
    // Install custom fatal error handler (pass nullptr to restore default std::cerr output)
    SetAtomicPtrErrorHandler(&OnAtomicSharedPtrFatalError);
}
```