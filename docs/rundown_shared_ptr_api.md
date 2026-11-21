# `rundown_shared_ptr.h` Client API Reference

`rundown_shared_ptr.h` provides an Epoch-Based, NUMA- and Cache-Aware Rundown shared pointer for high-throughput user-mode applications across Windows, Linux, and macOS (x64 and ARM64). It decouples short-lived transient reads (`RundownSharedPtr<T>::read` and `RundownSharedPtr<T>::acquire_batched`) from long-lived persistent ownership (`PersistentSharedPtr<T>`), achieving wait-free reader scaling with zero per-object atomic reference-count contention on the read path and $O(1)$ batched memory reclamation.

## Table of Contents

1. [Component & Concurrency Summary](#1-component--concurrency-summary)
2. [Single-Allocation NUMA-Aware Factories](#2-single-allocation-numa-aware-factories)
   * [`MakeRundownShared`](#makerundownshared)
   * [`MakeRundownSharedNuma`](#makerundownsharednuma)
3. [Shared Rundown Anchor (`RundownSharedPtr<T>`)](#3-shared-rundown-anchor-rundownsharedptrt)
   * [Constructors & Destructor](#constructors--destructor)
   * [`read`](#read)
   * [`acquire_batched`](#acquire_batched)
   * [`load_persistent`](#load_persistent)
   * [`store`](#store)
4. [Batched Rundown Guard (`BatchedRundownGuard<T>`)](#4-batched-rundown-guard-batchedrundownguardt)
   * [Constructors & Destructor](#constructors--destructor-1)
   * [`read`](#read-1)
5. [Persistent Reference-Counted Handle (`PersistentSharedPtr<T>`)](#5-persistent-reference-counted-handle-persistentsharedptrt)
   * [Constructors & Destructor](#constructors--destructor-2)
   * [Assignment Operators](#assignment-operators)
   * [`swap`](#swap)
   * [`reset`](#reset)
   * [`get`](#get)
   * [`operator->` & `operator*`](#operator--operator)
   * [`operator bool`](#operator-bool)
6. [Epoch Rundown Garbage Collector (`SharedPtrGarbageCollector`)](#6-epoch-rundown-garbage-collector-sharedptrgarbagecollector)
   * [`get_instance`](#get_instance)
   * [`set_wake_callback`](#set_wake_callback)
   * [`process_garbage_pipeline`](#process_garbage_pipeline)
   * [Process Teardown (`~SharedPtrGarbageCollector`)](#process-teardown-sharedptrgarbagecollector)
7. [Pluggable Fatal Error Handling](#7-pluggable-fatal-error-handling)
   * [`RundownFatalErrorHandler`](#rundownfatalerrorhandler)
   * [`SetRundownPtrErrorHandler`](#setrundownptrerrorhandler)
   * [`rundown_fatal_panic`](#rundown_fatal_panic)

---

## 1. Component & Concurrency Summary

`RundownSharedPtr<T>` and `RundownControlBlock<T>` are automatically aligned to `CACHE_LINE_SIZE` (64 bytes on x64; 128 bytes on ARM64 and Apple Silicon) to prevent false sharing across adjacent anchors, control blocks, or surrounding struct members.

| Component / Function | Role | Exception Specification | Progress Guarantee |
| :--- | :--- | :--- | :--- |
| **`MakeRundownShared` / `MakeRundownSharedNuma`** | Single-allocation NUMA-aware object factory | Propagates `T` ctor exceptions (`noexcept` allocation) | Lock-free on per-slot cache hit; spin-locked on node spare stack / 64 KB NUMA slab refill |
| **`RundownSharedPtr<T>::read`** | Scoped transient execute-around read | `noexcept(std::is_nothrow_invocable_v<F, T*>)` | Wait-free (`0` per-object atomic RMWs; non-RMW release store on exit for primary slot owners) |
| **`RundownSharedPtr<T>::acquire_batched`** | Pins epoch slot across a multi-read burst | `noexcept` | Wait-free |
| **`BatchedRundownGuard<T>::read`** | Zero-atomic-overhead read within a pinned batch | `noexcept(std::is_nothrow_invocable_v<F, T*>)` | Wait-free (`0` atomic operations per call) |
| **`RundownSharedPtr<T>::load_persistent`** | Promotes active object to `PersistentSharedPtr<T>` | `noexcept` | Wait-free |
| **`RundownSharedPtr<T>::store`** | Atomically replaces anchor and retires displaced object | `noexcept` | Lock-free |
| **`PersistentSharedPtr<T>`** | Reference-counted handle usable outside `read()` scope | `noexcept` | Wait-free copy, move, and destroy |
| **`SharedPtrGarbageCollector::get_instance`** | Retrieves the process-wide collector singleton | `noexcept` | Wait-free after static initialization |
| **`SharedPtrGarbageCollector::set_wake_callback`** | Configures high-watermark wake notification | `noexcept` | Wait-free |
| **`SharedPtrGarbageCollector::process_garbage_pipeline`** | Advances epoch, checks rundown drain, and frees batch | `noexcept` | Lock-free try-reclaim (`is_flush == false`); blocking spin-wait (`is_flush == true`) |
| **`SetRundownPtrErrorHandler`** | Registers custom invariant-violation panic callback | `noexcept` | Wait-free |

---

## 2. Single-Allocation NUMA-Aware Factories

Both factory functions allocate a single cache-aligned memory block placing `RundownControlBlock<T>` at offset `0` and embedding the `T` payload at a compile-time constant cache-aligned offset (`RundownControlBlock<T>::get_payload_offset()`). This allows fast-path readers to resolve `T*` via constant pointer arithmetic without touching the control block's mutable cache line.

### `MakeRundownShared`

```cpp
template <typename T, typename... Args>
[[nodiscard]] PersistentSharedPtr<T> MakeRundownShared(Args&&... args);
```

* **Purpose:** Allocates and constructs a new object of type `T` on the calling processor's local NUMA node (`NumaAllocator::ANY_NODE_OK`), returning a persistent handle ready to be published into a `RundownSharedPtr<T>` anchor via `store()` or retained directly.
* **Restrictions & Guarantees:**
  * Consults the calling thread's exclusive per-slot block cache (`SlotBlockCache`) first, then the local NUMA node's multi-batch `SpareBatchStack`, and finally carves up to 16 blocks from a 64 KB node-local NUMA slab (or cache-aligned `::operator new` on single-node UMA / macOS systems).
  * Does not throw `std::bad_alloc` on allocation failure; returns an empty `PersistentSharedPtr<T>` (`nullptr`) if memory is exhausted.
  * If `T`'s constructor throws an exception, the raw control block is automatically returned to its NUMA cache/slab before the exception is rethrown.
* **Parameters:**
  * `args`: Constructor arguments perfectly forwarded to `T`'s constructor.
* **Return Value:**
  * A `PersistentSharedPtr<T>` managing the newly constructed object, or an empty `PersistentSharedPtr<T>` (`nullptr`) if memory allocation fails.

### `MakeRundownSharedNuma`

```cpp
template <typename T, typename... Args>
[[nodiscard]] PersistentSharedPtr<T> MakeRundownSharedNuma(uint32_t preferred_node,
                                                           Args&&... args);
```

* **Purpose:** Allocates and constructs a new object of type `T` physically affinitized to a specified NUMA node.
* **Restrictions & Guarantees:**
  * If `preferred_node` is a valid online physical NUMA node ID (`NumaAllocator::GetTopology().IsValidNode(preferred_node)`), allocates from that node's spare stack or 64 KB NUMA slab without evicting the calling thread's local-node slot cache.
  * If `preferred_node` is `NumaAllocator::ANY_NODE_OK` (`0xFFFFFFFFu`) or invalid, falls back to the calling processor's current NUMA node (`NumaAllocator::GetCurrentNode()`).
  * Returns an empty `PersistentSharedPtr<T>` (`nullptr`) if memory allocation fails, and cleanly recycles the raw block if `T`'s constructor throws.
* **Parameters:**
  * `preferred_node`: Zero-based physical NUMA node identifier where the control block and payload should reside, or `NumaAllocator::ANY_NODE_OK`.
  * `args`: Constructor arguments perfectly forwarded to `T`'s constructor.
* **Return Value:**
  * A `PersistentSharedPtr<T>` managing the newly constructed object, or an empty `PersistentSharedPtr<T>` (`nullptr`) if memory allocation fails.

---

## 3. Shared Rundown Anchor (`RundownSharedPtr<T>`)

`RundownSharedPtr<T>` is a cache-line-isolated (`alignas(CACHE_LINE_SIZE)`) global or shared publication anchor. Multiple threads may concurrently read from and write to the same `RundownSharedPtr<T>` instance without external locks.

### Constructors & Destructor

#### `RundownSharedPtr::RundownSharedPtr` (Default Constructor)

```cpp
RundownSharedPtr() noexcept;
```

* **Purpose:** Initializes an empty rundown anchor containing no active object (`nullptr`) and ensures the process-wide `SharedPtrGarbageCollector` singleton is constructed prior to the anchor so static destruction order is preserved at process exit.
* **Restrictions:**
  * `RundownSharedPtr<T>` is neither copyable nor movable.
* **Parameters:**
  * None.
* **Return Value:**
  * None.

#### `RundownSharedPtr::RundownSharedPtr` (Initializing Constructor)

```cpp
explicit RundownSharedPtr(PersistentSharedPtr<T> desired) noexcept;
```

* **Purpose:** Initializes the rundown anchor, ensures the `SharedPtrGarbageCollector` singleton is constructed, and publishes `desired` as the active object.
* **Restrictions:**
  * `desired` must not already be anchored in another `RundownSharedPtr` instance or pending reclamation in the garbage collector.
* **Parameters:**
  * `desired`: The persistent handle whose managed object is transferred into the anchor (pass via `std::move()` to transfer baseline ownership without atomic reference-count changes).
* **Return Value:**
  * None.

#### `RundownSharedPtr::~RundownSharedPtr` (Destructor)

```cpp
~RundownSharedPtr() noexcept;
```

* **Purpose:** Displaces any currently anchored object into the `SharedPtrGarbageCollector` staging pipeline via `store(PersistentSharedPtr<T>())`.
* **Restrictions:**
  * Does not block waiting for readers to drain; reclamation is handled asynchronously by `SharedPtrGarbageCollector::process_garbage_pipeline()` or synchronously during final process shutdown.
* **Parameters:**
  * None.
* **Return Value:**
  * None.

### `read`

```cpp
template <typename F>
FORCE_INLINE decltype(auto) read(F&& func) const noexcept(std::is_nothrow_invocable_v<F, T*>);
```

* **Purpose:** Grants transient, scoped access to the currently anchored object by executing `func` under epoch rundown protection.
* **Restrictions & Guarantees:**
  * **Wait-Free Per-Thread Slot Assignment:** The first 256 concurrent reader threads claim exclusive ownership of a cache-isolated `RundownRef` slot, using a single-writer non-RMW release store on exit. Additional threads beyond 256 (or late `thread_local` destructors) transparently share a slot via an atomic `shared_readers` counter.
  * **Exception Safety:** Conditionally `noexcept` matching `func`. If `func` throws an exception, the internal `BatchedRundownGuard` / `ScopedRundownGuard` destructor automatically releases the thread's rundown slot during stack unwinding.
  * **No Smuggling Rule:** `func` **must not** return a raw pointer or C++ reference (enforced at compile time via `static_assert`), and must not cache the raw `T*` argument for use after `read()` returns.
  * **Hold Duration:** Keep `func` short and non-blocking. Holding an active rundown slot across long sleeps or blocking I/O delays `SharedPtrGarbageCollector::process_garbage_pipeline()` from reclaiming the outgoing epoch shard. For long-lived or asynchronous operations, call `load_persistent()` instead.
* **Parameters:**
  * `func`: A callable or lambda accepting `T*` (or `const T*`). Receives a pointer to the active payload object, or `nullptr` if the anchor is empty.
* **Return Value:**
  * Forwards the by-value return result of `func` (or `void` if `func` returns `void`).

### `acquire_batched`

```cpp
[[nodiscard]] FORCE_INLINE BatchedRundownGuard<T> acquire_batched() const noexcept;
```

* **Purpose:** Acquires and validates epoch rundown protection once and returns a stack-bound `BatchedRundownGuard<T>` pinning the current control block, allowing tight loops to execute multiple `read()` calls without repeating atomic epoch slot acquisition/release on every iteration.
* **Restrictions:**
  * Relies on C++17 guaranteed copy elision (RVO) because `BatchedRundownGuard<T>` is non-copyable and non-movable.
  * In long-running loops (e.g., hundreds or thousands of iterations), periodically let `BatchedRundownGuard<T>` go out of scope and re-call `acquire_batched()` (for example, every 32 reads) so the thread does not pin an old epoch indefinitely and delay background garbage collection.
* **Parameters:**
  * None.
* **Return Value:**
  * A `BatchedRundownGuard<T>` pinning the active epoch slot and control block (or an empty guard if the anchor is currently `nullptr`).

### `load_persistent`

```cpp
[[nodiscard]] PersistentSharedPtr<T> load_persistent() const noexcept;
```

* **Purpose:** Promotes the currently anchored object to an independently reference-counted `PersistentSharedPtr<T>` under transient epoch protection, allowing the object to outlive the transient scope (e.g., for asynchronous tasks, stream contexts, or blocking I/O).
* **Restrictions:**
  * Delegates epoch slot acquisition and validation to `acquire_batched()`, increments `persistent_ref_count` (`memory_order_acq_rel`), and releases transient epoch rundown protection before returning.
* **Parameters:**
  * None.
* **Return Value:**
  * A `PersistentSharedPtr<T>` owning a strong persistent reference to the active object, or an empty `PersistentSharedPtr<T>` (`nullptr`) if the anchor is empty.

### `store`

```cpp
bool store(PersistentSharedPtr<T> desired) noexcept;
```

* **Purpose:** Atomically replaces the currently anchored object with `desired` (or clears the anchor if `desired` is empty/`nullptr`) and pushes any displaced control block onto the calling processor's NUMA-local lock-free GC staging queue.
* **Restrictions & Guarantees:**
  * Takes `desired` by value (pass via `std::move()` to transfer ownership directly).
  * If the caller holds the sole reference to `desired` (`persistent_ref_count == 1`) and `anchor_state == 0`, transitions `anchor_state` to `1` without an extra atomic CAS before executing `active_control_block.exchange()`.
  * A control block may only be anchored in one `RundownSharedPtr<T>` at a time and cannot be re-stored until any previous displacement has completed epoch reclamation in `SharedPtrGarbageCollector`.
* **Parameters:**
  * `desired`: The `PersistentSharedPtr<T>` to install into the anchor, or `nullptr` / `PersistentSharedPtr<T>()` to clear the anchor.
* **Return Value:**
  * `true` if the store succeeded (or if clearing an already empty anchor); `false` if `desired` was rejected because its control block is already anchored or awaiting GC reclamation.

---

## 4. Batched Rundown Guard (`BatchedRundownGuard<T>`)

`BatchedRundownGuard<T>` is a stack-only RAII guard returned by `RundownSharedPtr<T>::acquire_batched()` that holds an acquired epoch rundown slot for its lifetime.

### Constructors & Destructor

```cpp
FORCE_INLINE BatchedRundownGuard() noexcept;
~BatchedRundownGuard() noexcept;
```

* **Purpose:**
  * **Default Constructor:** Constructs an empty guard holding no rundown slot and pinning no control block (`safe_block == nullptr`).
  * **Destructor:** Automatically releases the underlying `ScopedRundownGuard` (via `release_exclusive()` or `release_shared()`) when the guard leaves scope.
* **Restrictions:**
  * Non-copyable and non-movable (`operator=` and copy constructor are deleted). Must remain stack-bound within the calling thread.
* **Parameters:** None.
* **Return Value:** None.

### `read`

```cpp
template <typename F>
FORCE_INLINE decltype(auto) read(F&& func) const noexcept(std::is_nothrow_invocable_v<F, T*>);
```

* **Purpose:** Invokes `func` with a raw `T*` pointer to the pinned payload (or `nullptr` if empty) with zero atomic operations.
* **Restrictions:**
  * `func` **must not** return a raw pointer or C++ reference (enforced via `static_assert`), and must not cache the `T*` pointer beyond the lifetime of the owning `BatchedRundownGuard<T>`.
* **Parameters:**
  * `func`: A callable or lambda accepting `T*` (or `const T*`).
* **Return Value:**
  * Forwards the by-value return result of `func`.

---

## 5. Persistent Reference-Counted Handle (`PersistentSharedPtr<T>`)

`PersistentSharedPtr<T>` is a standard atomic reference-counted smart pointer returned by `MakeRundownShared`, `MakeRundownSharedNuma`, and `RundownSharedPtr<T>::load_persistent()`. It keeps the managed object alive across arbitrary thread boundaries and blocking operations completely outside the epoch rundown pipeline.

### Constructors & Destructor

#### `PersistentSharedPtr::PersistentSharedPtr` (Default & `nullptr` Constructors)

```cpp
constexpr PersistentSharedPtr() noexcept;
constexpr PersistentSharedPtr(decltype(nullptr)) noexcept;
```

* **Purpose:** Initializes an empty handle managing no control block (`get() == nullptr`).
* **Parameters:** None (or `nullptr`).
* **Return Value:** None.

#### `PersistentSharedPtr::PersistentSharedPtr` (Copy Constructor)

```cpp
PersistentSharedPtr(const PersistentSharedPtr& other) noexcept;
```

* **Purpose:** Shares persistent ownership of the object managed by `other` by atomically incrementing `persistent_ref_count` (`memory_order_acq_rel`).
* **Restrictions:**
  * Includes overflow protection: invokes `rundown_fatal_panic("Persistent reference count overflow")` (calling any registered `RundownFatalErrorHandler` prior to `std::abort()`) if `persistent_ref_count` overflows `INT32_MAX`.
  * Copying distinct `PersistentSharedPtr<T>` instances referencing the same control block is thread-safe; mutating a single `PersistentSharedPtr<T>` instance concurrently from multiple threads requires external synchronization.
* **Parameters:**
  * `other`: Source handle to copy from.
* **Return Value:** None.

#### `PersistentSharedPtr::PersistentSharedPtr` (Move Constructor)

```cpp
PersistentSharedPtr(PersistentSharedPtr&& other) noexcept;
```

* **Purpose:** Transfers persistent ownership from `other` to `*this` without modifying atomic reference counts, leaving `other` empty (`nullptr`).
* **Parameters:**
  * `other`: Rvalue reference to the source handle.
* **Return Value:** None.

#### `PersistentSharedPtr::~PersistentSharedPtr` (Destructor)

```cpp
~PersistentSharedPtr() noexcept;
```

* **Purpose:** Atomically decrements `persistent_ref_count`. When the reference count drops to `0`, invokes `~T()` on the payload and recycles the control block into the calling thread's per-slot / per-NUMA-node block cache (or releases it to its 64 KB NUMA slab / heap if the collector is unloading).
* **Restrictions:**
  * Includes underflow protection: invokes `rundown_fatal_panic("Double free detected: persistent reference count dropped below zero")` if `persistent_ref_count` drops below `0`.
* **Parameters:** None.
* **Return Value:** None.

### Assignment Operators

```cpp
PersistentSharedPtr& operator=(const PersistentSharedPtr& other) noexcept;
PersistentSharedPtr& operator=(PersistentSharedPtr&& other) noexcept;
PersistentSharedPtr& operator=(decltype(nullptr)) noexcept;
```

* **Purpose:**
  * **Copy Assignment:** Shares ownership of `other`'s object via copy-and-swap and releases any previously managed object.
  * **Move Assignment:** Releases any currently managed object and transfers ownership from `other` to `*this` without extra atomic increments, leaving `other` empty.
  * **`nullptr` Assignment:** Releases any currently managed object and resets `*this` to empty.
* **Parameters:**
  * `other`: Source handle to copy or move from (or `nullptr`).
* **Return Value:**
  * Reference to `*this`.

### `swap`

```cpp
void swap(PersistentSharedPtr& other) noexcept;
```

* **Purpose:** Exchanges the control block pointers of `*this` and `other` without atomic reference-count operations.
* **Parameters:**
  * `other`: Handle to swap state with.
* **Return Value:**
  * None.

### `reset`

```cpp
void reset() noexcept;
```

* **Purpose:** Releases ownership of the currently managed object (destroying and recycling it if this was the final reference) and resets `*this` to empty (`nullptr`).
* **Parameters:**
  * None.
* **Return Value:**
  * None.

### `get`

```cpp
[[nodiscard]] FORCE_INLINE T* get() const noexcept;
```

* **Purpose:** Resolves and returns a raw pointer to the managed payload object via compile-time constant offset arithmetic (`control_block->get_payload()`), or `nullptr` if empty.
* **Restrictions:**
  * The returned pointer remains valid only while at least one `PersistentSharedPtr<T>` handle continues to own the object.
* **Parameters:**
  * None.
* **Return Value:**
  * A raw pointer `T*` to the managed payload object, or `nullptr` if `*this` is empty.

### `operator->` & `operator*`

```cpp
FORCE_INLINE T* operator->() const noexcept;
FORCE_INLINE T& operator*() const noexcept;
```

* **Purpose:** Provides direct pointer member access (`operator->`) and dereferencing (`operator*`) to the managed payload object.
* **Restrictions:**
  * Behavior is undefined if `*this` is empty (`get() == nullptr`); verify validity via `operator bool()` before dereferencing.
* **Parameters:**
  * None.
* **Return Value:**
  * `operator->`: Returns `T*` (`get()`).
  * `operator*`: Returns `T&` (`*get()`).

### `operator bool`

```cpp
FORCE_INLINE explicit operator bool() const noexcept;
```

* **Purpose:** Checks whether `*this` manages a valid control block and payload object.
* **Parameters:**
  * None.
* **Return Value:**
  * `true` if `*this` owns a non-null control block (`get() != nullptr`); `false` otherwise.

---

## 6. Epoch Rundown Garbage Collector (`SharedPtrGarbageCollector`)

`SharedPtrGarbageCollector` is the process-wide singleton that manages the two alternating 256-slot cache-aware rundown epoch shards (`rundown_refs[2]`), per-NUMA-node lock-free retirement SLists (`NumaQuiescentSList`), per-slot and per-node control-block recycling caches, and 64 KB NUMA bump slabs.

### `get_instance`

```cpp
static SharedPtrGarbageCollector& get_instance() noexcept;
```

* **Purpose:** Returns a reference to the process-wide `SharedPtrGarbageCollector` singleton, constructing it and discovering the host's NUMA topology on first access.
* **Parameters:**
  * None.
* **Return Value:**
  * Reference to the global `SharedPtrGarbageCollector` instance.

### `set_wake_callback`

```cpp
void set_wake_callback(void (*callback)(),
                       int32_t watermark = 1000) noexcept;
```

* **Purpose:** Registers (or clears) an optional client-provided notification callback triggered when the approximate count of displaced objects awaiting collection reaches or exceeds `watermark`.
* **Restrictions & Guarantees:**
  * **Callback Execution Context:** `callback` is invoked directly on the writer thread inside `RundownSharedPtr<T>::store()` (or anchor destruction) when `queue_depth` crosses `watermark`. To keep `store()` wait/lock-free, `callback` should only signal a background GC worker thread (e.g., via an atomic flag + condition variable or `std::atomic::notify_one()`) and **must not** call `process_garbage_pipeline()` synchronously on the writer thread if the caller is currently inside a `read()` scope on the same thread.
  * Coalesces wake signals: `callback` is fired at most once per accumulation cycle until `process_garbage_pipeline()` resets `wake_requested`.
  * Pass `nullptr` before joining your background GC worker thread on shutdown.
* **Parameters:**
  * `callback`: Function pointer `void (*)()` invoked when pending retired objects reach `watermark`, or `nullptr` to disable wake notifications.
  * `watermark`: Queue depth threshold required to trigger `callback` (defaults to `1000`).
* **Return Value:**
  * None.

### `process_garbage_pipeline`

```cpp
void process_garbage_pipeline(bool is_flush = false) noexcept;
```

* **Purpose:** Executes the two-phase epoch advancement and batch reclamation pipeline:
  1. Checks whether any previously staged batches in `pending_free_lists` have drained all active readers on their outgoing epoch shard (`rundown_refs[pending_epoch & 1]`) and, if drained, resets their `anchor_state` to `0` and drops their baseline reference.
  2. Atomically detaches all newly retired blocks across the fallback SList and all per-NUMA-node SLists in $O(\text{nodes})$ time, advances the 64-bit `active_epoch` with `memory_order_seq_cst`, and loops back once to immediately reclaim the batch if the outgoing shard is already idle (including a brief 8-pause hardware grace check when `is_flush == false`).
* **Restrictions & Guarantees:**
  * Safe to call concurrently from any thread; serialized internally via `is_gc_active`.
  * **Head-of-Line Blocking Protection (`is_flush == false`):** If a reader is still executing in the outgoing epoch shard, `process_garbage_pipeline(false)` leaves the batch in `pending_free_lists`, resets `queue_depth` and `wake_requested` so future watermark batches can re-signal the worker, releases `is_gc_active`, and returns immediately without spinning.
  * **Synchronous Flush (`is_flush == true`):** Waits to acquire `is_gc_active`, spins with `AdaptiveBackoff` until all pending epochs and cascading retired objects drain, flushes all per-slot and per-node control-block caches back to their origin allocators/slabs, and tightens the active reader slot scan bound (`recompute_active_slot_bound()`).
  * **Deadlock Warning:** Never call `process_garbage_pipeline(true)` from inside a `RundownSharedPtr<T>::read()` lambda or while the calling thread holds a live `BatchedRundownGuard<T>`, as the thread will spin waiting for its own rundown slot to drain.
* **Parameters:**
  * `is_flush`: `false` *(default)* for non-blocking background GC worker passes; `true` for synchronous full reclamation and cache flushing.
* **Return Value:**
  * None.

### Process Teardown (`~SharedPtrGarbageCollector`)

```cpp
~SharedPtrGarbageCollector() noexcept;
```

* **Purpose:** Automatically invoked at process exit. Sets `g_rundown_gc_alive` to `0` (disabling block-cache recycling), executes `process_garbage_pipeline(true)` to drain and free all pending objects and block caches, reclaims any final straggler SList entries via `reclaim_remaining_slists()`, and unmaps all per-NUMA-node `NumaQuiescentSList` shards and active 64 KB NUMA slabs.
* **Restrictions:**
  * Stop and join any application-spawned background GC threads that call `process_garbage_pipeline()` before `main()` returns.

---

## 7. Pluggable Fatal Error Handling

`rundown_shared_ptr.h` continuously validates internal reference-counting and allocator invariants (persistent reference-count overflow at `INT32_MAX`, double-free/underflow when `persistent_ref_count` drops below `0`, 64 KB NUMA slab `active_blocks` underflow, and shared reader slot `overflow_slot_refs` underflow).

### `RundownFatalErrorHandler`

```cpp
using RundownFatalErrorHandler = void(*)(const char* reason) noexcept;
```

* **Purpose:** Function pointer signature for custom process-wide fatal error notifications invoked prior to `std::abort()`.

### `SetRundownPtrErrorHandler`

```cpp
inline void SetRundownPtrErrorHandler(RundownFatalErrorHandler handler) noexcept;
```

* **Purpose:** Registers (or clears) a custom process-wide callback invoked by `rundown_fatal_panic()` when an unrecoverable invariant violation is detected.
* **Restrictions & Guarantees:**
  * Thread-safe; stores `handler` into `g_rundown_fatal_error_handler` with `std::memory_order_release`.
  * Pass `nullptr` to restore the default `std::cerr` diagnostic logging behavior.
* **Parameters:**
  * `handler`: Custom `noexcept` callback function pointer, or `nullptr`.
* **Return Value:**
  * None.

---
