# `atomic_shared_ptr.h` Client API Reference

`atomic_shared_ptr.h` provides a lock-free, thread-safe shared pointer implementation for high-throughput user-mode applications across Windows, Linux, and macOS (x64 and ARM64). Based on Anthony Williams' Split Reference Counting algorithm, it pairs a 128-bit atomic publication anchor (`AtomicSharedPtr<T>`) with a zero-indirection thread-local handle (`LocalSharedPtr<T>`) and a wait-free 64-bit packed control block (`ControlBlock<T>`), achieving full `std::atomic<std::shared_ptr<T>>` semantic parity without mutexes or lock tables.

## Table of Contents

1. [Component & Concurrency Summary](#1-component--concurrency-summary)
2. [Single-Allocation Cache-Aligned Factory](#2-single-allocation-cache-aligned-factory)
   * [`MakeLocalShared`](#makelocalshared)
3. [Shared Lock-Free Anchor (`AtomicSharedPtr<T>`)](#3-shared-lock-free-anchor-atomicsharedptrt)
   * [Constructors & Destructor](#constructors--destructor)
   * [Assignment & Conversion Operators](#assignment--conversion-operators)
   * [`is_lock_free`](#is_lock_free)
   * [`load`](#load)
   * [`store`](#store)
   * [`exchange`](#exchange)
   * [`compare_exchange_weak`](#compare_exchange_weak)
   * [`compare_exchange_strong`](#compare_exchange_strong)
4. [Thread-Local Reference-Counted Handle (`LocalSharedPtr<T>`)](#4-thread-local-reference-counted-handle-localsharedptrt)
   * [Constructors & Destructor](#constructors--destructor-1)
   * [Assignment Operators](#assignment-operators)
   * [`swap`](#swap)
   * [`reset`](#reset)
   * [`use_count`](#use_count)
   * [`get`](#get)
   * [`operator->` & `operator*`](#operator--operator)
   * [`operator bool`](#operator-bool)
5. [Pluggable Fatal Error Handling](#5-pluggable-fatal-error-handling)
   * [`FatalErrorHandler`](#fatalerrorhandler)
   * [`SetAtomicPtrErrorHandler`](#setatomicptrerrorhandler)
   * [`fatal_panic`](#fatal_panic)

---

## 1. Component & Concurrency Summary

Both `AtomicSharedPtr<T>` and `ControlBlock<T>` are automatically aligned to `CACHE_LINE_SIZE` (64 bytes on x64; 128 bytes on ARM64 and Apple Silicon) to eliminate false sharing across adjacent anchors, control blocks, and embedded payloads.

| Component / Function | Role | Exception Specification | Progress Guarantee |
| :--- | :--- | :--- | :--- |
| **`MakeLocalShared`** | Single-allocation cache-isolated object factory | Propagates `std::bad_alloc` and `T` ctor exceptions | Heap-allocated (single aligned allocation) |
| **`AtomicSharedPtr<T>::is_lock_free`** | Queries hardware 128-bit lock-free support | `noexcept` | Wait-free |
| **`AtomicSharedPtr<T>::load`** | Atomically checks out a `LocalSharedPtr<T>` handle | `noexcept` | Lock-free (128-bit CAS + wait-free 64-bit `fetch_add`) |
| **`AtomicSharedPtr<T>::store` (`LocalSharedPtr`)** | Atomically replaces anchor and reconciles old counts | `noexcept` | Lock-free (128-bit CAS + wait-free 64-bit `fetch_add`) |
| **`AtomicSharedPtr<T>::store` (`T*`)** | Wraps raw pointer and replaces anchor | Propagates `std::bad_alloc` | Lock-free after control block allocation |
| **`AtomicSharedPtr<T>::exchange` (`LocalSharedPtr`)** | Atomically swaps anchor and returns displaced handle | `noexcept` | Lock-free (128-bit CAS + wait-free 64-bit `fetch_add`) |
| **`AtomicSharedPtr<T>::exchange` (`T*`)** | Wraps raw pointer, swaps anchor, returns old handle | Propagates `std::bad_alloc` | Lock-free after control block allocation |
| **`AtomicSharedPtr<T>::compare_exchange_weak`** | Single-attempt conditional anchor replacement | `noexcept` | Wait-free on pointer mismatch; lock-free on reload |
| **`AtomicSharedPtr<T>::compare_exchange_strong`** | Spurious-immune conditional anchor replacement | `noexcept` | Lock-free (hoisted refcount increment; retries spurious reader collisions) |
| **`LocalSharedPtr<T>` (Copy / Move / Destroy)** | Thread-local handle with cached `T*` payload pointer | `noexcept` (except `T*` ctor/reset) | Wait-free (`0` atomic ops on move/deref; single 64-bit `fetch_add` on copy/destroy) |
| **`SetAtomicPtrErrorHandler`** | Registers custom invariant-violation panic callback | `noexcept` | Wait-free |

---

## 2. Single-Allocation Cache-Aligned Factory

### `MakeLocalShared`

```cpp
template <typename T, typename... Args>
[[nodiscard]] LocalSharedPtr<T> MakeLocalShared(Args&&... args);
```

* **Purpose:** Allocates and constructs a `ControlBlock<T>` and an embedded payload object of type `T` within a single cache-aligned memory block, returning a `LocalSharedPtr<T>` with initial split reference counts `{internal = 1, external = 0}`.
* **Restrictions & Guarantees:**
  * Aligns the allocation to `std::max(CACHE_LINE_SIZE, alignof(T))` (`_aligned_malloc` on Windows, `posix_memalign` on POSIX).
  * Places `ControlBlock<T>` at offset `0` and embeds `T` at a cache-aligned offset (`(sizeof(ControlBlock<T>) + alloc_align - 1) & ~(alloc_align - 1)`), preventing atomic reference-count updates on `ControlBlock<T>::counts` from invalidating the CPU cache line holding the payload `T`.
  * Pre-populates `LocalSharedPtr<T>::ptr` with the resolved payload address so subsequent `get()`, `operator->()`, and `operator*()` calls execute with zero control-block pointer indirection.
  * **Exception Safety:** Throws `std::bad_alloc` if memory allocation fails. If `T`'s constructor throws an exception, the aligned memory block is automatically freed via `ControlBlock<T>::operator delete` before the exception propagates.
* **Parameters:**
  * `args`: Constructor arguments perfectly forwarded to `T`'s constructor.
* **Return Value:**
  * A `LocalSharedPtr<T>` managing the newly constructed object.

---

## 3. Shared Lock-Free Anchor (`AtomicSharedPtr<T>`)

`AtomicSharedPtr<T>` is a cache-line-isolated (`alignas(CACHE_LINE_SIZE)`) shared publication anchor wrapping a 16-byte `CountedPtr<T>` state (`{ControlBlock<T>* cb, int32_t external_count, uint32_t generation}`). Multiple threads may concurrently invoke `load()`, `store()`, `exchange()`, `compare_exchange_weak()`, and `compare_exchange_strong()` on the same `AtomicSharedPtr<T>` instance without external synchronization.

### Constructors & Destructor

#### `AtomicSharedPtr::AtomicSharedPtr` (Default Constructor)

```cpp
constexpr AtomicSharedPtr() noexcept = default;
```

* **Purpose:** Initializes an empty anchor (`{cb = nullptr, external_count = 0, generation = 0}`).
* **Restrictions:**
  * `AtomicSharedPtr<T>` is neither copyable nor movable (`AtomicSharedPtr(const AtomicSharedPtr&) = delete` and `operator=(const AtomicSharedPtr&) = delete`).
* **Parameters:**
  * None.
* **Return Value:**
  * None.

#### `AtomicSharedPtr::AtomicSharedPtr` (`LocalSharedPtr` Checkout Constructor)

```cpp
explicit AtomicSharedPtr(LocalSharedPtr<T> desired) noexcept;
```

* **Purpose:** Initializes the anchor by adopting baseline ownership directly from `desired` (`external_count = 0, generation = 0`) without performing any atomic reference-count operations.
* **Restrictions:**
  * Pass `desired` via `std::move()` to avoid an unnecessary copy increment/decrement pair.
* **Parameters:**
  * `desired`: The `LocalSharedPtr<T>` whose control block ownership is transferred into the anchor.
* **Return Value:**
  * None.

#### `AtomicSharedPtr::AtomicSharedPtr` (Raw Pointer Constructor)

```cpp
explicit AtomicSharedPtr(T* ptr);
```

* **Purpose:** Allocates a standalone `ControlBlock<T>` wrapping `ptr` (if non-null) and initializes the anchor to manage it.
* **Restrictions & Guarantees:**
  * `ptr` must be a valid heap-allocated pointer suitable for `delete ptr` (or `nullptr`).
  * **Exception Safety:** If `new ControlBlock<T>(ptr)` throws `std::bad_alloc`, `delete ptr` is automatically invoked before rethrowing to prevent memory leaks.
* **Parameters:**
  * `ptr`: Raw pointer to take ownership of, or `nullptr`.
* **Return Value:**
  * None.

#### `AtomicSharedPtr::~AtomicSharedPtr` (Destructor)

```cpp
~AtomicSharedPtr();
```

* **Purpose:** Releases the currently anchored control block (if non-null) by atomically applying `{int_add = -1, ext_add = old_val.external_count}` via `apply_counts()` with `std::memory_order_acq_rel`, destroying the control block and payload if both `internal` and `external` counts reach zero.
* **Restrictions:**
  * No other threads may concurrently access the `AtomicSharedPtr<T>` instance while its destructor is executing. Existing `LocalSharedPtr<T>` handles previously checked out from the anchor remain valid and independently keep the payload alive.
* **Parameters:**
  * None.
* **Return Value:**
  * None.

### Assignment & Conversion Operators

#### `AtomicSharedPtr::operator=`

```cpp
void operator=(LocalSharedPtr<T> desired) noexcept;
```

* **Purpose:** Atomically replaces the anchored object with `desired` by forwarding `std::move(desired)` to `store()`.
* **Parameters:**
  * `desired`: New `LocalSharedPtr<T>` handle to publish into the anchor.
* **Return Value:**
  * None.

#### `AtomicSharedPtr::operator LocalSharedPtr<T>`

```cpp
operator LocalSharedPtr<T>() const noexcept;
```

* **Purpose:** Implicitly checks out a `LocalSharedPtr<T>` snapshot of the currently anchored object by calling `load(std::memory_order_acquire)`.
* **Parameters:**
  * None.
* **Return Value:**
  * A `LocalSharedPtr<T>` owning an independent internal reference to the active object, or an empty handle if the anchor is null.

### `is_lock_free`

```cpp
[[nodiscard]] bool is_lock_free() const noexcept;
```

* **Purpose:** Reports whether 128-bit atomic operations on this anchor execute lock-free in hardware (`CMPXCHG16B` on x64, `CASP` / `LDXP`+`STXP` on ARM64).
* **Parameters:**
  * None.
* **Return Value:**
  * `true` if 128-bit atomics are lock-free on the target platform/compiler; `false` otherwise.

### `load`

```cpp
[[nodiscard]] LocalSharedPtr<T> load(std::memory_order order = std::memory_order_acquire) const noexcept;
```

* **Purpose:** Atomically checks out a thread-local `LocalSharedPtr<T>` handle referencing the currently anchored object.
* **Restrictions & Guarantees:**
  * **Two-Phase Split Reference Checkout:**
    1. Increments `external_count` inside the 128-bit `CountedPtr<T>` anchor via 128-bit CAS (using `AdaptiveBackoff` on contention), preventing any concurrent writer from destroying the `ControlBlock<T>` while the checkout is in progress.
    2. Converts the transient external checkout into a pinned internal reference on the control block via `apply_counts(cb->counts, +1, -1, std::memory_order_relaxed)`.
  * **Overflow Flush Protection (`EXT_COUNT_FLUSH_THRESHOLD`):** If `external_count` in the anchor reaches $2^{29}$ (`1 << 29`) under sustained read-only load without intervening writes, the reader attempts a 128-bit CAS to reset the anchor's `external_count` to `0` (incrementing `generation`) and flushes `+expected_flush.external_count` into `cb->counts.external` while holding its pinned internal reference. If `external_count` ever reaches `INT32_MAX`, triggers `fatal_panic()`.
  * **Memory Order Rules:** `std::memory_order_relaxed` is automatically promoted to `std::memory_order_acquire` so control-block dereferences are always safely synchronized with publishing writers. Passing `std::memory_order_release` or `std::memory_order_acq_rel` is invalid and triggers `fatal_panic("Invalid load memory order")`.
* **Parameters:**
  * `order`: Memory ordering constraint (`acquire`, `consume`, `seq_cst`, or `relaxed` which promotes to `acquire`). Defaults to `std::memory_order_acquire`.
* **Return Value:**
  * A `LocalSharedPtr<T>` managing the checked-out object, or an empty `LocalSharedPtr<T>` (`nullptr`) if the anchor is empty.

### `store`

```cpp
void store(LocalSharedPtr<T> desired,
           std::memory_order order = std::memory_order_release) noexcept;

void store(T*                ptr,
           std::memory_order order = std::memory_order_release);
```

* **Purpose:** Atomically replaces the currently anchored object with `desired` (or a newly constructed `LocalSharedPtr<T>(ptr)`), increments the anchor's 32-bit `generation` counter to prevent ABA collisions, and reconciles the displaced anchor's reference counts.
* **Restrictions & Guarantees:**
  * Adopts baseline internal ownership directly from `desired` (`new_val.external_count = 0, new_val.generation = old_val.generation + 1`).
  * On the displaced control block `old_val.cb` (if non-null), atomically applies `{int_add = -1, ext_add = old_val.external_count}` with `std::memory_order_acq_rel`, deleting `old_val.cb` if both `internal` and `external` counts reach zero.
  * **Memory Order Normalization:** Since `store()` both publishes `desired` and acquires responsibility for reconciling/destroying the displaced `old_val.cb`, the 128-bit CAS order is normalized via `effective_rmw_order(order)` to at least `std::memory_order_acq_rel` (or `std::memory_order_seq_cst` if requested).
  * **Exception Safety:** The `LocalSharedPtr<T>` overload is `noexcept`. The raw pointer `T*` overload allocates a `ControlBlock<T>`; if `std::bad_alloc` is thrown, `delete ptr` is called automatically and the anchor remains unmodified.
* **Parameters:**
  * `desired` / `ptr`: The new `LocalSharedPtr<T>` handle or raw pointer `T*` to install into the anchor.
  * `order`: Memory ordering constraint (defaults to `std::memory_order_release`, normalized internally to `acq_rel` or `seq_cst`).
* **Return Value:**
  * None.

### `exchange`

```cpp
[[nodiscard]] LocalSharedPtr<T> exchange(LocalSharedPtr<T> desired,
                                         std::memory_order order = std::memory_order_acq_rel) noexcept;

[[nodiscard]] LocalSharedPtr<T> exchange(T*                ptr,
                                         std::memory_order order = std::memory_order_acq_rel);
```

* **Purpose:** Atomically replaces the currently anchored object with `desired` (or `LocalSharedPtr<T>(ptr)`) and returns the displaced object as a `LocalSharedPtr<T>`.
* **Restrictions & Guarantees:**
  * Transfers the displaced anchor's baseline internal reference (`1`) directly into the returned `LocalSharedPtr<T>` without decrementing `internal`.
  * If `old_val.external_count > 0`, flushes `{int_add = 0, ext_add = old_val.external_count}` into `old_val.cb->counts` with `std::memory_order_acq_rel` before returning the handle.
  * Normalizes `order` via `effective_rmw_order(order)` to at least `std::memory_order_acq_rel` (or `std::memory_order_seq_cst`).
  * **Exception Safety:** The `LocalSharedPtr<T>` overload is `noexcept`. The `T*` overload deletes `ptr` and leaves the anchor unchanged if `ControlBlock<T>` allocation throws `std::bad_alloc`.
* **Parameters:**
  * `desired` / `ptr`: The new `LocalSharedPtr<T>` handle or raw pointer `T*` to install into the anchor.
  * `order`: Memory ordering constraint (defaults to `std::memory_order_acq_rel`).
* **Return Value:**
  * A `LocalSharedPtr<T>` owning the previously anchored object, or an empty `LocalSharedPtr<T>` (`nullptr`) if the anchor was empty.

### `compare_exchange_weak`

```cpp
// (1) Lvalue desired, dual memory order
[[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&       expected,
                                         const LocalSharedPtr<T>& desired,
                                         std::memory_order        succ,
                                         std::memory_order        fail) noexcept;

// (2) Rvalue desired, dual memory order
[[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&  expected,
                                         LocalSharedPtr<T>&& desired,
                                         std::memory_order   succ,
                                         std::memory_order   fail) noexcept;

// (3) Lvalue desired, single memory order
[[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&       expected,
                                         const LocalSharedPtr<T>& desired,
                                         std::memory_order        order = std::memory_order_acq_rel) noexcept;

// (4) Rvalue desired, single memory order
[[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&  expected,
                                         LocalSharedPtr<T>&& desired,
                                         std::memory_order   order = std::memory_order_acq_rel) noexcept;
```

* **Purpose:** Conditionally replaces the anchored object with `desired` if the anchor's current `ControlBlock<T>*` matches `expected.cb`. Allowed to fail spuriously if a concurrent reader increments `external_count` or flushes counts on the anchor during the CAS window.
* **Restrictions & Guarantees:**
  * **On Success (`returns true`):**
    * The anchor points to `desired`'s control block with `{external_count = 0, generation = current.generation + 1}`.
    * `expected` is not modified.
    * For **Lvalue `desired` (1, 3)**: `desired` retains its ownership and the anchor acquires an additional internal reference (`+1 internal`).
    * For **Rvalue `desired` (2, 4)**: Ownership is moved directly from `desired` into the anchor without any atomic increment on `desired.cb->counts`, leaving `desired` empty (`nullptr`).
    * The displaced control block `current.cb` (if non-null) has `{int_add = -1, ext_add = current.external_count}` applied with `std::memory_order_acq_rel` and is deleted if both counts reach zero.
  * **On Failure (`returns false`):**
    * If the failure was due to a true pointer mismatch (`current.cb != expected.cb`), `expected` is updated to hold a checked-out `LocalSharedPtr<T>` of the current anchor (or reset to empty if `current.cb == nullptr`).
    * If the failure was spurious (`current.cb == expected.cb`, caused by a concurrent reader changing `external_count` or `generation`), `expected` remains valid and unchanged.
    * For **Rvalue `desired` (2, 4)**: `desired` is **not** moved from or modified on failure, preserving its original ownership and reference count.
  * **Memory Order Normalization:** `succ` is normalized to at least `std::memory_order_acq_rel`. Invalid failure orders `acq_rel` and `release` are mapped to `acquire` and `relaxed` respectively, and `relaxed` failure loads are promoted to `acquire` when checking out `expected`.
* **Parameters:**
  * `expected`: In/out reference to the `LocalSharedPtr<T>` expected to match the anchor's current control block. Updated to the actual anchored object on pointer mismatch.
  * `desired`: Lvalue or rvalue `LocalSharedPtr<T>` to install on match.
  * `succ` / `fail` / `order`: Memory ordering constraints for success and failure paths (defaults to `std::memory_order_acq_rel`).
* **Return Value:**
  * `true` if the anchor was successfully swapped; `false` on pointer mismatch or spurious 128-bit CAS collision.

### `compare_exchange_strong`

```cpp
// (1) Lvalue desired, dual memory order
[[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&       expected,
                                           const LocalSharedPtr<T>& desired,
                                           std::memory_order        succ,
                                           std::memory_order        fail) noexcept;

// (2) Rvalue desired, dual memory order
[[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&  expected,
                                           LocalSharedPtr<T>&& desired,
                                           std::memory_order   succ,
                                           std::memory_order   fail) noexcept;

// (3) Lvalue desired, single memory order
[[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&       expected,
                                           const LocalSharedPtr<T>& desired,
                                           std::memory_order        order = std::memory_order_acq_rel) noexcept;

// (4) Rvalue desired, single memory order
[[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&  expected,
                                           LocalSharedPtr<T>&& desired,
                                           std::memory_order   order = std::memory_order_acq_rel) noexcept;
```

* **Purpose:** Atomically compares the anchor's `ControlBlock<T>*` against `expected.cb` and replaces the anchor with `desired` on match, retrying internally on spurious `external_count` or `generation` updates so it only returns `false` when the anchored `ControlBlock<T>*` genuinely differs from `expected.cb`.
* **Restrictions & Guarantees:**
  * **Spurious-Failure Immunity:** Concurrent readers calling `load()` (which mutates `external_count` and `generation` in the 128-bit anchor while `current.cb == expected.cb`) never cause `compare_exchange_strong()` to return `false`. When a 128-bit CAS fails with `current.cb == expected.cb`, the loop retries immediately without backoff while the cache line is already held in Modified (M) state by the failed `CMPXCHG16B`.
  * **Hoisted Reference Increment (Lvalue Overloads 1, 3):** If `current.cb == expected.cb` on initial inspection, `desired.cb->counts` is incremented at most **once** outside the retry loop (and rolled back via `apply_counts(desired_cb->counts, -1, 0, acq_rel)` only if a concurrent writer replaces `current.cb` before the CAS commits).
  * **Move-Only-On-Success Guarantee (Rvalue Overloads 2, 4):** Ownership of `desired` is transferred into the anchor strictly upon 128-bit CAS success. If `compare_exchange_strong()` returns `false`, `desired` remains intact and unmodified with zero atomic operations performed on `desired.cb->counts`.
  * **On Failure (`returns false`):** `expected` is updated to a checked-out `LocalSharedPtr<T>` of the actual anchored object (or reset to empty if the anchor is `nullptr`).
* **Parameters:**
  * `expected`: In/out reference to the `LocalSharedPtr<T>` expected to match the anchor's current control block. Updated to the actual anchored object on failure.
  * `desired`: Lvalue or rvalue `LocalSharedPtr<T>` to install on match.
  * `succ` / `fail` / `order`: Memory ordering constraints for success and failure paths (defaults to `std::memory_order_acq_rel`).
* **Return Value:**
  * `true` if the anchor was replaced; `false` if and only if the anchor's control block pointer did not match `expected.cb`.

---

## 4. Thread-Local Reference-Counted Handle (`LocalSharedPtr<T>`)

`LocalSharedPtr<T>` is a lightweight handle holding both the `ControlBlock<T>*` pointer and a cached `T*` payload pointer. Like `std::shared_ptr<T>`, distinct `LocalSharedPtr<T>` instances referencing the same control block may be copied, moved, read, and destroyed concurrently across threads, while concurrent mutation of a *single* `LocalSharedPtr<T>` variable requires synchronization or `AtomicSharedPtr<T>`.

### Constructors & Destructor

#### `LocalSharedPtr::LocalSharedPtr` (Default & `nullptr` Constructors)

```cpp
constexpr LocalSharedPtr() noexcept;
constexpr LocalSharedPtr(std::nullptr_t) noexcept;
```

* **Purpose:** Initializes an empty handle (`cb == nullptr, ptr == nullptr`).
* **Parameters:** None (or `nullptr`).
* **Return Value:** None.

#### `LocalSharedPtr::LocalSharedPtr` (Raw Pointer Constructor)

```cpp
explicit LocalSharedPtr(T* p);
```

* **Purpose:** Takes ownership of a heap-allocated raw pointer `p` (if non-null) by allocating a standalone `ControlBlock<T>(p)` with initial split reference counts `{internal = 1, external = 0}`.
* **Restrictions & Guarantees:**
  * `p` must be a valid pointer allocated with single-object `new` (or `nullptr`). Prefer `MakeLocalShared<T>()` whenever possible to co-locate the control block and payload in a single cache-isolated allocation.
  * **Exception Safety:** If `new ControlBlock<T>(p)` throws `std::bad_alloc`, `delete p` is automatically invoked and `ptr` is cleared before the exception propagates.
* **Parameters:**
  * `p`: Raw pointer to take ownership of, or `nullptr`.
* **Return Value:** None.

#### `LocalSharedPtr::LocalSharedPtr` (Copy Constructor)

```cpp
LocalSharedPtr(const LocalSharedPtr& other) noexcept;
```

* **Purpose:** Shares ownership of the object managed by `other`, copying both `cb` and the cached `ptr` and atomically incrementing `internal` by `+1` via `apply_counts(cb->counts, 1, 0, std::memory_order_relaxed)`.
* **Restrictions:**
  * Calls `fatal_panic("Internal count overflow")` if `internal` exceeds `INT32_MAX`.
* **Parameters:**
  * `other`: Source handle to copy.
* **Return Value:** None.

#### `LocalSharedPtr::LocalSharedPtr` (Move Constructor)

```cpp
LocalSharedPtr(LocalSharedPtr&& other) noexcept;
```

* **Purpose:** Transfers ownership of `other`'s control block and cached `T*` pointer to `*this` with zero atomic operations, leaving `other` empty (`nullptr`).
* **Parameters:**
  * `other`: Rvalue reference to the source handle.
* **Return Value:** None.

#### `LocalSharedPtr::~LocalSharedPtr` (Destructor)

```cpp
~LocalSharedPtr();
```

* **Purpose:** Atomically decrements the control block's `internal` count by `-1` via `apply_counts(cb->counts, -1, 0, std::memory_order_acq_rel)`. If both `internal` and `external` counts reach `0`, destroys the payload object and frees the control block.
* **Parameters:** None.
* **Return Value:** None.

### Assignment Operators

```cpp
LocalSharedPtr& operator=(const LocalSharedPtr& other) noexcept;
LocalSharedPtr& operator=(LocalSharedPtr&& other) noexcept;
LocalSharedPtr& operator=(std::nullptr_t) noexcept;
```

* **Purpose:**
  * **Copy Assignment:** Shares ownership with `other` via copy-and-swap and releases any previously held reference. Safe against self-assignment (`this == &other`).
  * **Move Assignment:** Transfers ownership from `other` via move-and-swap without atomic increments, releasing any previously held reference and leaving `other` empty. Safe against self-assignment.
  * **`nullptr` Assignment:** Releases any currently managed object via `reset()` and sets `*this` to empty.
* **Parameters:**
  * `other`: Source handle to copy or move from (or `nullptr`).
* **Return Value:**
  * Reference to `*this`.

### `swap`

```cpp
void swap(LocalSharedPtr& other) noexcept;
```

* **Purpose:** Exchanges both the `ControlBlock<T>*` and cached `T*` pointers between `*this` and `other` with zero atomic operations.
* **Parameters:**
  * `other`: Handle to swap state with.
* **Return Value:**
  * None.

### `reset`

```cpp
void reset() noexcept;
void reset(T* p);
```

* **Purpose:**
  * `reset()`: Releases ownership of the managed object (destroying it if this was the last internal and external reference) and resets `*this` to empty (`nullptr`).
  * `reset(T* p)`: Replaces the managed object with a newly allocated `ControlBlock<T>` wrapping `p` (or resets to empty if `p == nullptr`). If `ControlBlock<T>` allocation throws `std::bad_alloc`, `delete p` is called and `*this` remains unchanged.
* **Parameters:**
  * `p` *(optional)*: Raw pointer to take ownership of.
* **Return Value:**
  * None.

### `use_count`

```cpp
[[nodiscard]] long use_count() const noexcept;
```

* **Purpose:** Returns an approximate snapshot of the active reference count (`internal + max(0, external)`) for diagnostics, testing, and debugging.
* **Restrictions:**
  * Loaded with `std::memory_order_relaxed`. Since transient reader checkouts increment `external_count` inside `AtomicSharedPtr<T>` and convert them via `{+1 internal, -1 external}` in `ControlBlock<T>::counts` (which can momentarily make `c.external` negative before the anchor flushes or is replaced), negative `c.external` values are clamped to `0L` when computing the approximate count.
* **Parameters:**
  * None.
* **Return Value:**
  * Approximate reference count as `long`, or `0` if `*this` is empty.

### `get`

```cpp
[[nodiscard]] FORCE_INLINE T* get() const noexcept;
```

* **Purpose:** Returns the cached raw pointer `ptr` to the managed payload object with zero control-block indirection, or `nullptr` if empty.
* **Parameters:**
  * None.
* **Return Value:**
  * Raw pointer `T*` to the managed object, or `nullptr`.

### `operator->` & `operator*`

```cpp
FORCE_INLINE T* operator->() const noexcept;
FORCE_INLINE T& operator*() const noexcept;
```

* **Purpose:** Provides direct zero-indirection member access (`operator->`) and dereferencing (`operator*`) using the cached `T*` pointer.
* **Restrictions:**
  * Behavior is undefined if `*this` is empty (`get() == nullptr`); check validity via `operator bool()` before dereferencing.
* **Parameters:**
  * None.
* **Return Value:**
  * `operator->`: Returns `T*` (`ptr`).
  * `operator*`: Returns `T&` (`*ptr`).

### `operator bool`

```cpp
FORCE_INLINE explicit operator bool() const noexcept;
```

* **Purpose:** Checks whether `*this` manages a non-null control block and object.
* **Parameters:**
  * None.
* **Return Value:**
  * `true` if `cb != nullptr`; `false` otherwise.

---

## 5. Pluggable Fatal Error Handling

`atomic_shared_ptr.h` continuously validates split-reference invariants (such as 32-bit signed counter overflow/underflow, double-free detection when `new_internal < 0`, reader saturation at `INT32_MAX`, and invalid memory order arguments).

### `FatalErrorHandler`

```cpp
using FatalErrorHandler = void(*)(const char* reason) noexcept;
```

* **Purpose:** Function pointer signature for custom process-wide fatal error notifications invoked before `std::abort()`.

### `SetAtomicPtrErrorHandler`

```cpp
inline void SetAtomicPtrErrorHandler(FatalErrorHandler handler) noexcept;
```

* **Purpose:** Registers (or clears) a custom process-wide callback invoked by `fatal_panic()` when an unrecoverable invariant violation occurs.
* **Restrictions & Guarantees:**
  * Thread-safe; stores `handler` into `g_fatal_error_handler` with `std::memory_order_release`.
  * Pass `nullptr` to restore the default `std::cerr` diagnostic logging behavior.
* **Parameters:**
  * `handler`: Custom `noexcept` callback function pointer, or `nullptr`.
* **Return Value:**
  * None.

---
