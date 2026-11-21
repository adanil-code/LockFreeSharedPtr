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

#include <atomic>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <new>
#include <cstdlib>
#include <bit>
#include <algorithm>
#include <cstddef>
#include <cassert>

#include "shared_ptr_utils.h"

// ========================================================================================
// Platform-Specific Includes
// ========================================================================================
#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#endif

// ========================================================================================
// Architecture & Algorithm Overview
// ========================================================================================
// This file implements a lock-free, thread-safe shared pointer system using a
// "Split Reference Counting" architecture based on Anthony Williams' algorithm.
// It handles highly contended concurrent access without mutexes, relying
// entirely on 128-bit atomic Compare-And-Swap (CAS) loops and wait-free 64-bit
// packed counter arithmetic.
//
// Core Algorithmic Concepts:
//
// 1. Split Reference Counting & 64-Bit Packed Arithmetic:
//    The control block (`ControlBlock<T>`) maintains two separate 32-bit signed
//    reference counts packed into a single 64-bit atomic word (`SplitCounts`):
//      - `internal`: Tracks standard persistent `LocalSharedPtr` instances and the
//        `AtomicSharedPtr` anchor's baseline ownership (initialized to 1).
//      - `external`: Reconciles transient reader checkouts accumulated inside the
//        anchor's 128-bit `CountedPtr<T>` state.
//    `apply_counts()` updates both counters simultaneously in a single wait-free
//    64-bit `fetch_add` using two's complement borrow compensation across the
//    32-bit boundary, destroying the control block only when both `internal` and
//    `external` reach zero.
//
// 2. 128-Bit Atomic Anchor & Overflow Flush (`AtomicSharedPtr`):
//    The globally shared, thread-safe anchor point packs the `ControlBlock<T>*`
//    pointer, a 32-bit `external_count`, and a 32-bit `generation` counter into an
//    aligned 16-byte `CountedPtr<T>` updated via 128-bit CAS (`CMPXCHG16B` on x64,
//    `CASP`/`LDXP`+`STXP` on ARM64):
//      - On `load()`, a reader atomically increments `external_count` in the anchor
//        via 128-bit CAS (shielding the control block from concurrent deletion) and
//        then converts that checkout into a pinned internal reference on the
//        control block (`+1 internal, -1 external`).
//      - If `external_count` in the anchor reaches `EXT_COUNT_FLUSH_THRESHOLD`
//        (2^20) under sustained read-only load, the reader atomically resets the
//        anchor's `external_count` to 0 (bumping `generation`) and flushes the
//        accumulated count into `ControlBlock::counts` to prevent `int32_t`
//        saturation across single or multiple anchors.
//      - When an anchor is overwritten (`store`, `exchange`, `compare_exchange_*`),
//        the writer increments `generation` to prevent ABA races and transfers the
//        displaced anchor's accumulated `external_count` (minus 1 for the anchor's
//        own baseline reference on `store`/`compare_exchange`) into the displaced
//        `ControlBlock::counts`.
//
// 3. Zero-Indirection Thread-Local Accessor (`LocalSharedPtr`):
//    A lightweight, non-thread-safe handle checked out from an `AtomicSharedPtr`
//    (or created via `MakeLocalShared`), used safely within a single thread, and
//    eventually destroyed or stored back. `LocalSharedPtr<T>` caches both the
//    `ControlBlock<T>*` and the resolved `T*` payload pointer so `get()`,
//    `operator->()`, and `operator*()` execute with zero control-block indirection.
//
// 4. Cache-Isolated Single-Allocation Factory (`MakeLocalShared`):
//    `ControlBlock<T>` is aligned to `CACHE_LINE_SIZE` (64 bytes on x64, 128 bytes
//    on ARM64/Apple Silicon). `MakeLocalShared` embeds both `ControlBlock<T>` and
//    the payload object `T` into a single aligned memory allocation, placing `T`
//    at a cache-aligned offset (`payload_offset`) to maximize spatial locality,
//    eliminate heap fragmentation, and prevent false sharing between refcount
//    atomic updates and payload reads.
//
// 5. Standard Library Parity & Memory Order Normalization:
//    The API semantically mirrors `std::atomic<std::shared_ptr<T>>` (providing
//    `compare_exchange_weak`, `compare_exchange_strong`, `is_lock_free`,
//    `use_count`, `reset`, and `swap`):
//      - To prioritize low tail latency and maximum throughput, default memory
//        orders use Acquire/Release/AcqRel rather than `std::memory_order_seq_cst`.
//      - `load()` promotes `std::memory_order_relaxed` to `std::memory_order_acquire`
//        so control block dereferences are always safely synchronized, and RMW
//        operations normalize success orders via `effective_rmw_order()` to at
//        least `std::memory_order_acq_rel` to synchronize control block reclamation.
//      - `compare_exchange_strong` hoists the `desired` internal refcount increment
//        outside the CAS loop and retries immediately on spurious `external_count`
//        or `generation` changes while the cache line remains in Modified (M) state.
//      - NOTE: Invalid load orders (`release`, `acq_rel`) trigger a fatal panic.
// ========================================================================================

// ========================================================================================
// Pluggable Fatal Error Handling
// ========================================================================================

// Function pointer type for custom fatal error notifications prior to process termination.
using FatalErrorHandler = void(*)(const char* reason) noexcept;

// Process-wide atomic pointer storing an optional user-registered fatal error callback.
inline std::atomic<FatalErrorHandler> g_fatal_error_handler{nullptr};

// ----------------------------------------------------------------------------------------
// SetAtomicPtrErrorHandler
// Purpose:
//   Registers a custom process-wide callback invoked by fatal_panic() prior to
//   calling std::abort() when an unrecoverable invariant violation (such as
//   reference count overflow, underflow, or invalid memory order) is detected.
// Parameters:
//   handler - Function pointer to invoke on fatal error, or nullptr to restore
//             default stderr logging.
// Return Value:
//   None.
// ----------------------------------------------------------------------------------------
inline void SetAtomicPtrErrorHandler(FatalErrorHandler handler) noexcept
{
    g_fatal_error_handler.store(handler, std::memory_order_release);
}

// ----------------------------------------------------------------------------------------
// fatal_panic
// Purpose:
//   Invokes the registered FatalErrorHandler (or logs the failure reason to
//   std::cerr if none is installed) and immediately terminates the process via
//   std::abort().
// Parameters:
//   reason - Null-terminated string describing the fatal invariant violation.
// Return Value:
//   None (does not return).
// ----------------------------------------------------------------------------------------
NO_INLINE_ODR void fatal_panic(const char* reason) noexcept
{
    if (auto handler = g_fatal_error_handler.load(std::memory_order_acquire))
    {
        handler(reason);
    }
    else
    {
        std::cerr << "ATOMIC_SHARED_PTR FATAL ERROR: " << reason << '\n';
    }

    std::abort();
}

// Evaluates condition and invokes fatal_panic(reason) on failure.
#define ATOMIC_PTR_EXPECT(condition, reason) \
    do { \
        if (!(condition)) [[unlikely]] { fatal_panic(reason); } \
    } while(false)

// ========================================================================================
// Forward Declarations
// ========================================================================================

template <typename T>
class LocalSharedPtr;
template <typename T>
class AtomicSharedPtr;

template <typename T, typename... Args>
LocalSharedPtr<T> MakeLocalShared(Args&&... args);

// ========================================================================================
// Lock-Free Split Reference Counting
// ========================================================================================

// ----------------------------------------------------------------------------------------
// SplitCounts
// A 64-bit structure representing the dual-counter state of a control block.
// Packs cleanly into a single std::atomic<uint64_t>. Private utility.
// ----------------------------------------------------------------------------------------
struct SplitCounts
{
    int32_t internal; // Lower 32 bits: count of active LocalSharedPtr handles plus anchor baseline ownership
    int32_t external; // Upper 32 bits: net reconciled count of transient external checkouts from AtomicSharedPtr
};

static_assert(sizeof(SplitCounts) == sizeof(uint64_t), "SplitCounts must perfectly map to 64 bits");
static_assert(std::endian::native == std::endian::little, "SplitCounts arithmetic requires little-endian layout");

// ----------------------------------------------------------------------------------------
// apply_counts
// Purpose:
//   Atomically applies delta modifications to internal and external counts.
//   Uses a single wait-free 64-bit atomic fetch_add with two's complement carry
//   compensation across the 32-bit boundary. Private utility.
// Parameters:
//   atomic_counts - Reference to the 64-bit packed SplitCounts atomic variable.
//   int_add       - Signed delta to add to the lower 32-bit internal count.
//   ext_add       - Signed delta to add to the upper 32-bit external count.
//   order         - Memory ordering constraint for the atomic fetch_add.
// Return Value:
//   true if both internal and external counts reached zero after applying the
//   deltas (indicating the caller must destroy the ControlBlock); false otherwise.
// ----------------------------------------------------------------------------------------
FORCE_INLINE bool apply_counts(std::atomic<uint64_t>& atomic_counts,
                               int32_t                int_add,
                               int32_t                ext_add,
                               std::memory_order      order = std::memory_order_acq_rel) noexcept
{
    const uint64_t low_delta  = static_cast<uint32_t>(int_add);
    const uint64_t high_delta = static_cast<uint64_t>(static_cast<uint32_t>(ext_add) - (int_add < 0 ? 1u : 0u)) << 32;
    const uint64_t delta      = low_delta + high_delta;

    const uint64_t old_val = atomic_counts.fetch_add(delta, order);
    const SplitCounts c = std::bit_cast<SplitCounts>(old_val);

    // Explicit check before increments to prevent UB signed overflow/underflow
    if (int_add > 0 && c.internal > (std::numeric_limits<int32_t>::max)() - int_add) [[unlikely]] fatal_panic("Internal count overflow");
    if (ext_add > 0 && c.external > (std::numeric_limits<int32_t>::max)() - ext_add) [[unlikely]] fatal_panic("External count overflow");
    if (int_add < 0 && c.internal < (std::numeric_limits<int32_t>::min)() - int_add) [[unlikely]] fatal_panic("Internal count underflow");
    if (ext_add < 0 && c.external < (std::numeric_limits<int32_t>::min)() - ext_add) [[unlikely]] fatal_panic("External count underflow");

    const int32_t new_internal = c.internal + int_add;
    const int32_t new_external = c.external + ext_add;

    if (new_internal < 0) [[unlikely]]
    {
        fatal_panic("Double free detected: internal count dropped below zero");
    }

    return (new_internal == 0 && new_external == 0);
}

// ----------------------------------------------------------------------------------------
// ControlBlock
// The shared state managing the payload pointer and split reference counts.
// Aligned to the hardware cache line size to prevent false sharing contention.
// Private backend primitive.
// ----------------------------------------------------------------------------------------
template <typename T>
struct alignas(CACHE_LINE_SIZE) ControlBlock
{
    T*                    ptr;         // Pointer to the managed payload object T (either heap-allocated or embedded)
    std::atomic<uint64_t> counts;      // Packed 64-bit SplitCounts tracking internal and external references
    bool                  is_embedded; // true if T was placement-new'd into the same block by MakeLocalShared; false if separate

    // ------------------------------------------------------------------------------------
    // ControlBlock (Constructor)
    // Purpose:
    //   Initializes the control block with the target payload pointer, sets the
    //   initial SplitCounts state to {internal = 1, external = 0}, and records
    //   whether the payload is embedded in the same allocation.
    // Parameters:
    //   p        - Pointer to the constructed payload object of type T.
    //   embedded - true if allocated via MakeLocalShared; false if p is a standalone heap object.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    explicit ControlBlock(T*   p,
                          bool embedded = false) noexcept : ptr(p),
                                                            counts(std::bit_cast<uint64_t>(SplitCounts{1, 0})),
                                                            is_embedded(embedded)
    {
    }

    // ------------------------------------------------------------------------------------
    // ~ControlBlock (Destructor)
    // Purpose:
    //   Destroys the managed payload object, either by explicitly invoking ~T()
    //   for embedded allocations or calling delete for standalone pointers.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    ~ControlBlock()
    {
        if (is_embedded)
        {
            if (ptr)
            {
                ptr->~T();
            }
        }
        else
        {
            delete ptr;
        }
    }

    ControlBlock(const ControlBlock&) = delete;
    ControlBlock& operator=(const ControlBlock&) = delete;

    // ------------------------------------------------------------------------------------
    // operator new
    // Purpose:
    //   Allocates a cache-line-aligned memory block for the ControlBlock (and
    //   optional embedded payload T).
    // Parameters:
    //   size - Number of bytes to allocate.
    // Return Value:
    //   Pointer to the newly allocated aligned memory block (throws std::bad_alloc on failure).
    // ------------------------------------------------------------------------------------
    void* operator new(std::size_t size)
    {
        constexpr std::size_t alloc_align = (std::max)(CACHE_LINE_SIZE, alignof(T));
#if defined(_MSC_VER)
        void* p = _aligned_malloc(size, alloc_align);
        if (!p)
        {
            throw std::bad_alloc();
        }

        return p;
#else
        void* p = nullptr;
        if (posix_memalign(&p, alloc_align, size) != 0)
        {
            throw std::bad_alloc();
        }

        return p;
#endif
    }

    // ------------------------------------------------------------------------------------
    // operator new (Aligned Overload)
    // Purpose:
    //   Forwards C++17 over-aligned allocation requests to the primary cache-aligned
    //   ControlBlock::operator new implementation.
    // Parameters:
    //   size  - Number of bytes to allocate.
    //   align - Requested alignment tag (superseded by max(CACHE_LINE_SIZE, alignof(T))).
    // Return Value:
    //   Pointer to the newly allocated aligned memory block.
    // ------------------------------------------------------------------------------------
    void* operator new(std::size_t      size,
                       std::align_val_t /*align*/)
    {
        return operator new(size);
    }

    // ------------------------------------------------------------------------------------
    // operator delete
    // Purpose:
    //   Releases a cache-aligned memory block previously allocated by
    //   ControlBlock::operator new.
    // Parameters:
    //   p - Pointer to the memory block to free (no-op if nullptr).
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void operator delete(void* p) noexcept
    {
        if (!p)
        {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    // ------------------------------------------------------------------------------------
    // operator delete (Aligned Overload)
    // Purpose:
    //   Forwards C++17 over-aligned deallocation requests to the primary
    //   ControlBlock::operator delete implementation.
    // Parameters:
    //   p     - Pointer to the memory block to free.
    //   align - Alignment tag associated with the allocation.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void operator delete(void*            p,
                         std::align_val_t /*align*/) noexcept
    {
        operator delete(p);
    }
};

// ----------------------------------------------------------------------------------------
// CountedPtr
// A 16-byte aligned structure used by AtomicSharedPtr. Bundles the control
// block, external reference count, and a generation counter. Private utility.
// ----------------------------------------------------------------------------------------
template <typename T>
struct alignas(16) CountedPtr
{
    ControlBlock<T>* cb;             // Pointer to the active ControlBlock<T>, or nullptr if the anchor is empty
    int32_t          external_count; // Transient external checkouts accumulated on this anchor since it was installed or flushed
    uint32_t         generation;     // Monotonically increasing ABA-prevention counter incremented on anchor writes/flushes

    // ------------------------------------------------------------------------------------
    // operator==
    // Purpose:
    //   Performs memberwise equality comparison across cb, external_count, and generation.
    // Parameters:
    //   other - The CountedPtr instance to compare against.
    // Return Value:
    //   true if all three fields match; false otherwise.
    // ------------------------------------------------------------------------------------
    bool operator==(const CountedPtr& other) const = default;
};

// ----------------------------------------------------------------------------------------
// LocalSharedPtr
// The thread-local handle for accessing the payload. It is NOT thread-safe
// for concurrent mutation. Check this out from an AtomicSharedPtr for usage.
// Matches the semantics of std::shared_ptr.
// ----------------------------------------------------------------------------------------
template <typename T>
class LocalSharedPtr
{
    ControlBlock<T>* cb;  // Pointer to the shared control block managing reference counts
    T*               ptr; // Cached direct pointer to the payload object T for zero-indirection access

    friend class AtomicSharedPtr<T>;

    template <typename U, typename... Args>
    friend LocalSharedPtr<U> MakeLocalShared(Args&&... args);

    // Private checkout constructor adopting an existing internal reference from AtomicSharedPtr.
    LocalSharedPtr(ControlBlock<T>* block,
                   bool             /*adopt*/) noexcept : cb(block),
                                                          ptr(block ? block->ptr : nullptr)
    {
    }

    // Private checkout constructor with pre-resolved payload pointer utilized by MakeLocalShared.
    LocalSharedPtr(ControlBlock<T>* block,
                   T*               payload,
                   bool             /*adopt*/) noexcept : cb(block),
                                                          ptr(payload)
    {
    }

public:
    // ------------------------------------------------------------------------------------
    // Constructor (Default)
    // Purpose:
    //   Initializes an empty LocalSharedPtr holding no object.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    constexpr LocalSharedPtr() noexcept : cb(nullptr),
                                          ptr(nullptr)
    {
    }

    // ------------------------------------------------------------------------------------
    // Constructor (Nullptr)
    // Purpose:
    //   Initializes an empty LocalSharedPtr holding no object.
    // Parameters:
    //   std::nullptr_t - Literal nullptr.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    constexpr LocalSharedPtr(std::nullptr_t) noexcept : cb(nullptr),
                                                        ptr(nullptr)
    {
    }

    // ------------------------------------------------------------------------------------
    // Constructor (Raw Pointer)
    // Purpose:
    //   Initializes a LocalSharedPtr owning the provided raw pointer. Allocates
    //   an independent ControlBlock (and deletes p if ControlBlock allocation throws).
    // Parameters:
    //   p - Raw pointer to take ownership of.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    explicit LocalSharedPtr(T* p) : cb(nullptr),
                                    ptr(p)
    {
        if (p)
        {
            try
            {
                cb = new ControlBlock<T>(p);
            }
            catch (...)
            {
                ptr = nullptr;
                delete p;
                throw;
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // Destructor
    // Purpose:
    //   Destroys the local handle, atomically decrementing the internal reference
    //   count. Releases memory if both internal and external counts reach zero.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    ~LocalSharedPtr()
    {
        if (cb)
        {
            if (apply_counts(cb->counts, -1, 0, std::memory_order_acq_rel))
            {
                delete cb;
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // Copy Constructor
    // Purpose:
    //   Increments the internal reference count safely to share ownership.
    // Parameters:
    //   other - The source LocalSharedPtr to copy.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    LocalSharedPtr(const LocalSharedPtr& other) noexcept : cb(other.cb),
                                                           ptr(other.ptr)
    {
        if (cb)
        {
            apply_counts(cb->counts, 1, 0, std::memory_order_relaxed);
        }
    }

    // ------------------------------------------------------------------------------------
    // Copy Assignment Operator
    // Purpose:
    //   Shares ownership with other via copy-and-swap mechanism.
    // Parameters:
    //   other - The source LocalSharedPtr to copy.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------------------
    LocalSharedPtr& operator=(const LocalSharedPtr& other) noexcept
    {
        if (this != &other)
        {
            LocalSharedPtr temp(other);
            swap(temp);
        }

        return *this;
    }

    // ------------------------------------------------------------------------------------
    // Move Constructor
    // Purpose:
    //   Transfers ownership from rvalue handle without atomic count operations.
    // Parameters:
    //   other - The rvalue LocalSharedPtr to move from.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    LocalSharedPtr(LocalSharedPtr&& other) noexcept : cb(other.cb),
                                                      ptr(other.ptr)
    {
        other.cb  = nullptr;
        other.ptr = nullptr;
    }

    // ------------------------------------------------------------------------------------
    // Move Assignment Operator
    // Purpose:
    //   Transfers ownership from other via move-and-swap mechanism.
    // Parameters:
    //   other - The rvalue LocalSharedPtr to move from.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------------------
    LocalSharedPtr& operator=(LocalSharedPtr&& other) noexcept
    {
        if (this != &other)
        {
            LocalSharedPtr temp(std::move(other));
            swap(temp);
        }

        return *this;
    }

    // ------------------------------------------------------------------------------------
    // Nullptr Assignment Operator
    // Purpose:
    //   Resets the pointer to an empty state, releasing previous ownership.
    // Parameters:
    //   std::nullptr_t - Literal nullptr.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------------------
    LocalSharedPtr& operator=(std::nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    // ------------------------------------------------------------------------------------
    // swap
    // Purpose:
    //   Swaps the managed control block and payload pointers between this and
    //   another handle without atomic operations.
    // Parameters:
    //   other - The LocalSharedPtr instance to swap with.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void swap(LocalSharedPtr& other) noexcept
    {
        std::swap(cb, other.cb);
        std::swap(ptr, other.ptr);
    }

    // ------------------------------------------------------------------------------------
    // reset (Empty)
    // Purpose:
    //   Releases ownership of the managed object, reverting to empty state.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void reset() noexcept
    {
        LocalSharedPtr().swap(*this);
    }

    // ------------------------------------------------------------------------------------
    // reset (Pointer)
    // Purpose:
    //   Replaces the managed object with a new raw pointer allocation.
    // Parameters:
    //   p - Raw pointer to the new object.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void reset(T* p)
    {
        LocalSharedPtr(p).swap(*this);
    }

    // ------------------------------------------------------------------------------------
    // use_count
    // Purpose:
    //   Returns the approximate combined internal and transient external refs.
    // Parameters:
    //   None.
    // Return Value:
    //   Total approximated reference count as long.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] long use_count() const noexcept
    {
        if (!cb)
        {
            return 0;
        }

        SplitCounts c = std::bit_cast<SplitCounts>(cb->counts.load(std::memory_order_relaxed));
        const long transient_external = (c.external > 0) ? static_cast<long>(c.external) : 0L;

        return static_cast<long>(c.internal) + transient_external;
    }

    // ------------------------------------------------------------------------------------
    // get
    // Purpose:
    //   Retrieves the underlying raw pointer without granting ownership.
    // Parameters:
    //   None.
    // Return Value:
    //   Pointer to managed object, or nullptr.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE T* get() const noexcept
    {
        return ptr;
    }

    // ------------------------------------------------------------------------------------
    // operator->
    // Purpose:
    //   Member access operator mapping directly to the underlying payload.
    // Parameters:
    //   None.
    // Return Value:
    //   Pointer to managed object.
    // ------------------------------------------------------------------------------------
    FORCE_INLINE T* operator->() const noexcept
    {
        return ptr;
    }

    // ------------------------------------------------------------------------------------
    // operator*
    // Purpose:
    //   Dereferences the underlying payload. Caller must ensure the handle is non-empty.
    // Parameters:
    //   None.
    // Return Value:
    //   Reference to managed object.
    // ------------------------------------------------------------------------------------
    FORCE_INLINE T& operator*() const noexcept
    {
        return *ptr;
    }

    // ------------------------------------------------------------------------------------
    // operator bool
    // Purpose:
    //   Checks whether an object is currently managed.
    // Parameters:
    //   None.
    // Return Value:
    //   True if non-null, false if empty.
    // ------------------------------------------------------------------------------------
    FORCE_INLINE explicit operator bool() const noexcept
    {
        return cb != nullptr;
    }
};

// ----------------------------------------------------------------------------------------
// AtomicSharedPtr
// The globally shared, thread-safe anchor. Uses 128-bit atomic instructions
// to allow concurrent threads to check out or swap local handles without locks.
// Matches the semantics of std::atomic<std::shared_ptr>.
// ----------------------------------------------------------------------------------------
template <typename T>
class AtomicSharedPtr
{
    static_assert(sizeof(CountedPtr<T>)    == 16, "CountedPtr must be exactly 16 bytes for 128-bit CAS");
    static_assert(alignof(CountedPtr<T>)   == 16, "CountedPtr must be 16-byte aligned");
    static_assert(alignof(ControlBlock<T>) == CACHE_LINE_SIZE, "ControlBlock must be cache-line aligned");

    // Threshold at which load() flushes accumulated external_count back to the ControlBlock
    // to prevent int32_t saturation during read-only workloads or multi-anchor sharing.
    static constexpr int32_t EXT_COUNT_FLUSH_THRESHOLD = 1 << 20;

    // Normalizes RMW memory order so operations that both publish desired state and
    // acquire ownership of displaced ControlBlocks always establish acquire-release edges.
    FORCE_INLINE static constexpr std::memory_order effective_rmw_order(std::memory_order order) noexcept
    {
        return (order == std::memory_order_seq_cst) ? std::memory_order_seq_cst : std::memory_order_acq_rel;
    }

#if defined(_MSC_VER) && !defined(_M_ARM64)
    // Helper 128-bit pod layout for unpacking CountedPtr<T> into 64-bit register pairs for _InterlockedCompareExchange128.
    struct Raw128
    {
        LONG64 low;  // Lower 64 bits (ControlBlock<T>* cb)
        LONG64 high; // Upper 64 bits (external_count and generation)
    };

    // Prevent false sharing by isolating the anchor to its own hardware cache line.
    alignas(CACHE_LINE_SIZE) mutable CountedPtr<T> atomic_ptr{nullptr, 0, 0};

    // Guaranteed tear-free 128-bit load using CMPXCHG16B.
    FORCE_INLINE CountedPtr<T> atomic_load(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        ATOMIC_PTR_EXPECT(order != std::memory_order_release && order != std::memory_order_acq_rel, "Invalid load memory order");

        CountedPtr<T> current = {nullptr, 0, 0};
        
        // Executes a 128-bit CAS against a dummy zero state. If atomic_ptr is non-zero,
        // it accurately captures the current state. If atomic_ptr is genuinely zero, 
        // it cleanly re-writes zero, securing the bus lock to prevent tearing.
        _InterlockedCompareExchange128(reinterpret_cast<volatile LONG64*>(&atomic_ptr),
                                       0,
                                       0,
                                       reinterpret_cast<LONG64*>(&current));

        return current;
    }

    // Executes a 128-bit interlocked compare-and-swap via Windows compiler intrinsics.
    FORCE_INLINE bool atomic_cas(CountedPtr<T>&    expected,
                                 CountedPtr<T>     desired,
                                 std::memory_order /*succ*/,
                                 std::memory_order /*fail*/) const noexcept
    {
        const Raw128 desired64 = std::bit_cast<Raw128>(desired);
        return _InterlockedCompareExchange128(reinterpret_cast<volatile LONG64*>(&atomic_ptr),
                                              desired64.high,
                                              desired64.low,
                                              reinterpret_cast<LONG64*>(&expected)) != 0;
    }

    // Non-atomic 128-bit structure initialization used during raw-pointer construction.
    FORCE_INLINE void atomic_store_init(CountedPtr<T> init) noexcept
    {
        atomic_ptr = init;
    }
#else
    // Prevent false sharing by isolating the anchor to its own hardware cache line.
    alignas(CACHE_LINE_SIZE) mutable std::atomic<CountedPtr<T>> atomic_ptr{CountedPtr<T>{nullptr, 0, 0}};

    // Standard std::atomic compliant 128-bit load.
    FORCE_INLINE CountedPtr<T> atomic_load(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        ATOMIC_PTR_EXPECT(order != std::memory_order_release && order != std::memory_order_acq_rel, "Invalid load memory order");
        return atomic_ptr.load(order);
    }

    // Standard std::atomic compliant weak compare-and-exchange.
    FORCE_INLINE bool atomic_cas(CountedPtr<T>&    expected,
                                 CountedPtr<T>     desired,
                                 std::memory_order succ,
                                 std::memory_order fail) const noexcept
    {
        return atomic_ptr.compare_exchange_weak(expected, desired, succ, fail);
    }

    // Standard std::atomic compliant store initialization.
    FORCE_INLINE void atomic_store_init(CountedPtr<T> init) noexcept
    {
        atomic_ptr.store(init, std::memory_order_release);
    }
#endif

public:
    AtomicSharedPtr(const AtomicSharedPtr&) = delete;
    AtomicSharedPtr& operator=(const AtomicSharedPtr&) = delete;

    // ------------------------------------------------------------------------------------
    // Constructor (Default)
    // Purpose:
    //   Initializes an empty AtomicSharedPtr anchor structure.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    constexpr AtomicSharedPtr() noexcept = default;

    // ------------------------------------------------------------------------------------
    // Constructor (LocalSharedPtr Checkout)
    // Purpose:
    //   Consumes a LocalSharedPtr, permanently anchoring it globally without
    //   extra atomic reference count operations.
    // Parameters:
    //   desired - The local handle whose ownership is transferred to the anchor.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    explicit AtomicSharedPtr(LocalSharedPtr<T> desired) noexcept : atomic_ptr{CountedPtr<T>{desired.cb, 0, 0}}
    {
        desired.cb  = nullptr;
        desired.ptr = nullptr;
    }

    // ------------------------------------------------------------------------------------
    // Constructor (Raw Pointer)
    // Purpose:
    //   Allocates a control block anchoring a raw pointer globally (and deletes
    //   ptr if ControlBlock allocation throws).
    // Parameters:
    //   ptr - Raw pointer to wrap and anchor.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    explicit AtomicSharedPtr(T* ptr) : AtomicSharedPtr()
    {
        if (ptr)
        {
            CountedPtr<T> init = {nullptr, 0, 0};
            try
            {
                init.cb = new ControlBlock<T>(ptr);
            }
            catch (...)
            {
                delete ptr;
                throw;
            }

            atomic_store_init(init);
        }
    }

    // ------------------------------------------------------------------------------------
    // Destructor
    // Purpose:
    //   Releases any currently anchored references directly without a CAS loop,
    //   reconciling both the anchor's baseline internal reference (-1) and its
    //   accumulated external_count with the ControlBlock.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    ~AtomicSharedPtr()
    {
        CountedPtr<T> old_val = atomic_load(std::memory_order_relaxed);
        if (old_val.cb)
        {
            if (apply_counts(old_val.cb->counts, -1, old_val.external_count, std::memory_order_acq_rel))
            {
                delete old_val.cb;
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // operator= (Assignment)
    // Purpose:
    //   Atomically assigns and stores a new local handle using release semantics.
    // Parameters:
    //   desired - New LocalSharedPtr handle replacing the anchor.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void operator=(LocalSharedPtr<T> desired) noexcept
    {
        store(std::move(desired));
    }

    // ------------------------------------------------------------------------------------
    // Conversion Operator
    // Purpose:
    //   Implicitly executes an atomic lock-free load, granting a LocalSharedPtr.
    // Parameters:
    //   None.
    // Return Value:
    //   Checked-out LocalSharedPtr pointing to the shared asset.
    // ------------------------------------------------------------------------------------
    operator LocalSharedPtr<T>() const noexcept
    {
        return load();
    }

    // ------------------------------------------------------------------------------------
    // is_lock_free
    // Purpose:
    //   Indicates if the underlying architecture guarantees lock-free 128-bit atomics.
    // Parameters:
    //   None.
    // Return Value:
    //   True if lock-free 128-bit atomics are supported; false otherwise.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool is_lock_free() const noexcept
    {
#if defined(_MSC_VER) && !defined(_M_ARM64)
        return true;
#elif defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
        return true;
#else
        return atomic_ptr.is_lock_free();
#endif
    }

    // ------------------------------------------------------------------------------------
    // load
    // Purpose:
    //   Atomically checks out a local handle, incrementing the transient external
    //   count in the 128-bit anchor to shield the underlying object from concurrent
    //   thread deletion before converting the checkout into a pinned internal
    //   reference. Periodically flushes external_count if EXT_COUNT_FLUSH_THRESHOLD
    //   is reached.
    // Parameters:
    //   order - Memory ordering constraint (relaxed is promoted to acquire;
    //           release and acq_rel trigger a fatal panic).
    // Return Value:
    //   Securely checked-out LocalSharedPtr handle (or empty handle if null).
    // ------------------------------------------------------------------------------------
    [[nodiscard]] LocalSharedPtr<T> load(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        ATOMIC_PTR_EXPECT(order != std::memory_order_release && order != std::memory_order_acq_rel, "Invalid load memory order");

        const std::memory_order effective_order = (order == std::memory_order_relaxed) ? std::memory_order_acquire : order;

        CountedPtr<T> old_val = atomic_load(effective_order);
        CountedPtr<T> new_val;
        AdaptiveBackoff backoff;

        while (true)
        {
            if (!old_val.cb)
            {
                return LocalSharedPtr<T>();
            }

            ATOMIC_PTR_EXPECT(old_val.external_count < (std::numeric_limits<int32_t>::max)(),
                              "AtomicSharedPtr reader saturation: maximum concurrent lock-free loads exceeded");

            new_val = old_val;
            new_val.external_count++; // Check out a temporary external reference

            if (atomic_cas(old_val, new_val, effective_order, std::memory_order_acquire)) [[likely]]
            {
                break;
            }

            backoff();
        }

        // Convert the checked-out external reference into a pinned internal reference first.
        apply_counts(old_val.cb->counts, 1, -1, std::memory_order_relaxed);

        // If external_count crossed the flush threshold, flush it back to the ControlBlock
        // while holding our live internal reference so cb->counts.internal cannot reach zero.
        // Loop using the updated expected_flush from atomic_cas so concurrent readers
        // incrementing external_count during apply_counts() above cannot starve the flush.
        if (new_val.external_count >= EXT_COUNT_FLUSH_THRESHOLD) [[unlikely]]
        {
            CountedPtr<T> expected_flush = new_val;
            while (expected_flush.cb == new_val.cb &&
                   expected_flush.external_count >= EXT_COUNT_FLUSH_THRESHOLD)
            {
                CountedPtr<T> desired_flush = {new_val.cb, 0, expected_flush.generation + 1};
                if (atomic_cas(expected_flush, desired_flush, std::memory_order_acq_rel, std::memory_order_relaxed))
                {
                    apply_counts(old_val.cb->counts, 0, expected_flush.external_count, std::memory_order_acq_rel);
                    break;
                }
            }
        }

        return LocalSharedPtr<T>(old_val.cb, true);
    }

    // ------------------------------------------------------------------------------------
    // store (LocalSharedPtr)
    // Purpose:
    //   Atomically replaces the current anchored pointer with a new LocalSharedPtr,
    //   bumping the generation counter and deleting the older anchor if no internal
    //   or external dependencies remain.
    // Parameters:
    //   desired - Value to permanently store into the anchor.
    //   order   - Memory order constraint (normalized to at least acq_rel).
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void store(LocalSharedPtr<T> desired,
               std::memory_order order = std::memory_order_release) noexcept
    {
        CountedPtr<T> new_val = {nullptr, 0, 0};
        if (desired)
        {
            new_val.cb  = desired.cb;
            desired.cb  = nullptr;
            desired.ptr = nullptr;
        }

        const std::memory_order cas_order = effective_rmw_order(order);
        CountedPtr<T> old_val = atomic_load(std::memory_order_relaxed);
        AdaptiveBackoff backoff;

        while (true)
        {
            new_val.generation = old_val.generation + 1;
            if (atomic_cas(old_val, new_val, cas_order, std::memory_order_relaxed)) [[likely]]
            {
                break;
            }

            backoff();
        }

        if (old_val.cb)
        {
            if (apply_counts(old_val.cb->counts, -1, old_val.external_count, std::memory_order_acq_rel))
            {
                delete old_val.cb;
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // store (Raw Pointer)
    // Purpose:
    //   Atomically replaces the current anchored pointer by wrapping a raw pointer
    //   in a new LocalSharedPtr and storing it into the anchor.
    // Parameters:
    //   ptr   - Raw pointer to store into the anchor.
    //   order - Memory order constraint.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------------------
    void store(T*                ptr,
               std::memory_order order = std::memory_order_release)
    {
        LocalSharedPtr<T> local(ptr);
        store(std::move(local), order);
    }

    // ------------------------------------------------------------------------------------
    // exchange (LocalSharedPtr)
    // Purpose:
    //   Atomically executes a replacement and returns the previous active anchor as
    //   a LocalSharedPtr. Reconciles any accumulated external reference counts on
    //   the displaced anchor into its ControlBlock while transferring the anchor's
    //   baseline internal reference to the returned LocalSharedPtr.
    // Parameters:
    //   desired - Value to install into the anchor.
    //   order   - Memory order constraint (normalized to at least acq_rel).
    // Return Value:
    //   The preceding LocalSharedPtr instance held by the anchor.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] LocalSharedPtr<T> exchange(LocalSharedPtr<T> desired,
                                             std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        CountedPtr<T> new_val = {nullptr, 0, 0};
        if (desired)
        {
            new_val.cb  = desired.cb;
            desired.cb  = nullptr;
            desired.ptr = nullptr;
        }

        const std::memory_order cas_order = effective_rmw_order(order);
        CountedPtr<T> old_val = atomic_load(std::memory_order_relaxed);
        AdaptiveBackoff backoff;

        while (true)
        {
            new_val.generation = old_val.generation + 1;
            if (atomic_cas(old_val, new_val, cas_order, std::memory_order_relaxed)) [[likely]]
            {
                break;
            }

            backoff();
        }

        if (!old_val.cb)
        {
            return LocalSharedPtr<T>();
        }

        if (old_val.external_count > 0)
        {
            apply_counts(old_val.cb->counts, 0, old_val.external_count, std::memory_order_acq_rel);
        }

        return LocalSharedPtr<T>(old_val.cb, true);
    }

    // ------------------------------------------------------------------------------------
    // exchange (Raw Pointer)
    // Purpose:
    //   Atomically executes a replacement via a raw pointer, returning the previous
    //   active anchor as a LocalSharedPtr.
    // Parameters:
    //   ptr   - Raw pointer to integrate into the anchor.
    //   order - Memory order constraint.
    // Return Value:
    //   The preceding LocalSharedPtr instance held by the anchor.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] LocalSharedPtr<T> exchange(T*                ptr,
                                             std::memory_order order = std::memory_order_acq_rel)
    {
        LocalSharedPtr<T> local(ptr);
        return exchange(std::move(local), order);
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_weak (Lvalue, Dual-Order)
    // Purpose:
    //   Attempts to swap underlying state if `expected` matches the anchor's
    //   ControlBlock. May fail spuriously if concurrent read/write modifications
    //   increment external_count or generation. Updates `expected` when the
    //   underlying ControlBlock pointer mismatches.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Value securely assigned to anchor if exchange succeeds.
    //   succ     - Evaluated memory order upon CAS success.
    //   fail     - Evaluated memory order upon CAS mismatch.
    // Return Value:
    //   True verifying exchange completeness, false if preempted or mismatched.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&       expected,
                                             const LocalSharedPtr<T>& desired,
                                             std::memory_order        succ,
                                             std::memory_order        fail) noexcept
    {
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        const std::memory_order load_fail_order = (fail == std::memory_order_relaxed) ? std::memory_order_acquire : fail;
        const std::memory_order cas_succ_order = effective_rmw_order(succ);

        CountedPtr<T> current = atomic_load(load_fail_order);

        if (current.cb != expected.cb)
        {
            if (!current.cb)
            {
                expected.reset();
            }
            else
            {
                expected = load(load_fail_order);
            }

            return false;
        }

        ControlBlock<T>* const desired_cb = desired.cb;
        if (desired_cb)
        {
            apply_counts(desired_cb->counts, 1, 0, std::memory_order_relaxed);
            current = atomic_load(load_fail_order);
        }

        if (current.cb == expected.cb)
        {
            CountedPtr<T> new_val = {desired_cb, 0, current.generation + 1};

            if (atomic_cas(current, new_val, cas_succ_order, load_fail_order)) [[likely]]
            {
                if (current.cb)
                {
                    if (apply_counts(current.cb->counts, -1, current.external_count, std::memory_order_acq_rel))
                    {
                        delete current.cb;
                    }
                }

                return true;
            }
        }

        if (desired_cb)
        {
            if (apply_counts(desired_cb->counts, -1, 0, std::memory_order_acq_rel))
            {
                delete desired_cb;
            }
        }

        if (current.cb != expected.cb)
        {
            if (!current.cb)
            {
                expected.reset();
            }
            else
            {
                expected = load(load_fail_order);
            }
        }

        return false;
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_weak (Rvalue, Dual-Order)
    // Purpose:
    //   Evaluates a weak swap on matched state. Rvalue properties guarantee `desired` 
    //   transfers ownership only upon explicit CAS commit, avoiding unnecessary
    //   atomic increments on `desired`.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Rvalue securely moved strictly upon verification success.
    //   succ     - Evaluated memory order upon CAS success.
    //   fail     - Evaluated memory order upon CAS mismatch.
    // Return Value:
    //   True verifying exchange completeness, false if preempted or mismatched.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&  expected,
                                             LocalSharedPtr<T>&& desired,
                                             std::memory_order   succ,
                                             std::memory_order   fail) noexcept
    {
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        const std::memory_order load_fail_order = (fail == std::memory_order_relaxed) ? std::memory_order_acquire : fail;
        const std::memory_order cas_succ_order = effective_rmw_order(succ);

        CountedPtr<T> current = atomic_load(load_fail_order);

        if (current.cb != expected.cb)
        {
            if (!current.cb)
            {
                expected.reset();
            }
            else
            {
                expected = load(load_fail_order);
            }

            return false;
        }

        CountedPtr<T> new_val = {desired.cb, 0, current.generation + 1};

        if (atomic_cas(current, new_val, cas_succ_order, load_fail_order)) [[likely]]
        {
            desired.cb  = nullptr;
            desired.ptr = nullptr;

            if (current.cb)
            {
                if (apply_counts(current.cb->counts, -1, current.external_count, std::memory_order_acq_rel))
                {
                    delete current.cb;
                }
            }

            return true;
        }

        if (current.cb != expected.cb)
        {
            if (!current.cb)
            {
                expected.reset();
            }
            else
            {
                expected = load(load_fail_order);
            }
        }

        return false;
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_weak (Lvalue, Single-Order)
    // Purpose:
    //   Maps standard weak CAS sequence relying on unified memory order rules.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Value securely assigned to anchor if exchange succeeds.
    //   order    - Success memory order. Failure resolves equivalently standard-compliant.
    // Return Value:
    //   True verifying exchange completeness, false if preempted or mismatched.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&       expected,
                                             const LocalSharedPtr<T>& desired,
                                             std::memory_order        order = std::memory_order_acq_rel) noexcept
    {
        std::memory_order fail = order;
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        return compare_exchange_weak(expected, desired, order, fail);
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_weak (Rvalue, Single-Order)
    // Purpose:
    //   Maps standard weak CAS rvalue sequence relying on unified memory order rules.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Rvalue securely moved strictly upon verification success.
    //   order    - Success memory order. Failure resolves equivalently standard-compliant.
    // Return Value:
    //   True verifying exchange completeness, false if preempted or mismatched.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_weak(LocalSharedPtr<T>&  expected,
                                             LocalSharedPtr<T>&& desired,
                                             std::memory_order   order = std::memory_order_acq_rel) noexcept
    {
        std::memory_order fail = order;
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        return compare_exchange_weak(expected, std::move(desired), order, fail);
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_strong (Lvalue, Dual-Order)
    // Purpose:
    //   Compares active underlying state, intercepting and retrying on spurious 
    //   interruptions (such as concurrent reader external_count updates). Optimized
    //   to increment the desired handle's reference count at most once before
    //   entering the CAS loop.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Value securely assigned to anchor if exchange succeeds.
    //   succ     - Evaluated memory order upon CAS success.
    //   fail     - Evaluated memory order upon CAS mismatch.
    // Return Value:
    //   True strictly bypassing spurious faults, false indicating physical pointer swap.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&       expected,
                                               const LocalSharedPtr<T>& desired,
                                               std::memory_order        succ,
                                               std::memory_order        fail) noexcept
    {
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        const std::memory_order load_fail_order = (fail == std::memory_order_relaxed) ? std::memory_order_acquire : fail;
        const std::memory_order cas_succ_order = effective_rmw_order(succ);

        CountedPtr<T> current = atomic_load(load_fail_order);

        // Immediate exit if the underlying pointers are unequivocally different.
        if (current.cb != expected.cb)
        {
            if (!current.cb)
            {
                expected.reset();
                return false;
            }

            LocalSharedPtr<T> actual = load(load_fail_order);
            if (actual.cb != expected.cb)
            {
                expected = std::move(actual);
                return false;
            }

            current = atomic_load(load_fail_order);
        }

        // Hoisted optimization: Bump local desired handle reference safely exactly once.
        ControlBlock<T>* const desired_cb = desired.cb;
        if (desired_cb)
        {
            apply_counts(desired_cb->counts, 1, 0, std::memory_order_relaxed);
        }

        while (true)
        {
            // Real failure indicates explicit pointer replacement triggered concurrently.
            if (current.cb != expected.cb)
            {
                if (!current.cb)
                {
                    if (desired_cb)
                    {
                        if (apply_counts(desired_cb->counts, -1, 0, std::memory_order_acq_rel))
                        {
                            delete desired_cb;
                        }
                    }

                    expected.reset();
                    return false;
                }

                LocalSharedPtr<T> actual = load(load_fail_order);
                if (actual.cb != expected.cb)
                {
                    if (desired_cb)
                    {
                        if (apply_counts(desired_cb->counts, -1, 0, std::memory_order_acq_rel))
                        {
                            delete desired_cb;
                        }
                    }

                    expected = std::move(actual);
                    return false;
                }

                current = atomic_load(load_fail_order);
                continue;
            }

            CountedPtr<T> new_val = {desired_cb, 0, current.generation + 1};

            if (atomic_cas(current, new_val, cas_succ_order, load_fail_order)) [[likely]]
            {
                if (current.cb)
                {
                    if (apply_counts(current.cb->counts, -1, current.external_count, std::memory_order_acq_rel))
                    {
                        delete current.cb;
                    }
                }

                return true;
            }

            // Spurious iteration (current.cb == expected.cb) retries immediately without backoff
            // while holding the cache line in M state from the failed CMPXCHG16B.
        }
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_strong (Rvalue, Dual-Order)
    // Purpose:
    //   Internally defeats false external_count/generation collisions while
    //   transferring ownership from `desired` strictly upon CAS success. `desired`
    //   remains undisturbed if the exchange fails.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Rvalue securely moved strictly upon verification success.
    //   succ     - Evaluated memory order upon CAS success.
    //   fail     - Evaluated memory order upon CAS mismatch.
    // Return Value:
    //   True strictly bypassing spurious faults, false indicating physical pointer swap.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&  expected,
                                               LocalSharedPtr<T>&& desired,
                                               std::memory_order   succ,
                                               std::memory_order   fail) noexcept
    {
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        const std::memory_order load_fail_order = (fail == std::memory_order_relaxed) ? std::memory_order_acquire : fail;
        const std::memory_order cas_succ_order = effective_rmw_order(succ);

        CountedPtr<T> current = atomic_load(load_fail_order);

        while (true)
        {
            if (current.cb != expected.cb)
            {
                if (!current.cb)
                {
                    expected.reset();
                    return false;
                }

                LocalSharedPtr<T> actual = load(load_fail_order);
                if (actual.cb != expected.cb)
                {
                    expected = std::move(actual);
                    return false;
                }

                current = atomic_load(load_fail_order);
                continue;
            }

            CountedPtr<T> new_val = {desired.cb, 0, current.generation + 1};

            if (atomic_cas(current, new_val, cas_succ_order, load_fail_order)) [[likely]]
            {
                desired.cb  = nullptr;
                desired.ptr = nullptr;

                if (current.cb)
                {
                    if (apply_counts(current.cb->counts, -1, current.external_count, std::memory_order_acq_rel))
                    {
                        delete current.cb;
                    }
                }
                return true;
            }

            // Spurious iteration (current.cb == expected.cb) retries immediately without backoff
            // while holding the cache line in M state from the failed CMPXCHG16B.
        }
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_strong (Lvalue, Single-Order)
    // Purpose:
    //   Executes robust spurious-safe CAS utilizing parameter mirroring for standard parity.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Value securely assigned to anchor if exchange succeeds.
    //   order    - Success memory order. Failure resolves equivalently standard-compliant.
    // Return Value:
    //   True strictly bypassing spurious faults, false indicating physical pointer swap.
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&       expected,
                                               const LocalSharedPtr<T>& desired,
                                               std::memory_order        order = std::memory_order_acq_rel) noexcept
    {
        std::memory_order fail = order;
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        return compare_exchange_strong(expected, desired, order, fail);
    }

    // ------------------------------------------------------------------------------------
    // compare_exchange_strong (Rvalue, Single-Order)
    // Purpose:
    //   Resolves robust rvalue CAS mapping safely over a single unified memory 
    //   order.
    // Parameters:
    //   expected - In/out reference to expected lock-free handle. Updates on fail.
    //   desired  - Rvalue securely moved strictly upon verification success.
    //   order    - Success memory order. Failure resolves equivalently standard-compliant.
    // Return Value:
    //   True strictly bypassing spurious faults, false indicating physical pointer swap.    
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool compare_exchange_strong(LocalSharedPtr<T>&  expected,
                                               LocalSharedPtr<T>&& desired,
                                               std::memory_order   order = std::memory_order_acq_rel) noexcept
    {
        std::memory_order fail = order;
        if (fail == std::memory_order_acq_rel)
        {
            fail = std::memory_order_acquire;
        }
        else if (fail == std::memory_order_release)
        {
            fail = std::memory_order_relaxed;
        }

        return compare_exchange_strong(expected, std::move(desired), order, fail);
    }
};

// ========================================================================================
// Single-Allocation Factory
// ========================================================================================

// ----------------------------------------------------------------------------------------
// MakeLocalShared
// Purpose:
//   Creates a LocalSharedPtr in a single cache-aligned heap allocation block. It
//   places the ControlBlock at the front and embeds the payload object directly
//   adjacent to it at a cache-aligned offset, ensuring strong data locality,
//   avoiding heap fragmentation, and isolating the payload from control-block
//   atomic mutations.
// Parameters:
//   args - Arguments perfectly forwarded to the payload constructor of type T.
// Return Value:
//   Newly constructed LocalSharedPtr owning the embedded control block and 
//   object.
// ----------------------------------------------------------------------------------------
template <typename T, typename... Args>
[[nodiscard]] LocalSharedPtr<T> MakeLocalShared(Args&&... args)
{
    constexpr std::size_t cb_size = sizeof(ControlBlock<T>);
    constexpr std::size_t alloc_align = (std::max)(CACHE_LINE_SIZE, alignof(T));

    constexpr std::size_t payload_offset = (cb_size + alloc_align - 1) & ~(alloc_align - 1);
    constexpr std::size_t total_size = payload_offset + sizeof(T);

    void* mem = ControlBlock<T>::operator new(total_size);

    T* payload = nullptr;
    try
    {
        payload = ::new (static_cast<char*>(mem) + payload_offset) T(std::forward<Args>(args)...);
    }
    catch (...)
    {
        ControlBlock<T>::operator delete(mem);
        throw;
    }

    ControlBlock<T>* cb = ::new (mem) ControlBlock<T>(payload, true);

    return LocalSharedPtr<T>(cb, payload, true);
}