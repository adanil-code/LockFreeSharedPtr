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
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <utility>
#include <algorithm>
#include <type_traits>

#include "shared_ptr_utils.h"

// ============================================================================
// Architecture & Algorithm Overview
// ============================================================================
// This file implements an Epoch-Based, NUMA- & Cache-Aware Shared Pointer
// tailored for high-throughput scalable user-mode environments (Windows, Linux,
// and macOS). It achieves wait-free reader scaling and O(1) memory reclamation
// by strictly decoupling short-lived Transient Access from long-lived
// Persistent Ownership.
//
// Core Algorithmic Concepts:
//
// 1. Epoch-Based Transient Access (Execute-Around Idiom): 
//    Fast-path reads (`read()` and `acquire_batched()`) are strictly bound to a
//    scoped C++ lambda or stack-bound guard. Instead of mutating a central
//    atomic counter (which causes cache-line ping-ponging), readers acquire
//    atomic rundown protection on a globally shared "Active Epoch" shard
//    (`rundown_refs[epoch & 1]`) and re-verify that the 64-bit monotonic
//    `active_epoch` did not advance prior to loading the active pointer.
//    Payload addresses are resolved via constant compile-time offset arithmetic
//    (`get_payload()`) so fast-path readers never touch the control block's
//    mutable cache line. Primary slot owners use a single-writer non-RMW
//    release store on exit (while oversubscribed threads >256 or late TLS
//    destructors use a decoupled `shared_readers` counter), allowing millions
//    of concurrent reads to scale linearly across symmetric multiprocessors
//    and SMT siblings.
//
// 2. Persistent Promotion (PersistentSharedPtr): 
//    When an object must outlive the transient scope (e.g., pended requests 
//    or long-lived stream contexts), readers execute `load_persistent()`. While
//    protected by a transient `ScopedRundownGuard`, this increments a standard
//    atomic reference count (`persistent_ref_count`) inside the control block, 
//    securely promoting the object out of the epoch-based GC pipeline.
//
// 3. Asynchronous Two-Phase O(1) Garbage Collection: 
//    Displaced pointers are pushed to a NUMA-node-local lock-free staging queue.
//    A background process periodically executes a two-phase, zero-blocking
//    reclamation (`process_garbage_pipeline()`):
//      - Phase A (Try Swap & Reclaim): Attempts to drain any pending batches
//        from a prior cycle (`pending_free_lists`), atomically detaches all
//        NUMA SList shards in O(nodes) without pointer chasing, advances the
//        global `active_epoch`, and loops back once to check if the previous
//        epoch shard has drained. If readers are still executing during a
//        non-flush run, it smoothly yields and retries on the next cycle.
//      - Phase B (Batch Free): Once the old epoch shard drains, instantly
//        reclaims the entire batch of orphaned objects across all NUMA shards
//        by resetting their anchor state and dropping their baseline reference.
//        This eliminates per-object rundown allocation overhead and completely
//        bypasses GC Head-of-Line blocking.
//
// 4. NUMA-Affinitized Slab Bump Arena & Node-Local Recycling:
//    - Displaced blocks are pushed to per-NUMA-node SList shards
//      (`NumaQuiescentSList`) allocated in node-local physical RAM via
//      `NumaAllocator::Allocate`, eliminating cross-socket cache-line bouncing
//      during concurrent `store()` operations.
//    - Control-block allocations consult the calling thread's per-slot lock-free
//      cache (`SlotBlockCache`) for exclusive local-node requests, then the
//      target node's multi-batch `node_spare_stack` (supporting up to
//      64 batches / 2,048 blocks and preserving heterogeneous payload sizes).
//      On multi-socket NUMA systems, cache misses bump-allocate up to
//      `SLAB_CARVE_BATCH_SIZE` (16) blocks (for local nodes) or 1 block (for
//      remote nodes) from a per-node reference-counted 64 KB NUMA slab
//      (`NumaSlabHeader`).
//    - On single-node UMA machines and macOS, allocation seamlessly takes the
//      direct cache-aligned `::operator new` fast path with zero virtual-memory
//      or syscall overhead.
// ============================================================================

// ============================================================================
// Pluggable Fatal Error Handling
// ============================================================================

// Function pointer type for custom fatal error notifications prior to process termination.
using RundownFatalErrorHandler = void(*)(const char* reason) noexcept;

// Process-wide atomic pointer storing an optional user-registered fatal error callback.
inline std::atomic<RundownFatalErrorHandler> g_rundown_fatal_error_handler{nullptr};

// ----------------------------------------------------------------------------
// SetRundownPtrErrorHandler
// Purpose:
//   Registers a custom process-wide callback invoked by rundown_fatal_panic()
//   prior to calling std::abort() when an unrecoverable invariant violation
//   (such as persistent reference count overflow or underflow) is detected.
// Parameters:
//   handler - Function pointer to invoke on fatal error, or nullptr to restore
//             default stderr logging.
// Return Value:
//   None.
// ----------------------------------------------------------------------------
inline void SetRundownPtrErrorHandler(RundownFatalErrorHandler handler) noexcept
{
    g_rundown_fatal_error_handler.store(handler, std::memory_order_release);
}

// ----------------------------------------------------------------------------
// rundown_fatal_panic
// Purpose:
//   Invokes the registered RundownFatalErrorHandler (or logs the failure reason
//   to std::cerr if none is installed) and immediately terminates the process
//   via std::abort().
// Parameters:
//   reason - Null-terminated string describing the fatal invariant violation.
// Return Value:
//   None (does not return).
// ----------------------------------------------------------------------------
[[noreturn]] NO_INLINE_ODR void rundown_fatal_panic(const char* reason) noexcept
{
    if (auto handler = g_rundown_fatal_error_handler.load(std::memory_order_acquire))
    {
        handler(reason);
    }
    else
    {
        std::cerr << "RUNDOWN_SHARED_PTR FATAL ERROR: " << reason << '\n';
    }
    std::abort();
}

// Evaluates condition and invokes rundown_fatal_panic(reason) on failure.
#define RUNDOWN_PTR_EXPECT(condition, reason) \
    do { \
        if (!(condition)) [[unlikely]] { rundown_fatal_panic(reason); } \
    } while(false)

// Number of cache-isolated per-thread reader slots per epoch shard (must be a power of two).
constexpr size_t SHAREDPTR_READER_SLOTS = 256;
static_assert((SHAREDPTR_READER_SLOTS & (SHAREDPTR_READER_SLOTS - 1)) == 0,
              "SHAREDPTR_READER_SLOTS must be a power of two for bitmask modulo.");

// Tracks global collector lifetime across static and thread_local teardown ordering.
// Cache-line aligned so adjacent globals in .bss never cause false sharing on fast-path checks.
alignas(CACHE_LINE_SIZE) inline std::atomic<int32_t> g_rundown_gc_alive{0};

// ============================================================================
// Forward Declarations
// ============================================================================
template <typename T> struct RundownControlBlock;
template <typename T> class  RundownSharedPtr;
template <typename T> class  PersistentSharedPtr;
template <typename T> class  BatchedRundownGuard;

template <typename T, typename... Args>
PersistentSharedPtr<T> MakeRundownSharedNuma(uint32_t preferred_node, Args&&... args);

template <typename T, typename... Args> 
PersistentSharedPtr<T> MakeRundownShared(Args&&... args);

// ============================================================================
// Asynchronous Try-Reclaim Rundown Primitive
// ============================================================================
class RundownRef
{
    // Decoupled states to eliminate false-sharing RMW contention between readers and the GC.
    // active_readers is mutated exclusively by the primary thread owning this slot,
    // enabling a single-writer non-RMW release store on the exit path.
    // shared_readers handles oversubscribed fallback threads and late TLS destructors.
    std::atomic<int32_t> active_readers{0}; // Single-writer counter mutated exclusively by the slot's primary owner thread
    std::atomic<int32_t> shared_readers{0}; // Multi-writer atomic RMW counter for oversubscribed threads and late TLS destructors

public:
    // ------------------------------------------------------------------------
    // acquire_exclusive
    // Purpose:
    //   Acquires rundown protection on the exclusive fast path for the primary
    //   thread owning this slot, enforcing sequential consistency before
    //   subsequent epoch validation and control-block loads.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void acquire_exclusive() noexcept
    {
        // Since TLS slot assignment grants exclusive primary ownership of active_readers,
        // no other thread mutates this counter.
        // Use memory_order_seq_cst so the slot increment is strictly ordered before
        // subsequent active_epoch validation and active_control_block loads.
#if defined(__aarch64__) || defined(_M_ARM64)
        // On ARM64, LDADDAL's store half is a Store-Release (STLR), which the ARMv8 memory
        // model permits to reorder with subsequent non-LDAR loads (e.g., relaxed LDR or
        // ARMv8.3+ LDAPR) to different addresses. Emit a relaxed load/store followed by a
        // DMB ISH barrier on ARM64 to enforce strict Store-Load Dekker ordering without
        // atomic RMW or acquire-release overhead.
        // If cur > 0 (reentrant/nested read on the same epoch shard), the outer guard
        // already published non-zero active_readers and executed the seq_cst fence.
        const int32_t cur = active_readers.load(std::memory_order_relaxed);
        active_readers.store(cur + 1, std::memory_order_relaxed);
        if (cur == 0) [[likely]]
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
#else
        active_readers.fetch_add(1, std::memory_order_seq_cst);
#endif
    }

    // ------------------------------------------------------------------------
    // release_exclusive
    // Purpose:
    //   Releases rundown protection on the exclusive fast path using a
    //   single-writer non-RMW release store.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void release_exclusive() noexcept
    {
        // Because the primary slot owner is the sole writer to active_readers, a plain
        // release store (MOV on x64, STLR on ARM64) safely decrements the counter without
        // issuing a second LOCK-prefixed RMW or draining the core's store buffer.
        const int32_t cur = active_readers.load(std::memory_order_relaxed);
        active_readers.store(cur - 1, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // acquire_shared
    // Purpose:
    //   Acquires rundown protection on the shared fallback path for threads
    //   sharing a slot during >256-thread oversubscription or late TLS teardown.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void acquire_shared() noexcept
    {
#if defined(__aarch64__) || defined(_M_ARM64)
        shared_readers.fetch_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
#else
        shared_readers.fetch_add(1, std::memory_order_seq_cst);
#endif
    }

    // ------------------------------------------------------------------------
    // release_shared
    // Purpose:
    //   Releases rundown protection on the shared fallback path via an atomic
    //   release decrement on shared_readers.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void release_shared() noexcept
    {
        // Note: We retain fetch_sub on shared_readers because the slot allocator permits
        // slot sharing when threads exceed SHAREDPTR_READER_SLOTS. Using fetch_sub on the
        // dedicated shared counter guarantees safety during extreme oversubscription without
        // penalizing the primary owner's exclusive fast path.
        shared_readers.fetch_sub(1, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // acquire
    // Purpose:
    //   Dispatches rundown acquisition to either the shared fallback counter
    //   or the primary owner's exclusive counter based on is_shared.
    // Parameters:
    //   is_shared - true if the calling thread shares this slot; false if it
    //               holds exclusive primary ownership.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void acquire(bool is_shared = false) noexcept
    {
        if (is_shared) [[unlikely]]
        {
            acquire_shared();
        }
        else
        {
            acquire_exclusive();
        }
    }

    // ------------------------------------------------------------------------
    // release
    // Purpose:
    //   Dispatches rundown release to either the shared fallback counter or
    //   the primary owner's exclusive counter based on is_shared.
    // Parameters:
    //   is_shared - true if the calling thread acquired the slot in shared mode;
    //               false if acquired in exclusive mode.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void release(bool is_shared = false) noexcept
    {
        if (is_shared) [[unlikely]]
        {
            release_shared();
        }
        else
        {
            release_exclusive();
        }
    }

    // ------------------------------------------------------------------------
    // try_reinitialize
    // Purpose:
    //   Checks whether both exclusive and shared reader counters on this slot
    //   have drained to zero so the epoch shard can be reclaimed.
    // Parameters:
    //   None.
    // Return Value:
    //   true if no active or shared readers remain on this slot; false otherwise.
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE bool try_reinitialize() const noexcept
    {
        return (active_readers.load(std::memory_order_acquire) |
                shared_readers.load(std::memory_order_acquire)) == 0;
    }
};

// ============================================================================
// Type-Erased Global Garbage Collection Pipeline & NUMA Slab Structures
// ============================================================================

// Reference-counted header placed at offset 0 of each 64 KB NUMA slab allocated via
// NumaAllocator::Allocate. Tracks the number of sub-allocated control blocks still
// alive (plus 1 baseline reference while the slab is the active bump arena on its
// NumaQuiescentSList shard) so the underlying OS virtual slab is cleanly unmapped
// only when the slab is detached AND its final block is released back to the OS.
struct alignas(CACHE_LINE_SIZE) NumaSlabHeader
{
    std::atomic<uint32_t> active_blocks{0}; // Count of active sub-blocks + 1 while active in shard bump arena
    uint32_t              numa_node{0};     // Physical NUMA node ID where this slab resides
    size_t                slab_size{0};     // Total byte size of the slab allocation for NumaAllocator::Free
};

// Base structure providing type erasure for the global GC lock-free stack
struct alignas(16) RetiredBlock
{
    // Lock-free singly linked list pointer for the background staging queue.
    RetiredBlock* next;
    
    // Type-erased function pointer to strictly invoke the correct payload destructor.
    void (*delete_callback)(RetiredBlock* base_block);

    // Owning NUMA slab pointer (or nullptr if allocated directly via aligned ::operator new).
    NumaSlabHeader* slab_owner;
    
    // Tracks anchoring status: 0 = Unanchored, 1 = Active in Anchor or Pending in GC.
    std::atomic<int32_t> anchor_state;

    // Physical NUMA node ID where this control block resides for node-pinned recycling.
    uint32_t numa_node;
};

// Intrusive header overlaid on recycled control blocks while inside the per-slot or per-node cache
struct FreeBlockNode
{
    FreeBlockNode*  next_in_batch;        // Pointer to the next free control block in this homogeneous batch
    FreeBlockNode*  next_batch;           // Pointer to the next batch in a per-node spare batch stack
    size_t          allocation_size;      // Total byte size of the control block and embedded payload allocation
    size_t          allocation_alignment; // Byte alignment of the control block allocation
    NumaSlabHeader* slab_owner;           // Owning reference-counted NUMA slab (or nullptr if heap-allocated)
    uint32_t        batch_count;          // Number of blocks linked from this node to the end of the chain (1..32)
    uint32_t        numa_node;            // Physical NUMA node ID where this control block resides
};

static_assert(sizeof(FreeBlockNode) <= CACHE_LINE_SIZE,
              "FreeBlockNode must fit within a single cache-aligned control block.");

// Cache-isolated per-slot block recycling bin to bypass OS heap lock contention.
// Embeds writer_node on the same cache line to prevent cross-slot false sharing.
struct alignas(CACHE_LINE_SIZE) SlotBlockCache
{
    std::atomic<FreeBlockNode*> active_batch{nullptr};    // Primary lock-free batch of recycled homogeneous control blocks
    std::atomic<FreeBlockNode*> spare_batch{nullptr};     // Secondary batch bin holding overflow or alternate-sized local blocks
    std::atomic<uint32_t>       writer_node{0xFFFFFFFFu}; // Physical NUMA node associated with this active writer slot
};

// Cache-isolated multi-batch spare stack guarded by a lightweight spinlock.
// Accessed only once per 32-block batch on slot cache misses or overflows, preventing
// stack-count underflow and transient starvation under concurrent heterogeneous lookups.
struct alignas(CACHE_LINE_SIZE) SpareBatchStack
{
    std::atomic<uint32_t> lock{0};       // Lightweight spinlock protecting head and count mutations
    FreeBlockNode*        head{nullptr}; // Head of the linked list of homogeneous 32-block batches
    std::atomic<uint32_t> count{0};      // Atomic batch count enabling lock-free empty/full pre-checks (0..MAX_NODE_SPARE_BATCHES)
};

// Cache-isolated per-NUMA-node SList shard, multi-batch spare stack, and 64 KB slab bump arena
// allocated in node-local physical RAM via NumaAllocator::Allocate.
struct alignas(CACHE_LINE_SIZE) NumaQuiescentSList
{
    // Node-local lock-free staging stack head for retired blocks pushed by cores on this node.
    std::atomic<RetiredBlock*> slist_head{nullptr};

    // Node-local shared multi-batch spare stack so recycled control blocks remain on their origin NUMA node.
    SpareBatchStack node_spare_stack{};

    // Node-local 64 KB slab bump arena state (accessed only on slot/node cache misses).
    // Guarded by slab_lock so heterogeneous payload types can safely bump-allocate single
    // blocks from the same 64 KB NUMA slab without overflowing BLOCK_CACHE_BATCH_SIZE.
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> slab_lock{0};          // Lightweight spinlock serializing slab bump-pointer refills on this node
    NumaSlabHeader*                                current_slab{nullptr}; // Currently active 64 KB NUMA slab being carved by this shard
    uintptr_t                                      slab_cursor{0};        // Current bump-allocation byte address within current_slab
    uintptr_t                                      slab_limit{0};         // End byte address (one past the end) of current_slab

    // Physical NUMA node ID associated with this shard.
    uint32_t node_id{0};
};

struct alignas(CACHE_LINE_SIZE) CacheAwareRundownSlot
{
    // The underlying rundown primitive padded aggressively to prevent false sharing.
    RundownRef rundown_ref;
};

// Two-Phase Global Epoch Shard
struct alignas(CACHE_LINE_SIZE) CacheAwareRundownRef
{
    // Array of cache-isolated slots protecting wait-free reader registration globally.
    CacheAwareRundownSlot slots[SHAREDPTR_READER_SLOTS];

    // ------------------------------------------------------------------------
    // get_slot
    // Purpose:
    //   Resolves the cache-isolated RundownRef instance for the specified
    //   reader slot index using power-of-two bitmask modulo.
    // Parameters:
    //   slot - Reader slot index to resolve.
    // Return Value:
    //   Pointer to the RundownRef for the requested slot.
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE RundownRef* get_slot(uint32_t slot) noexcept
    {
        return &slots[slot & (SHAREDPTR_READER_SLOTS - 1)].rundown_ref;
    }

    // ------------------------------------------------------------------------
    // try_release_and_reinitialize
    // Purpose:
    //   Scans all active reader slots up to max_slots to verify whether all
    //   transient readers on this epoch shard have completely drained.
    // Parameters:
    //   max_slots - Upper bound of active reader slots to scan
    //               (default: SHAREDPTR_READER_SLOTS).
    // Return Value:
    //   true if every scanned slot has zero active and shared readers;
    //   false if at least one reader is still executing in this epoch shard.
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE bool try_release_and_reinitialize(uint32_t max_slots = SHAREDPTR_READER_SLOTS) const noexcept
    {
        for (uint32_t i = 0; i < max_slots; ++i)
        {
            if (!slots[i].rundown_ref.try_reinitialize())
            {
                return false;
            }
        }

        return true;
    }
};

class SharedPtrGarbageCollector
{
    template <typename T> friend struct RundownControlBlock;
    template <typename T> friend class RundownSharedPtr;

    template <typename U, typename... Args>
    friend PersistentSharedPtr<U> MakeRundownSharedNuma(uint32_t preferred_node, Args&&... args);

    // Function pointer type for optional client-supplied GC wake notifications.
    using wake_callback_t = void (*)();

    // Number of 64-bit bitmask words required to represent SHAREDPTR_READER_SLOTS.
    static constexpr uint32_t SLOT_MASK_WORDS        = static_cast<uint32_t>(SHAREDPTR_READER_SLOTS / 64);

    // Maximum number of homogeneous control blocks chained in a single recycling batch.
    static constexpr uint32_t BLOCK_CACHE_BATCH_SIZE = 32;

    // Maximum number of 32-block batches buffered per NUMA node spare stack (64 * 32 = 2,048 blocks).
    static constexpr uint32_t MAX_NODE_SPARE_BATCHES = 64;

    // Total number of independent SList heads tracked in pending_free_lists (1 fallback + MAX_NUMA_NODES).
    static constexpr uint32_t PENDING_SLIST_COUNT    = NumaAllocator::MAX_NUMA_NODES + 1u;

    // Default NUMA slab allocation size (64 KB matches Windows VirtualAllocExNuma granularity
    // and amortizes Linux mmap/mbind across hundreds of control blocks).
    static constexpr size_t   NUMA_SLAB_BYTE_SIZE    = 65536;

    // Maximum alignment supported by the 64 KB NUMA slab bump allocator before falling back to ::operator new.
    static constexpr size_t   NUMA_SLAB_MAX_ALIGN    = 4096;

    // Flag bit encoded into the thread-local slot handle when a thread shares a slot
    // (either during >256-thread oversubscription or during late TLS destruction after
    // ThreadSlotOwner teardown).
    static constexpr uint32_t SLOT_SHARED_FLAG       = 0x80000000u;

    // Bitmask extracting the 0..255 slot index from a raw thread-local slot token.
    static constexpr uint32_t SLOT_INDEX_MASK        = static_cast<uint32_t>(SHAREDPTR_READER_SLOTS - 1);

    // Sentinel value indicating no slot is excluded during batch distribution.
    static constexpr uint32_t INVALID_SLOT_INDEX     = 0xFFFFFFFFu;

    // ------------------------------------------------------------------------
    // Cache-Line Group 1a: Read-Hot Global Epoch State
    // Isolated on its own cache line so concurrent writer pushes (slist_head),
    // slot allocations, and GC state transitions never invalidate reader-side cache lines.
    // ------------------------------------------------------------------------
    
    // The global, monotonically increasing epoch counter dictating active garbage collection cycles.
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> active_epoch{0};

    // Alternating inline epoch shards handling wait-free scalable reader registration.
    // Stored inline to eliminate pointer indirection on read() and prevent shutdown UAFs.
    alignas(CACHE_LINE_SIZE) CacheAwareRundownRef rundown_refs[2];

    // ------------------------------------------------------------------------
    // Cache-Line Group 1b: Thread Slot Allocation, Writer Tracking & NUMA Directory
    // Isolated from active_epoch so thread creation/exit and first-time writer
    // registration never cause false sharing with reader-hot epoch loads.
    // ------------------------------------------------------------------------

    // Monotonically increasing fallback index for overflow slot hashing.
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> next_slot_index{0};

    // Packs a 32-bit monotonic allocation generation counter in the upper 32 bits and
    // the active slot bound (1..SHAREDPTR_READER_SLOTS) in the lower 32 bits.
    // Versioning prevents ABA races between recompute_active_slot_bound() and acquire_slot_id().
    std::atomic<uint64_t> max_allocated_slot{1};

    // Count of live threads currently operating in >256-thread shared overflow mode.
    // When > 0, get_active_slot_bound() scans all SHAREDPTR_READER_SLOTS slots while
    // allowing exiting primary owners to immediately release their exclusive slots.
    std::atomic<uint32_t> overflow_slot_refs{0};

    // Bitmask tracking active exclusive thread reader slots so exiting threads recycle their slot IDs.
    std::atomic<uint64_t> allocated_slot_mask[SLOT_MASK_WORDS]{};

    // Bitmask tracking exclusive slots that have actively allocated control blocks (writers),
    // preventing deallocate_control_block() from stranding batches in reader-only slots.
    std::atomic<uint64_t> writer_slot_mask[SLOT_MASK_WORDS]{};

    // Directory of pointers to cache-aligned per-NUMA-node SList shards allocated via NumaAllocator.
    std::atomic<NumaQuiescentSList*> node_slists[NumaAllocator::MAX_NUMA_NODES]{};

    // ------------------------------------------------------------------------
    // Cache-Line Group 2a: Write-Hot Fallback Lock-Free Staging Stack Head
    // Isolated on its own dedicated cache line so queue_depth fetch_add
    // never invalidates concurrent slist_head compare_exchange_weak loops.
    // Acts as the primary SList on single-node UMA / macOS systems and the
    // zero-allocation fallback SList if per-node allocation fails on NUMA systems.
    // ------------------------------------------------------------------------

    // Head pointer for the lock-free staging stack of newly displaced blocks awaiting collection.
    alignas(CACHE_LINE_SIZE) std::atomic<RetiredBlock*> slist_head{nullptr};
    
    // ------------------------------------------------------------------------
    // Cache-Line Group 2b: Write-Hot Queue Depth & Decoupled Wake Callback State
    // Mutated after slist_head push completes.
    // ------------------------------------------------------------------------

    // Approximate count of displaced objects currently pending background reclamation processing.
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> queue_depth{0};

    // Atomic flag ensuring the wake callback is explicitly signaled only once per accumulation cycle.
    std::atomic<int32_t> wake_requested{0};

    // The queue depth threshold required to successfully trigger the wake callback.
    std::atomic<int32_t> high_watermark{512};

    // Client-provided callback invoked when the uncollected staging queue exceeds the watermark.
    std::atomic<wake_callback_t> wake_callback{nullptr};

    // ------------------------------------------------------------------------
    // Cache-Line Group 3a: Background GC Pipeline State Machine
    // Accessed only by background GC workers holding is_gc_active (or teardown).
    // ------------------------------------------------------------------------

    // Detached SList heads actively transitioning through the two-phase epoch swap pipeline.
    // Stored as a compact array of non-empty per-shard heads mutated exclusively under is_gc_active
    // so epoch advancement is O(nodes) with zero pointer chasing and zero empty-slot scans.
    alignas(CACHE_LINE_SIZE) RetiredBlock* pending_free_lists[PENDING_SLIST_COUNT]{};

    // Number of non-empty SList heads currently populated in pending_free_lists (0..PENDING_SLIST_COUNT).
    uint32_t pending_list_count{0};
    
    // The global epoch snapshot corresponding to the blocks currently in pending_free_lists.
    int64_t pending_epoch{0};
    
    // Atomic lock state strictly preventing concurrent background GC pipeline execution overlap.
    // 0 = Idle, 1 = Active, 2 = Active with a deferred non-flush wake that arrived while busy.
    std::atomic<int32_t> is_gc_active{0};

    // ------------------------------------------------------------------------
    // Cache-Line Group 3b: Control-Block Batch Distribution & UMA Spare Stack
    // Isolated on its own cache line so batch recycling operations do not
    // invalidate background GC state.
    // ------------------------------------------------------------------------

    // Round-robin cursor distributing recycled control-block batches across active slots.
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> batch_distribute_cursor{0};

    // Shared multi-batch spare stack used on single-node UMA systems (or when node_slists[node] is null).
    SpareBatchStack uma_spare_stack{};

    // Per-slot lock-free control block caches eliminating OS heap lock contention under write load.
    SlotBlockCache slot_block_caches[SHAREDPTR_READER_SLOTS];

    // Releases a single non-null control block back to its reference-counted NUMA slab
    // (unmapping the slab when its final block is freed) or via aligned ::operator delete.
    static FORCE_INLINE void free_raw_control_block(void*           ptr,
                                                    size_t          alignment,
                                                    NumaSlabHeader* slab) noexcept
    {
        if (slab)
        {
            const uint32_t prev_blocks = slab->active_blocks.fetch_sub(1, std::memory_order_acq_rel);
            RUNDOWN_PTR_EXPECT(prev_blocks >= 1u, "NUMA slab active_blocks underflow");
            if (prev_blocks == 1u)
            {
                const size_t total_size = slab->slab_size;
                slab->~NumaSlabHeader();
                NumaAllocator::Free(static_cast<void*>(slab), total_size);
            }
        }
        else
        {
            ::operator delete(ptr, std::align_val_t{alignment});
        }
    }

    // Walks a linked chain of cached FreeBlockNodes within a single batch and releases each block.
    static void free_block_chain(FreeBlockNode* head) noexcept
    {
        while (head)
        {
            FreeBlockNode*  next      = head->next_in_batch;
            const size_t    alignment = head->allocation_alignment;
            NumaSlabHeader* slab      = head->slab_owner;
            free_raw_control_block(static_cast<void*>(head), alignment, slab);
            head = next;
        }
    }

    // Walks a multi-batch stack linked via next_batch and frees each homogeneous batch chain.
    static void free_batch_stack(FreeBlockNode* stack_head) noexcept
    {
        while (stack_head)
        {
            FreeBlockNode* next_b  = stack_head->next_batch;
            stack_head->next_batch = nullptr;
            free_block_chain(stack_head);
            stack_head = next_b;
        }
    }

    // Acquires the lightweight spinlock on a SpareBatchStack.
    static FORCE_INLINE void lock_spare_stack(SpareBatchStack& stack) noexcept
    {
        AdaptiveBackoff backoff;
        uint32_t        expected = 0;
        while (!stack.lock.compare_exchange_weak(expected,
                                                 1u,
                                                 std::memory_order_acquire,
                                                 std::memory_order_relaxed))
        {
            expected = 0;
            backoff();
        }
    }

    // Releases the lightweight spinlock on a SpareBatchStack.
    static FORCE_INLINE void unlock_spare_stack(SpareBatchStack& stack) noexcept
    {
        stack.lock.store(0u, std::memory_order_release);
    }

    // Pushes a homogeneous batch onto a bounded per-node or UMA spare batch stack.
    // Performs a lock-free relaxed capacity pre-check before acquiring stack.lock so
    // overflow batches never contend on the spinlock when the stack is already full.
    static bool try_push_spare_batch(SpareBatchStack& stack,
                                     FreeBlockNode*   batch) noexcept
    {
        if (stack.count.load(std::memory_order_relaxed) >= MAX_NODE_SPARE_BATCHES) [[unlikely]]
        {
            return false;
        }

        lock_spare_stack(stack);
        const uint32_t cur_count = stack.count.load(std::memory_order_relaxed);
        if (cur_count >= MAX_NODE_SPARE_BATCHES) [[unlikely]]
        {
            unlock_spare_stack(stack);
            return false;
        }

        batch->next_batch = stack.head;
        stack.head        = batch;
        stack.count.store(cur_count + 1u, std::memory_order_relaxed);
        unlock_spare_stack(stack);
        return true;
    }

    // Extracts the first batch matching (size, alignment, target_node) from a SpareBatchStack
    // while preserving any non-matching heterogeneous batches in place without transient starvation.
    // Performs a lock-free relaxed empty pre-check so cache misses on an empty spare stack
    // bypass spinlock acquisition completely.
    static FreeBlockNode* try_pop_matching_spare_batch(SpareBatchStack& stack,
                                                       size_t           size,
                                                       size_t           alignment,
                                                       uint32_t         target_node) noexcept
    {
        if (stack.count.load(std::memory_order_relaxed) == 0u)
        {
            return nullptr;
        }

        lock_spare_stack(stack);
        FreeBlockNode** prev_next = &stack.head;
        FreeBlockNode*  curr      = stack.head;

        while (curr)
        {
            if (curr->allocation_size == size &&
                curr->allocation_alignment == alignment &&
                curr->numa_node == target_node) [[likely]]
            {
                *prev_next       = curr->next_batch;
                curr->next_batch = nullptr;
                stack.count.store(stack.count.load(std::memory_order_relaxed) - 1u,
                                  std::memory_order_relaxed);
                unlock_spare_stack(stack);
                return curr;
            }

            prev_next = &curr->next_batch;
            curr      = curr->next_batch;
        }

        unlock_spare_stack(stack);
        return nullptr;
    }

    // Atomically detaches all batches from a SpareBatchStack and resets its count to zero.
    static FreeBlockNode* detach_all_spare_batches(SpareBatchStack& stack) noexcept
    {
        if (stack.count.load(std::memory_order_relaxed) == 0u)
        {
            return nullptr;
        }

        lock_spare_stack(stack);
        FreeBlockNode* detached = stack.head;
        stack.head              = nullptr;
        stack.count.store(0u, std::memory_order_relaxed);
        unlock_spare_stack(stack);
        return detached;
    }

    // Atomically detaches and frees both the active and spare block batches for a single slot.
    void flush_single_slot_block_cache(uint32_t slot) noexcept
    {
        FreeBlockNode* active = slot_block_caches[slot].active_batch.exchange(nullptr, std::memory_order_acquire);
        FreeBlockNode* spare  = slot_block_caches[slot].spare_batch.exchange(nullptr, std::memory_order_acquire);

        free_block_chain(active);
        free_block_chain(spare);
    }

    // Flushes and frees all cached control block batches across all reader slots, the UMA
    // spare stack, and all NUMA shards (releasing any completely idle active bump slabs).
    void flush_block_caches() noexcept
    {
        for (uint32_t i = 0; i < SHAREDPTR_READER_SLOTS; ++i)
        {
            flush_single_slot_block_cache(i);
        }

        free_batch_stack(detach_all_spare_batches(uma_spare_stack));

        const NumaAllocator::TopologyState& topo = NumaAllocator::GetTopology();
        for (size_t i = 0; i < topo.size(); ++i)
        {
            const uint32_t node = topo[i];
            if (node >= NumaAllocator::MAX_NUMA_NODES) [[unlikely]]
            {
                continue;
            }

            NumaQuiescentSList* shard = node_slists[node].load(std::memory_order_acquire);
            if (shard)
            {
                free_batch_stack(detach_all_spare_batches(shard->node_spare_stack));

                // If the shard's current bump slab has no live sub-allocated blocks remaining
                // (active_blocks == 1 baseline ref), release the slab back to the OS.
                AdaptiveBackoff backoff;
                uint32_t        expected_lock = 0;
                while (!shard->slab_lock.compare_exchange_weak(expected_lock,
                                                               1u,
                                                               std::memory_order_acquire,
                                                               std::memory_order_relaxed))
                {
                    expected_lock = 0;
                    backoff();
                }

                NumaSlabHeader* idle_slab = nullptr;
                if (shard->current_slab &&
                    shard->current_slab->active_blocks.load(std::memory_order_acquire) == 1u)
                {
                    idle_slab           = shard->current_slab;
                    shard->current_slab = nullptr;
                    shard->slab_cursor  = 0;
                    shard->slab_limit   = 0;
                }

                shard->slab_lock.store(0u, std::memory_order_release);

                if (idle_slab)
                {
                    free_raw_control_block(static_cast<void*>(idle_slab), CACHE_LINE_SIZE, idle_slab);
                }
            }
        }
    }

    // Lazily allocates or initializes per-NUMA-node SList shards in node-local physical memory.
    void initialize_numa_slists() noexcept
    {
        const NumaAllocator::TopologyState& topo = NumaAllocator::GetTopology();
        if (!topo.is_numa_available)
        {
            return;
        }

        for (size_t i = 0; i < topo.size(); ++i)
        {
            const uint32_t node = topo[i];
            if (node < NumaAllocator::MAX_NUMA_NODES &&
                node_slists[node].load(std::memory_order_acquire) == nullptr)
            {
                void* raw_mem = NumaAllocator::Allocate(sizeof(NumaQuiescentSList), node);
                if (raw_mem) [[likely]]
                {
                    NumaQuiescentSList* shard = ::new (raw_mem) NumaQuiescentSList();
                    shard->node_id            = node;

                    NumaQuiescentSList* expected = nullptr;
                    if (!node_slists[node].compare_exchange_strong(expected,
                                                                   shard,
                                                                   std::memory_order_release,
                                                                   std::memory_order_relaxed))
                    {
                        shard->~NumaQuiescentSList();
                        NumaAllocator::Free(raw_mem, sizeof(NumaQuiescentSList));
                    }
                }
            }
        }
    }

    // Drains and invokes delete_callback on a single linked list of RetiredBlocks.
    static void drain_retired_list(RetiredBlock* head) noexcept
    {
        while (head)
        {
            RetiredBlock* next_entry = head->next;
            head->next               = nullptr;
            head->anchor_state.store(0, std::memory_order_release);
            head->delete_callback(head);
            head = next_entry;
        }
    }

    // Synchronously drains and reclaims any remaining retired blocks in pending_free_lists,
    // the fallback slist_head, and all per-NUMA-node SList shards during final shutdown.
    void reclaim_remaining_slists() noexcept
    {
        while (true)
        {
            bool           any_reclaimed  = false;
            const uint32_t count_to_drain = pending_list_count;
            pending_list_count            = 0;

            for (uint32_t s = 0; s < count_to_drain; ++s)
            {
                RetiredBlock* pending = pending_free_lists[s];
                pending_free_lists[s] = nullptr;
                if (pending)
                {
                    any_reclaimed = true;
                    drain_retired_list(pending);
                }
            }

            RetiredBlock* fallback = slist_head.exchange(nullptr, std::memory_order_acq_rel);
            if (fallback)
            {
                any_reclaimed = true;
                drain_retired_list(fallback);
            }

            const NumaAllocator::TopologyState& topo = NumaAllocator::GetTopology();
            if (topo.is_numa_available)
            {
                for (size_t i = 0; i < topo.size(); ++i)
                {
                    const uint32_t node = topo[i];
                    if (node >= NumaAllocator::MAX_NUMA_NODES) [[unlikely]]
                    {
                        continue;
                    }

                    NumaQuiescentSList* shard = node_slists[node].load(std::memory_order_acquire);
                    if (shard)
                    {
                        RetiredBlock* node_batch = shard->slist_head.exchange(nullptr, std::memory_order_acq_rel);
                        if (node_batch)
                        {
                            any_reclaimed = true;
                            drain_retired_list(node_batch);
                        }
                    }
                }
            }

            if (!any_reclaimed)
            {
                break;
            }
        }
    }

    // Detaches and destroys all per-NUMA-node SList shards, releasing their spare batches and active bump slabs.
    void teardown_numa_slists() noexcept
    {
        for (uint32_t node = 0; node < NumaAllocator::MAX_NUMA_NODES; ++node)
        {
            NumaQuiescentSList* shard = node_slists[node].exchange(nullptr, std::memory_order_acq_rel);
            if (shard)
            {
                RetiredBlock* stranded = shard->slist_head.exchange(nullptr, std::memory_order_acquire);
                drain_retired_list(stranded);

                free_batch_stack(detach_all_spare_batches(shard->node_spare_stack));

                // Release the shard's baseline reference on its active bump slab (if any)
                if (shard->current_slab)
                {
                    NumaSlabHeader* active_slab = shard->current_slab;
                    shard->current_slab         = nullptr;
                    shard->slab_cursor          = 0;
                    shard->slab_limit           = 0;
                    free_raw_control_block(static_cast<void*>(active_slab), CACHE_LINE_SIZE, active_slab);
                }

                shard->~NumaQuiescentSList();
                NumaAllocator::Free(static_cast<void*>(shard), sizeof(NumaQuiescentSList));
            }
        }
    }

    // Resolves the SList head for the calling processor's local NUMA node, falling back to
    // the global zero-allocation slist_head on UMA systems or if node allocation failed.
    [[nodiscard]] FORCE_INLINE std::atomic<RetiredBlock*>& get_local_node_slist_head() noexcept
    {
        const NumaAllocator::TopologyState& topo = NumaAllocator::GetTopology();
        if (topo.is_numa_available)
        {
            const uint32_t current_node = NumaAllocator::GetCurrentNode();
            if (current_node < NumaAllocator::MAX_NUMA_NODES) [[likely]]
            {
                NumaQuiescentSList* shard = node_slists[current_node].load(std::memory_order_acquire);
                if (shard) [[likely]]
                {
                    return shard->slist_head;
                }
            }
        }

        return slist_head;
    }

    // Atomically detaches the fallback SList and all active per-NUMA-node SList shards into
    // a compact prefix of pending_free_lists in O(nodes) time with zero pointer chasing.
    [[nodiscard]] bool stage_all_slists_to_pending() noexcept
    {
        uint32_t      count          = 0;
        RetiredBlock* fallback_batch = slist_head.exchange(nullptr, std::memory_order_acq_rel);
        if (fallback_batch)
        {
            pending_free_lists[count++] = fallback_batch;
        }

        const NumaAllocator::TopologyState& topo = NumaAllocator::GetTopology();
        if (topo.is_numa_available)
        {
            for (size_t i = 0; i < topo.size() && count < PENDING_SLIST_COUNT; ++i)
            {
                const uint32_t node = topo[i];
                if (node >= NumaAllocator::MAX_NUMA_NODES) [[unlikely]]
                {
                    continue;
                }

                NumaQuiescentSList* shard = node_slists[node].load(std::memory_order_acquire);
                if (shard) [[likely]]
                {
                    RetiredBlock* node_batch = shard->slist_head.exchange(nullptr, std::memory_order_acq_rel);
                    if (node_batch)
                    {
                        pending_free_lists[count++] = node_batch;
                    }
                }
            }
        }

        pending_list_count = count;
        return count > 0;
    }

    // Number of blocks carved at once from a 64 KB NUMA slab when refilling a local slot.
    // Set to half of BLOCK_CACHE_BATCH_SIZE (16) so the 15 cached blocks leave 17 slots of
    // headroom in active_batch before deallocate_control_block() hits 32 and flushes.
    static constexpr uint32_t SLAB_CARVE_BATCH_SIZE  = BLOCK_CACHE_BATCH_SIZE / 2;

    // Allocate up to SLAB_CARVE_BATCH_SIZE (16) cache-aligned control blocks from
    // target_node's active 64 KB NUMA slab (or 1 block when allocating for a remote node
    // or shared overflow slot). Amortizes slab_lock and active_blocks atomic updates by 16x
    // without overflowing BLOCK_CACHE_BATCH_SIZE (32).
    [[nodiscard]] void* allocate_from_numa_slab(size_t              size,
                                                size_t              alignment,
                                                uint32_t            target_node,
                                                uint32_t            slot,
                                                bool                cache_extra_in_slot,
                                                NumaQuiescentSList* shard,
                                                NumaSlabHeader*&    out_slab,
                                                uint32_t&           out_node) noexcept
    {
        if (alignment > NUMA_SLAB_MAX_ALIGN) [[unlikely]]
        {
            out_slab = nullptr;
            out_node = target_node;
            return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
        }

        // Oversized payloads (> half a 64 KB slab) use a dedicated single-block NUMA slab
        // without acquiring slab_lock or evicting a partially populated shard->current_slab.
        if (size > (NUMA_SLAB_BYTE_SIZE / 2)) [[unlikely]]
        {
            const size_t header_space = (sizeof(NumaSlabHeader) + alignment - 1) & ~(alignment - 1);
            const size_t slab_bytes   = header_space + size + alignment;

            void* raw_slab = NumaAllocator::Allocate(slab_bytes, target_node);
            if (!raw_slab) [[unlikely]]
            {
                out_slab = nullptr;
                out_node = target_node;
                return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
            }

            NumaSlabHeader* dedicated_slab = ::new (raw_slab) NumaSlabHeader();
            dedicated_slab->numa_node      = target_node;
            dedicated_slab->slab_size      = slab_bytes;
            dedicated_slab->active_blocks.store(1u, std::memory_order_relaxed);

            const uintptr_t base_addr    = reinterpret_cast<uintptr_t>(raw_slab);
            const uintptr_t aligned_addr = (base_addr + sizeof(NumaSlabHeader) + alignment - 1) &
                                           ~(static_cast<uintptr_t>(alignment) - 1);

            out_slab = dedicated_slab;
            out_node = target_node;
            return reinterpret_cast<void*>(aligned_addr);
        }

        const size_t   block_stride = (size + alignment - 1) & ~(alignment - 1);
        const uint32_t desired_cnt  = cache_extra_in_slot ? SLAB_CARVE_BATCH_SIZE : 1u;

        // Acquire the node shard's lightweight bump lock (contended only on slot/node cache misses)
        AdaptiveBackoff backoff;
        uint32_t        expected_lock = 0;
        while (!shard->slab_lock.compare_exchange_weak(expected_lock,
                                                       1u,
                                                       std::memory_order_acquire,
                                                       std::memory_order_relaxed))
        {
            expected_lock = 0;
            backoff();
        }

        uintptr_t       first_addr      = 0;
        uint32_t        allocated_count = 0;
        NumaSlabHeader* slab_used       = nullptr;
        NumaSlabHeader* full_slab       = nullptr;

        if (shard->current_slab) [[likely]]
        {
            const uintptr_t aligned_addr = (shard->slab_cursor + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
            if (aligned_addr <= shard->slab_limit &&
                size <= static_cast<size_t>(shard->slab_limit - aligned_addr)) [[likely]]
            {
                const size_t avail_bytes = static_cast<size_t>(shard->slab_limit - aligned_addr);
                const size_t max_fit     = 1u + (avail_bytes - size) / block_stride;
                allocated_count          = static_cast<uint32_t>(std::min<size_t>(desired_cnt, max_fit));

                first_addr         = aligned_addr;
                shard->slab_cursor = aligned_addr + static_cast<uintptr_t>(allocated_count - 1u) * block_stride + size;
                slab_used          = shard->current_slab;

                slab_used->active_blocks.fetch_add(allocated_count, std::memory_order_relaxed);
            }
            else
            {
                // Current slab is full: detach now and drop the shard's baseline reference (1)
                // after releasing slab_lock so OS virtual-memory syscalls never run under a spinlock.
                full_slab           = shard->current_slab;
                shard->current_slab = nullptr;
                shard->slab_cursor  = 0;
                shard->slab_limit   = 0;
            }
        }

        shard->slab_lock.store(0u, std::memory_order_release);

        if (full_slab)
        {
            free_raw_control_block(static_cast<void*>(full_slab), CACHE_LINE_SIZE, full_slab);
        }

        if (!slab_used)
        {
            const size_t header_space = (sizeof(NumaSlabHeader) + alignment - 1) & ~(alignment - 1);
            const size_t min_required = header_space + size + alignment;
            const size_t slab_bytes   = std::max<size_t>(NUMA_SLAB_BYTE_SIZE, min_required);

            void* raw_slab = NumaAllocator::Allocate(slab_bytes, target_node);
            if (!raw_slab) [[unlikely]]
            {
                out_slab = nullptr;
                out_node = target_node;

                return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
            }

            NumaSlabHeader* new_slab = ::new (raw_slab) NumaSlabHeader();
            new_slab->numa_node      = target_node;
            new_slab->slab_size      = slab_bytes;

            const uintptr_t base_addr    = reinterpret_cast<uintptr_t>(raw_slab);
            const uintptr_t aligned_addr = (base_addr + sizeof(NumaSlabHeader) + alignment - 1) &
                                           ~(static_cast<uintptr_t>(alignment) - 1);
            const uintptr_t limit_addr   = base_addr + slab_bytes;

            const size_t avail_bytes = static_cast<size_t>(limit_addr - aligned_addr);
            const size_t max_fit     = 1u + (avail_bytes - size) / block_stride;
            allocated_count          = static_cast<uint32_t>(std::min<size_t>(desired_cnt, max_fit));

            const uintptr_t next_cursor = aligned_addr + static_cast<uintptr_t>(allocated_count - 1u) * block_stride + size;

            // Briefly re-acquire slab_lock solely to install new_slab if no other thread refilled it concurrently.
            AdaptiveBackoff install_backoff;
            expected_lock = 0;
            while (!shard->slab_lock.compare_exchange_weak(expected_lock,
                                                           1u,
                                                           std::memory_order_acquire,
                                                           std::memory_order_relaxed))
            {
                expected_lock = 0;
                install_backoff();
            }

            if (!shard->current_slab) [[likely]]
            {
                // Initialize active_blocks to 1 (for shard->current_slab baseline) + allocated_count
                new_slab->active_blocks.store(1u + allocated_count, std::memory_order_relaxed);

                shard->current_slab = new_slab;
                shard->slab_cursor  = next_cursor;
                shard->slab_limit   = limit_addr;
            }
            else
            {
                // Another thread already installed a fresh slab while we executed NumaAllocator::Allocate.
                // Retain our carved blocks with no shard baseline reference so new_slab unmaps once freed.
                new_slab->active_blocks.store(allocated_count, std::memory_order_relaxed);
            }

            shard->slab_lock.store(0u, std::memory_order_release);

            first_addr = aligned_addr;
            slab_used  = new_slab;
        }

        // Link any extra carved blocks [1 .. allocated_count - 1] (at most 15 blocks)
        // into the calling thread's local slot cache outside slab_lock.
        if (allocated_count > 1u) [[likely]]
        {
            FreeBlockNode* chain_head = nullptr;
            uint32_t       chain_len  = 0;

            for (uint32_t i = allocated_count - 1u; i >= 1u; --i)
            {
                void* block_mem = reinterpret_cast<void*>(first_addr + static_cast<uintptr_t>(i) * block_stride);

                FreeBlockNode* node        = ::new (block_mem) FreeBlockNode;
                node->next_in_batch        = chain_head;
                node->next_batch           = nullptr;
                node->allocation_size      = size;
                node->allocation_alignment = alignment;
                node->slab_owner           = slab_used;
                node->numa_node            = target_node;
                node->batch_count          = ++chain_len;
                chain_head                 = node;
            }

            FreeBlockNode* expected = nullptr;
            if (!slot_block_caches[slot].active_batch.compare_exchange_strong(expected,
                                                                              chain_head,
                                                                              std::memory_order_release,
                                                                              std::memory_order_relaxed))
            {
                expected = nullptr;
                if (!slot_block_caches[slot].spare_batch.compare_exchange_strong(expected,
                                                                                 chain_head,
                                                                                 std::memory_order_release,
                                                                                 std::memory_order_relaxed))
                {
                    distribute_batch_to_node(chain_head, slot);
                }
            }
        }

        out_slab = slab_used;
        out_node = target_node;
        return reinterpret_cast<void*>(first_addr);
    }

    // Routes an evicted or completed homogeneous batch of FreeBlockNodes to an active
    // writer slot on the same NUMA node, its origin NUMA node's multi-batch spare stack,
    // or the UMA spare batch stack.
    void distribute_batch_to_node(FreeBlockNode* batch,
                                  uint32_t       exclude_slot = INVALID_SLOT_INDEX) noexcept
    {
        batch->next_batch          = nullptr;
        const uint32_t target_node = batch->numa_node;
        const bool     is_full     = (batch->batch_count >= BLOCK_CACHE_BATCH_SIZE);

        // 1. Try active writer slots on the same NUMA node that have an empty spare (or active,
        //    for partial batches) slot, excluding the caller's own slot to prevent self-bouncing
        //    and unbounded batch-chain snowballing.
        const uint32_t max_slots = get_active_slot_bound();
        const uint32_t start_idx = batch_distribute_cursor.fetch_add(1, std::memory_order_relaxed);

        for (uint32_t i = 0; i < max_slots; ++i)
        {
            const uint32_t idx = (start_idx + i) % max_slots;
            if (idx == exclude_slot)
            {
                continue;
            }

            const uint32_t word     = idx / 64u;
            const uint64_t bit_mask = (1ULL << (idx & 63u));

            if ((writer_slot_mask[word].load(std::memory_order_relaxed) & bit_mask) == 0)
            {
                continue;
            }

            if (slot_block_caches[idx].writer_node.load(std::memory_order_relaxed) != target_node)
            {
                continue;
            }

            if (slot_block_caches[idx].spare_batch.load(std::memory_order_relaxed) == nullptr)
            {
                FreeBlockNode* expected = nullptr;
                if (slot_block_caches[idx].spare_batch.compare_exchange_strong(expected,
                                                                               batch,
                                                                               std::memory_order_release,
                                                                               std::memory_order_relaxed))
                {
                    return;
                }
            }

            // Never inject a full 32-block batch into another slot's active_batch (which
            // would cause immediate eviction/ping-pong if that slot is currently freeing).
            if (!is_full && slot_block_caches[idx].active_batch.load(std::memory_order_relaxed) == nullptr)
            {
                FreeBlockNode* expected = nullptr;
                if (slot_block_caches[idx].active_batch.compare_exchange_strong(expected,
                                                                                batch,
                                                                                std::memory_order_release,
                                                                                std::memory_order_relaxed))
                {
                    return;
                }
            }
        }

        // 2. Push onto the target NUMA node's multi-batch spare stack (or UMA spare stack if unallocated)
        NumaQuiescentSList* shard        = (target_node < NumaAllocator::MAX_NUMA_NODES)
                                               ? node_slists[target_node].load(std::memory_order_acquire)
                                               : nullptr;
        SpareBatchStack&    target_stack = shard ? shard->node_spare_stack : uma_spare_stack;
        if (!try_push_spare_batch(target_stack, batch))
        {
            free_block_chain(batch);
        }
    }

    // Atomically bumps the 32-bit allocation generation counter and raises the active slot upper bound.
    FORCE_INLINE void update_max_allocated_slot(uint32_t bound) noexcept
    {
        // Increment the upper 32-bit generation counter on every slot acquisition
        // while monotonically raising the lower 32-bit bound. Even if bound <= prev_bound,
        // bumping the generation counter forces any concurrent recompute_active_slot_bound()
        // to detect the newly acquired slot and re-scan allocated_slot_mask.
        uint64_t prev_state = max_allocated_slot.load(std::memory_order_relaxed);
        while (true)
        {
            const uint32_t prev_bound = static_cast<uint32_t>(prev_state & 0xFFFFFFFFULL);
            const uint64_t next_gen   = (prev_state >> 32) + 1ULL;
            const uint32_t next_bound = std::max<uint32_t>(prev_bound, bound);
            const uint64_t next_state = (next_gen << 32) | static_cast<uint64_t>(next_bound);

            if (max_allocated_slot.compare_exchange_weak(prev_state,
                                                         next_state,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_relaxed))
            {
                break;
            }
        }
    }

    // Scans allocated_slot_mask during synchronous flushes to tighten max_allocated_slot if high-indexed threads exited.
    void recompute_active_slot_bound() noexcept
    {
        uint64_t observed_state = max_allocated_slot.load(std::memory_order_acquire);

        while (true)
        {
            const uint32_t observed_max  = static_cast<uint32_t>(observed_state & 0xFFFFFFFFULL);
            uint32_t       highest_bound = 1;

            if (overflow_slot_refs.load(std::memory_order_acquire) > 0u)
            {
                highest_bound = static_cast<uint32_t>(SHAREDPTR_READER_SLOTS);
            }
            else
            {
                for (int32_t w = static_cast<int32_t>(SLOT_MASK_WORDS) - 1; w >= 0; --w)
                {
                    const uint64_t mask = allocated_slot_mask[w].load(std::memory_order_acquire);
                    if (mask != 0)
                    {
                        const uint32_t bit = 63u - static_cast<uint32_t>(std::countl_zero(mask));
                        highest_bound      = static_cast<uint32_t>(w) * 64u + bit + 1u;
                        break;
                    }
                }
            }

            if (highest_bound >= observed_max)
            {
                break;
            }

            const uint64_t next_gen   = (observed_state >> 32) + 1ULL;
            const uint64_t next_state = (next_gen << 32) | static_cast<uint64_t>(highest_bound);

            // Use compare_exchange_weak on the versioned 64-bit state so a concurrent
            // acquire_slot_id() that raises or reuses a lower slot index is never overwritten
            // with a stale lower bound.
            if (max_allocated_slot.compare_exchange_weak(observed_state,
                                                         next_state,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire))
            {
                break;
            }
        }
    }

    // Claims an exclusive reader slot ID from allocated_slot_mask, or attaches in shared mode
    // (SLOT_SHARED_FLAG) if all 256 exclusive slots are currently occupied.
    uint32_t acquire_slot_id() noexcept
    {
        for (uint32_t word = 0; word < SLOT_MASK_WORDS; ++word)
        {
            uint64_t mask = allocated_slot_mask[word].load(std::memory_order_relaxed);
            while (~mask != 0)
            {
                const uint32_t bit      = static_cast<uint32_t>(std::countr_zero(~mask));
                const uint64_t bit_mask = (1ULL << bit);
                if (allocated_slot_mask[word].compare_exchange_weak(mask,
                                                                    mask | bit_mask,
                                                                    std::memory_order_acq_rel,
                                                                    std::memory_order_relaxed))
                {
                    const uint32_t slot  = word * 64u + bit;
                    const uint32_t bound = slot + 1u;

                    slot_block_caches[slot].writer_node.store(NumaAllocator::GetCurrentNode(), std::memory_order_relaxed);
                    update_max_allocated_slot(bound);

                    // Return the slot index without SLOT_SHARED_FLAG to grant exclusive
                    // primary ownership of active_readers and slot_block_caches[slot].
                    return slot;
                }
            }
        }

        // Fallback if more than SHAREDPTR_READER_SLOTS threads are concurrently active.
        // Track overflow readers via overflow_slot_refs so exiting exclusive owners can
        // immediately release their allocated_slot_mask bit for new exclusive threads.
        overflow_slot_refs.fetch_add(1u, std::memory_order_acq_rel);
        const uint32_t slot = next_slot_index.fetch_add(1u, std::memory_order_relaxed) & (SHAREDPTR_READER_SLOTS - 1);
        update_max_allocated_slot(static_cast<uint32_t>(SHAREDPTR_READER_SLOTS));

        return slot | SLOT_SHARED_FLAG;
    }

    // Releases a thread slot upon thread exit: decrements overflow_slot_refs for shared
    // overflow threads, or flushes the slot cache and clears allocated_slot_mask for exclusive owners.
    void release_slot_id(uint32_t raw_slot) noexcept
    {
        if ((raw_slot & SLOT_SHARED_FLAG) != 0u)
        {
            const uint32_t prev_overflow = overflow_slot_refs.fetch_sub(1u, std::memory_order_acq_rel);
            RUNDOWN_PTR_EXPECT(prev_overflow >= 1u, "Reader slot overflow_slot_refs underflow");
            return;
        }

        const uint32_t slot     = raw_slot & SLOT_INDEX_MASK;
        const uint32_t word     = slot / 64u;
        const uint64_t bit_mask = (1ULL << (slot & 63u));

        // Clear writer_slot_mask BEFORE flushing the slot cache so concurrent
        // distribute_batch_to_node() calls stop targeting this exiting slot while
        // its batches are being freed, and BEFORE clearing allocated_slot_mask so a
        // newly arriving exclusive thread on this slot cannot have its state overwritten.
        writer_slot_mask[word].fetch_and(~bit_mask, std::memory_order_acq_rel);

        // Flush any cached control blocks held in this exclusive slot so exiting threads
        // do not leave partial batches stranded in idle slots.
        flush_single_slot_block_cache(slot);

        allocated_slot_mask[word].fetch_and(~bit_mask, std::memory_order_release);
    }

    // RAII helper managing the lifetime of a thread's assigned reader slot across thread_local teardown.
    struct ThreadSlotOwner
    {
        uint32_t  raw_slot;     // Assigned slot token (including SLOT_SHARED_FLAG if shared)
        uint32_t* tls_slot_ptr; // Pointer to the calling thread's fast-path TLS slot variable

        // Acquires a reader slot ID for the current thread and binds the fast-path TLS variable pointer.
        explicit ThreadSlotOwner(uint32_t* slot_ptr) noexcept : raw_slot(SharedPtrGarbageCollector::get_instance().acquire_slot_id()),
                                                                tls_slot_ptr(slot_ptr)
        {
        }

        // Redirects subsequent late TLS destructors on this thread to shared slot 0 and releases the acquired slot ID.
        ~ThreadSlotOwner() noexcept
        {
            // Route any subsequent thread_local destructors on this exiting thread
            // to shared slot 0 (which is always covered by get_active_slot_bound() >= 1
            // and bypasses per-slot block caching) before releasing raw_slot.
            *tls_slot_ptr = SLOT_SHARED_FLAG;

            if (g_rundown_gc_alive.load(std::memory_order_acquire) != 0)
            {
                SharedPtrGarbageCollector::get_instance().release_slot_id(raw_slot);
            }
        }
    };

    // Cold slow-path helper that initializes the thread_local ThreadSlotOwner on a thread's first touch.
    NO_INLINE_ODR static uint32_t allocate_thread_reader_slot_slow(uint32_t* tls_slot_ptr) noexcept
    {
        thread_local ThreadSlotOwner owner(tls_slot_ptr);
        return owner.raw_slot;
    }

    // Fast, collision-free per-thread reader slot assignment using a constant-initialized
    // thread_local index that compiles to a direct TLS load on the hot path and encodes
    // SLOT_SHARED_FLAG in bit 31 if the slot is shared.
    [[nodiscard]] FORCE_INLINE uint32_t get_thread_reader_slot_raw() noexcept
    {
        thread_local uint32_t tls_slot = 0xFFFFFFFFu;
        if (tls_slot == 0xFFFFFFFFu) [[unlikely]]
        {
            tls_slot = allocate_thread_reader_slot_slow(&tls_slot);
        }

        return tls_slot;
    }

    // Returns the current upper bound (1 .. SHAREDPTR_READER_SLOTS) of allocated reader slots for GC scans.
    [[nodiscard]] FORCE_INLINE uint32_t get_active_slot_bound() const noexcept
    {
        if (overflow_slot_refs.load(std::memory_order_acquire) > 0u) [[unlikely]]
        {
            return static_cast<uint32_t>(SHAREDPTR_READER_SLOTS);
        }

        const uint32_t assigned = static_cast<uint32_t>(max_allocated_slot.load(std::memory_order_acquire) & 0xFFFFFFFFULL);
        return std::min<uint32_t>(assigned, static_cast<uint32_t>(SHAREDPTR_READER_SLOTS));
    }

    // Discovers active NUMA topology, pre-allocates node-local SList shards, and marks the collector alive.
    void initialize_internal() noexcept
    {
        // All per-slot caches, bitmasks, and rundown_refs are zero/default-initialized via member initializers.
        initialize_numa_slists();

        g_rundown_gc_alive.store(1, std::memory_order_release);
    }

    // Constructs the singleton garbage collector instance and initializes internal state.
    SharedPtrGarbageCollector() noexcept
    {
        initialize_internal();
    }

    // Allocates an aligned control block and payload buffer, prioritizing the calling thread's
    // exclusive local-node slot cache, then the target node's multi-batch spare stack, and
    // finally a 64 KB NUMA slab bump arena (or aligned ::operator new on UMA / macOS systems).
    [[nodiscard]] FORCE_INLINE void* allocate_control_block(size_t           size,
                                                            size_t           alignment,
                                                            uint32_t         preferred_node,
                                                            NumaSlabHeader*& out_slab,
                                                            uint32_t&        out_node) noexcept
    {
        const NumaAllocator::TopologyState& topo          = NumaAllocator::GetTopology();
        const uint32_t                      current_node  = NumaAllocator::GetCurrentNode();
        const uint32_t                      target_node   = (preferred_node != NumaAllocator::ANY_NODE_OK &&
                                                             topo.IsValidNode(preferred_node)) ? preferred_node : current_node;
        const bool                          is_local_node = (target_node == current_node);

        if (g_rundown_gc_alive.load(std::memory_order_relaxed) != 0) [[likely]]
        {
            const uint32_t raw_slot          = get_thread_reader_slot_raw();
            const bool     is_exclusive_slot = (raw_slot & SLOT_SHARED_FLAG) == 0u;
            const uint32_t slot              = raw_slot & SLOT_INDEX_MASK;

            // Only use per-slot caches when the calling thread holds exclusive ownership of its slot
            if (is_exclusive_slot) [[likely]]
            {
                const uint32_t word     = slot / 64u;
                const uint64_t bit_mask = (1ULL << (slot & 63u));

                if (slot_block_caches[slot].writer_node.load(std::memory_order_relaxed) != current_node) [[unlikely]]
                {
                    slot_block_caches[slot].writer_node.store(current_node, std::memory_order_relaxed);
                }

                if ((writer_slot_mask[word].load(std::memory_order_relaxed) & bit_mask) == 0) [[unlikely]]
                {
                    writer_slot_mask[word].fetch_or(bit_mask, std::memory_order_release);
                }

                // Only consult the thread-local slot cache when allocating on the thread's local NUMA node
                // so explicit remote-node allocations never evict the calling thread's local-node cache.
                if (is_local_node) [[likely]]
                {
                    FreeBlockNode* batch = slot_block_caches[slot].active_batch.exchange(nullptr, std::memory_order_acquire);
                    if (batch &&
                        batch->allocation_size == size &&
                        batch->allocation_alignment == alignment &&
                        batch->numa_node == current_node) [[likely]]
                    {
                        FreeBlockNode* next = batch->next_in_batch;
                        if (next)
                        {
                            next->next_batch        = nullptr;
                            FreeBlockNode* expected = nullptr;
                            if (!slot_block_caches[slot].active_batch.compare_exchange_strong(expected,
                                                                                              next,
                                                                                              std::memory_order_release,
                                                                                              std::memory_order_relaxed))
                            {
                                expected = nullptr;
                                if (!slot_block_caches[slot].spare_batch.compare_exchange_strong(expected,
                                                                                                 next,
                                                                                                 std::memory_order_release,
                                                                                                 std::memory_order_relaxed))
                                {
                                    distribute_batch_to_node(next, slot);
                                }
                            }
                        }

                        out_slab = batch->slab_owner;
                        out_node = current_node;
                        return static_cast<void*>(batch);
                    }

                    // Active batch was empty, foreign-node, or a different payload size: check spare_batch
                    FreeBlockNode* spare = slot_block_caches[slot].spare_batch.exchange(nullptr, std::memory_order_acquire);
                    if (batch)
                    {
                        if (batch->numa_node == current_node)
                        {
                            FreeBlockNode* expected_spare = nullptr;
                            if (!slot_block_caches[slot].spare_batch.compare_exchange_strong(expected_spare,
                                                                                             batch,
                                                                                             std::memory_order_release,
                                                                                             std::memory_order_relaxed))
                            {
                                distribute_batch_to_node(batch, slot);
                            }
                        }
                        else
                        {
                            distribute_batch_to_node(batch, slot);
                        }
                    }

                    if (spare)
                    {
                        if (spare->allocation_size == size &&
                            spare->allocation_alignment == alignment &&
                            spare->numa_node == current_node) [[likely]]
                        {
                            FreeBlockNode* next = spare->next_in_batch;
                            if (next)
                            {
                                next->next_batch        = nullptr;
                                FreeBlockNode* expected = nullptr;
                                if (!slot_block_caches[slot].active_batch.compare_exchange_strong(expected,
                                                                                                  next,
                                                                                                  std::memory_order_release,
                                                                                                  std::memory_order_relaxed))
                                {
                                    expected = nullptr;
                                    if (!slot_block_caches[slot].spare_batch.compare_exchange_strong(expected,
                                                                                                     next,
                                                                                                     std::memory_order_release,
                                                                                                     std::memory_order_relaxed))
                                    {
                                        distribute_batch_to_node(next, slot);
                                    }
                                }
                            }

                            out_slab = spare->slab_owner;
                            out_node = current_node;
                            return static_cast<void*>(spare);
                        }

                        distribute_batch_to_node(spare, slot);
                    }
                }
            }

            // Secondary fast path: Check the target NUMA node's multi-batch spare stack (or UMA spare stack)
            NumaQuiescentSList* shard = (topo.is_numa_available && target_node < NumaAllocator::MAX_NUMA_NODES)
                                            ? node_slists[target_node].load(std::memory_order_acquire)
                                            : nullptr;
            SpareBatchStack&    spare_stack = shard ? shard->node_spare_stack : uma_spare_stack;
            FreeBlockNode*      spare_batch = try_pop_matching_spare_batch(spare_stack,
                                                                           size,
                                                                           alignment,
                                                                           target_node);
            if (spare_batch)
            {
                FreeBlockNode* next = spare_batch->next_in_batch;
                if (next)
                {
                    next->next_batch        = nullptr;
                    FreeBlockNode* expected = nullptr;
                    if (!is_exclusive_slot ||
                        !is_local_node ||
                        !slot_block_caches[slot].active_batch.compare_exchange_strong(expected,
                                                                                      next,
                                                                                      std::memory_order_release,
                                                                                      std::memory_order_relaxed))
                    {
                        if (!try_push_spare_batch(spare_stack, next))
                        {
                            free_block_chain(next);
                        }
                    }
                }

                out_slab = spare_batch->slab_owner;
                out_node = spare_batch->numa_node;
                return static_cast<void*>(spare_batch);
            }

            if (shard) [[likely]]
            {
                // Slow path on multi-node NUMA systems:
                // Bump-allocate up to 16 blocks (if exclusive & local) or 1 block (if remote/shared)
                // from target_node's 64 KB slab.
                return allocate_from_numa_slab(size,
                                               alignment,
                                               target_node,
                                               slot,
                                               is_exclusive_slot && is_local_node,
                                               shard,
                                               out_slab,
                                               out_node);
            }
        }

        out_slab = nullptr;
        out_node = target_node;
        return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
    }

    // Recycles a destroyed control block into the calling thread's per-slot homogeneous batch
    // cache (up to BLOCK_CACHE_BATCH_SIZE = 32) when the thread owns an exclusive slot, or
    // routes full/shared/evicted batches to their origin node's spare stack.
    FORCE_INLINE void deallocate_control_block(void*           buffer,
                                               size_t          size,
                                               size_t          alignment,
                                               NumaSlabHeader* slab_owner = nullptr,
                                               uint32_t        numa_node  = 0) noexcept
    {
        if (g_rundown_gc_alive.load(std::memory_order_relaxed) == 0) [[unlikely]]
        {
            free_raw_control_block(buffer, alignment, slab_owner);
            return;
        }

        FreeBlockNode* node        = ::new (buffer) FreeBlockNode;
        node->next_batch           = nullptr;
        node->allocation_size      = size;
        node->allocation_alignment = alignment;
        node->slab_owner           = slab_owner;
        node->numa_node            = numa_node;

        const uint32_t raw_slot          = get_thread_reader_slot_raw();
        const bool     is_exclusive_slot = (raw_slot & SLOT_SHARED_FLAG) == 0u;

        if (!is_exclusive_slot) [[unlikely]]
        {
            node->next_in_batch = nullptr;
            node->batch_count   = 1u;
            distribute_batch_to_node(node);
            return;
        }

        const uint32_t slot           = raw_slot & SLOT_INDEX_MASK;
        const uint32_t word           = slot / 64u;
        const uint64_t bit_mask       = (1ULL << (slot & 63u));
        const bool     is_writer_slot = (writer_slot_mask[word].load(std::memory_order_relaxed) & bit_mask) != 0;
        FreeBlockNode* chain          = slot_block_caches[slot].active_batch.exchange(nullptr, std::memory_order_acquire);

        if (chain && (chain->allocation_size != size ||
                      chain->allocation_alignment != alignment ||
                      chain->numa_node != numa_node ||
                      chain->batch_count >= BLOCK_CACHE_BATCH_SIZE)) [[unlikely]]
        {
            // Preserve the existing homogeneous chain in spare_batch only if this slot is an
            // active writer on the chain's local NUMA node; otherwise distribute it immediately
            // so dedicated non-writer GC threads never strand partial batches in spare_batch.
            if (is_writer_slot &&
                chain->numa_node == slot_block_caches[slot].writer_node.load(std::memory_order_relaxed))
            {
                FreeBlockNode* expected_spare = nullptr;
                if (!slot_block_caches[slot].spare_batch.compare_exchange_strong(expected_spare,
                                                                                 chain,
                                                                                 std::memory_order_release,
                                                                                 std::memory_order_relaxed))
                {
                    distribute_batch_to_node(chain, slot);
                }
            }
            else
            {
                distribute_batch_to_node(chain, slot);
            }

            chain = nullptr;
        }

        node->next_in_batch = chain;
        node->batch_count   = chain ? (chain->batch_count + 1u) : 1u;

        // Accumulate homogeneous chains up to BLOCK_CACHE_BATCH_SIZE (32) regardless of
        // whether the freeing thread is on the same node, so cross-node GC sweeps batch
        // all 32 blocks before handing the completed batch to the origin NUMA node.
        if (node->batch_count < BLOCK_CACHE_BATCH_SIZE) [[likely]]
        {
            FreeBlockNode* expected = nullptr;
            if (slot_block_caches[slot].active_batch.compare_exchange_strong(expected,
                                                                             node,
                                                                             std::memory_order_release,
                                                                             std::memory_order_relaxed))
            {
                return;
            }
        }

        // Full 32-block batch accumulated (or active_batch was concurrently populated):
        // Active writer threads may retain one local-node full batch in spare_batch before
        // routing overflow batches to other writer slots or the origin NUMA node's spare stack.
        if (is_writer_slot &&
            numa_node == slot_block_caches[slot].writer_node.load(std::memory_order_relaxed))
        {
            FreeBlockNode* expected_spare = nullptr;
            if (slot_block_caches[slot].spare_batch.compare_exchange_strong(expected_spare,
                                                                            node,
                                                                            std::memory_order_release,
                                                                            std::memory_order_relaxed))
            {
                return;
            }
        }

        // Route the batch to its origin NUMA node's spare stack or another active writer slot on that node.
        distribute_batch_to_node(node, slot);
    }

    // Performs a sequentially consistent load of the current 64-bit monotonic active epoch
    // for reader slot validation.
    FORCE_INLINE int64_t get_active_epoch() const noexcept
    {
        // Use memory_order_seq_cst (MOV on x64, LDAR on ARM64) so ARMv8.3+ CPUs do not
        // emit LDAPR and reorder the epoch load ahead of the rundown acquire barrier.
        return active_epoch.load(std::memory_order_seq_cst);
    }

    // Performs an acquire-load of the current 64-bit monotonic active epoch for the initial
    // optimistic shard routing guess prior to slot acquisition.
    FORCE_INLINE int64_t get_active_epoch_relaxed() const noexcept
    {
        // Used strictly for the optimistic initial routing guess to avoid hardware barriers.
        // The subsequent validation load MUST use get_active_epoch() to enforce seq_cst.
        return active_epoch.load(std::memory_order_acquire);
    }

    // Maps a 64-bit epoch number to its corresponding inline CacheAwareRundownRef shard (0 or 1).
    FORCE_INLINE CacheAwareRundownRef* get_epoch_rundown_ref(int64_t epoch) noexcept
    {
        const int32_t shard = static_cast<int32_t>(epoch & 1);
        return &rundown_refs[shard];
    }

    // Pushes a displaced control block onto the calling processor's NUMA-local lock-free SList
    // shard (or fallback SList) and triggers wake_callback if queue depth reaches high_watermark.
    FORCE_INLINE void push(RetiredBlock* retired_block) noexcept
    {
        if (g_rundown_gc_alive.load(std::memory_order_acquire) == 0) [[unlikely]]
        {
            retired_block->anchor_state.store(0, std::memory_order_release);
            retired_block->delete_callback(retired_block);
            return;
        }

        // Route displaced block to the calling processor's NUMA-local SList shard
        std::atomic<RetiredBlock*>& target_slist = get_local_node_slist_head();

        // Lock-free push
        RetiredBlock* head = target_slist.load(std::memory_order_relaxed);
        do
        {
            retired_block->next = head;
        } while (!target_slist.compare_exchange_weak(head, retired_block, std::memory_order_release, std::memory_order_relaxed));

        wake_callback_t callback = wake_callback.load(std::memory_order_acquire);
        
        if (callback)
        {
            const uint32_t prev_depth = queue_depth.fetch_add(1u, std::memory_order_relaxed);
            const uint32_t depth      = prev_depth + 1u;
            const uint32_t watermark  = static_cast<uint32_t>(high_watermark.load(std::memory_order_acquire));

            if (depth >= watermark || prev_depth == 0xFFFFFFFFu) [[unlikely]]
            {
                if (wake_requested.load(std::memory_order_relaxed) == 0)
                {
                    int32_t expected = 0;
                    if (wake_requested.compare_exchange_strong(expected, 1, std::memory_order_acq_rel))
                    {
                        callback();
                    }
                }
            }
        }
    }

public:
    SharedPtrGarbageCollector(const SharedPtrGarbageCollector&) = delete;
    SharedPtrGarbageCollector& operator=(const SharedPtrGarbageCollector&) = delete;
    SharedPtrGarbageCollector(SharedPtrGarbageCollector&&) = delete;
    SharedPtrGarbageCollector& operator=(SharedPtrGarbageCollector&&) = delete;

    // ------------------------------------------------------------------------
    // get_instance
    // Purpose:
    //   Retrieves the process-wide singleton instance of the garbage collector.
    // Parameters:
    //   None.
    // Return Value:
    //   Reference to the global SharedPtrGarbageCollector singleton.
    // ------------------------------------------------------------------------
    static SharedPtrGarbageCollector& get_instance() noexcept
    {
        static SharedPtrGarbageCollector instance;
        return instance;
    }

    // ------------------------------------------------------------------------
    // ~SharedPtrGarbageCollector (Destructor)
    // Purpose:
    //   Marks the collector as unloading, synchronously flushes all pending
    //   retired blocks and block caches through the GC pipeline, reclaims any
    //   final straggler SList entries, and tears down all NUMA SList shards.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    ~SharedPtrGarbageCollector() noexcept
    {
        g_rundown_gc_alive.store(0, std::memory_order_release);

        process_garbage_pipeline(true);
        reclaim_remaining_slists();
        teardown_numa_slists();
    }

    // ------------------------------------------------------------------------
    // set_wake_callback
    // Purpose:
    //   Registers an optional client-provided callback triggered when the
    //   uncollected staging queue depth reaches or exceeds the specified
    //   high-watermark.
    // Parameters:
    //   callback  - Optional function pointer invoked to wake a background GC worker.
    //   watermark - Queue depth threshold that triggers callback (default: 1000, clamped >= 1).
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    void set_wake_callback(void (*callback)(), 
                           int32_t watermark = 1000) noexcept
    {
        high_watermark.store(std::max<int32_t>(1, watermark), std::memory_order_release);
        wake_callback.store(callback, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // process_garbage_pipeline
    // Purpose:
    //   Executes the asynchronous two-phase epoch advancement and batch
    //   reclamation pipeline. Executed periodically by a background GC thread
    //   or synchronously during flush/shutdown.
    // Parameters:
    //   is_flush - If true, waits to acquire GC exclusivity, spins until all
    //              pending epochs and cascading retired blocks drain, flushes
    //              block caches, and recomputes the active slot bound. If false,
    //              performs a non-blocking try-reclaim and single epoch swap.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    void process_garbage_pipeline(bool is_flush = false) noexcept
    {
        int32_t expected = 0;
        while (!is_gc_active.compare_exchange_weak(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            if (!is_flush)
            {
                // Record a deferred wake (1 -> 2) if another worker currently owns the pipeline
                // so a callback dispatched while is_gc_active == 1 cannot leave wake_requested
                // permanently latched at 1 after the active worker exits.
                if (expected == 2)
                {
                    return;
                }

                if (expected == 1 &&
                    is_gc_active.compare_exchange_strong(expected, 2, std::memory_order_acq_rel, std::memory_order_acquire))
                {
                    return;
                }

                if (expected == 2)
                {
                    return;
                }

                // Active worker just released is_gc_active back to 0: retry acquiring 0 -> 1 directly.
                expected = 0;
                continue;
            }

            // Atomically acquire is_gc_active when synchronously flushing on teardown.
            // Must use compare_exchange_weak rather than a passive load so that
            // is_gc_active is transitioned back to 1 before this thread enters the pipeline.
            AdaptiveBackoff backoff;

            do
            {
                expected = 0;
                backoff();
            } while (!is_gc_active.compare_exchange_weak(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire));
            break;
        }

        bool            swapped_this_cycle = false;
        AdaptiveBackoff flush_backoff;

        do
        {
            // 1. Process asynchronous lists from previous epoch swaps to prevent GC pipeline overlap
            if (pending_list_count > 0)
            {
                const int64_t         current_pending_epoch = pending_epoch;
                const int32_t         old_shard             = static_cast<int32_t>(current_pending_epoch & 1);
                CacheAwareRundownRef* old_ref               = &rundown_refs[old_shard];
                const uint32_t        max_slots             = get_active_slot_bound();

                bool drained = old_ref->try_release_and_reinitialize(max_slots);
                if (!drained && !is_flush && swapped_this_cycle)
                {
                    // Brief hardware grace pause right after advancing active_epoch so normal
                    // nanosecond-scale in-flight readers complete on the current wake cycle.
                    for (uint32_t spin = 0; spin < 8u; ++spin)
                    {
                        HARDWARE_PAUSE();
                        if (old_ref->try_release_and_reinitialize(max_slots))
                        {
                            drained = true;
                            break;
                        }
                    }
                }

                if (!drained) [[unlikely]]
                {
                    if (!is_flush)
                    {
                        // Head-of-Line Blocking Protection:
                        // If a reader is stalled on the old epoch, we MUST exit the pipeline.
                        // Reset queue_depth before clearing wake_requested so event-driven
                        // callers of set_wake_callback() are neither deadlocked nor flooded
                        // with a callback storm on every single subsequent push(), but instead
                        // re-signaled once another watermark batch of writes accumulates.
                        queue_depth.store(0u, std::memory_order_relaxed);
                        wake_requested.store(0, std::memory_order_release);
                        const int32_t prev_state = is_gc_active.exchange(0, std::memory_order_seq_cst);

                        if (prev_state == 2 && wake_requested.load(std::memory_order_acquire) != 0)
                        {
                            if (wake_callback_t callback = wake_callback.load(std::memory_order_acquire))
                            {
                                callback();
                            }
                        }

                        return;
                    }
                    
                    flush_backoff();
                    continue;
                }

                // Batch-delete all blocks across all staged NUMA shards with zero further OS sync calls
                const uint32_t count_to_drain = pending_list_count;
                pending_list_count            = 0;
                for (uint32_t s = 0; s < count_to_drain; ++s)
                {
                    RetiredBlock* pending_free = pending_free_lists[s];
                    pending_free_lists[s]      = nullptr;
                    drain_retired_list(pending_free);
                }
            }

            // Background processing executes at most one swap cycle per wake to avoid
            // continuously spinning, but loops back once after advancing active_epoch
            // to immediately drain pending_free_lists if the old epoch is already idle.
            if (!is_flush && swapped_this_cycle)
            {
                break;
            }

            // 2. Attempt to ingest newly retired objects across all NUMA shards in O(nodes)
            if (!stage_all_slists_to_pending())
            {
                // Reset queue_depth BEFORE clearing wake_requested so delayed writers that
                // incremented queue_depth after a prior stage cannot trigger an infinite
                // wake_callback() storm on empty SLists during the exit re-check.
                queue_depth.store(0u, std::memory_order_relaxed);
                wake_requested.store(0, std::memory_order_release);
                break;
            }

            // Optimization: depth reset BEFORE clearing wake_requested so concurrent
            // writers pushing right after wake_requested is cleared do not read a stale
            // pre-flush depth and fire a spurious wake_callback().
            queue_depth.store(0u, std::memory_order_relaxed);

            // Now that all SLists and queue_depth are clear, it is safe to reset wake_requested.
            // Writers can signal again when the NEW queue hits the watermark.
            wake_requested.store(0, std::memory_order_release);

            const int64_t old_epoch = active_epoch.load(std::memory_order_acquire);
            
            // 3. Register state prior to advancing global epoch to secure two-phase serialization
            pending_epoch = old_epoch;

            // Advance active_epoch with seq_cst ordering so the epoch store is globally
            // visible before slot active_readers scans.
            active_epoch.store(old_epoch + 1, std::memory_order_seq_cst);

            // 4. Emit a full sequential consistency barrier before polling
            // old_ref->try_release_and_reinitialize() on the next loop iteration.
            // On ARM64 (ARMv8.3+ RCPC), a seq_cst store (STLR) followed by an acquire
            // load (LDAPR) to a different address can be reordered by hardware without
            // a DMB ISH fence.
            std::atomic_thread_fence(std::memory_order_seq_cst);

            swapped_this_cycle = true;

        } while (true);

        if (is_flush)
        {
            flush_block_caches();
            recompute_active_slot_bound();
        }

        const int32_t prev_state = is_gc_active.exchange(0, std::memory_order_seq_cst);

        // Re-check deferred wakes and queue depth to catch pushes that happened while
        // is_gc_active was held or during our exit window.
        wake_callback_t callback = wake_callback.load(std::memory_order_acquire);
        if (callback)
        {
            if (prev_state == 2 && wake_requested.load(std::memory_order_acquire) != 0)
            {
                callback();
            }
            else if (queue_depth.load(std::memory_order_acquire) >=
                     static_cast<uint32_t>(high_watermark.load(std::memory_order_acquire)))
            {
                int32_t expected_wake = 0;
                if (wake_requested.compare_exchange_strong(expected_wake, 1, std::memory_order_acq_rel))
                {
                    callback();
                }
            }
        }
    }
};

// ============================================================================
// Cache-Aware Dual-Refcount Control Block
// The payload object T is placed at a compile-time constant offset immediately
// following RundownControlBlock<T> on its own cache line. Computing get_payload()
// via pointer arithmetic eliminates a dependent memory load and guarantees that
// read() never touches the RundownControlBlock<T> cache line.
// ============================================================================
template <typename T>
struct alignas(CACHE_LINE_SIZE) RundownControlBlock : public RetiredBlock
{
    static_assert(!std::is_void_v<T> && !std::is_reference_v<T> && !std::is_array_v<T>,
                  "RundownControlBlock<T> requires a non-void, non-reference, non-array object type T.");

    // Standard atomic reference count securely tracking long-lived PersistentSharedPtr checkouts.
    std::atomic<int32_t> persistent_ref_count; 

    // ------------------------------------------------------------------------
    // get_allocation_alignment
    // Purpose:
    //   Computes the compile-time byte alignment required for the single-block
    //   control block and payload allocation (at least CACHE_LINE_SIZE).
    // Parameters:
    //   None.
    // Return Value:
    //   Compile-time byte alignment for RundownControlBlock<T> and payload T.
    // ------------------------------------------------------------------------
    static constexpr size_t get_allocation_alignment() noexcept
    {
        return std::max<size_t>(CACHE_LINE_SIZE, alignof(T));
    }

    // ------------------------------------------------------------------------
    // get_payload_offset
    // Purpose:
    //   Computes the constant cache-aligned byte offset where MakeRundownShared
    //   embeds the T payload relative to the start of RundownControlBlock<T>.
    //   Allows fast-path readers to resolve the T* address via pointer
    //   arithmetic without loading the control block's mutable cache line.
    // Parameters:
    //   None.
    // Return Value:
    //   Compile-time byte offset from the control block base to the T payload.
    // ------------------------------------------------------------------------
    static constexpr size_t get_payload_offset() noexcept
    {
        constexpr size_t alignment = get_allocation_alignment();
        return (sizeof(RundownControlBlock<T>) + alignment - 1) & ~(alignment - 1);
    }

    // ------------------------------------------------------------------------
    // get_payload
    // Purpose:
    //   Zero-load constant-offset calculation: resolves the embedded payload
    //   object address without touching the control block cache line. Uses
    //   std::launder to satisfy C++17/20 strict object-lifetime rules across
    //   recycled storage.
    // Parameters:
    //   None.
    // Return Value:
    //   Pointer to the constructed payload object of type T.
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE T* get_payload() noexcept
    {
        return std::launder(reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(this) + get_payload_offset()));
    }

    // ------------------------------------------------------------------------
    // add_persistent_reference
    // Purpose:
    //   Atomically increments the persistent reference count with overflow
    //   protection against reference-count wrap-around use-after-free.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void add_persistent_reference() noexcept
    {
        // Caller already holds a valid PersistentSharedPtr or active rundown guard, so
        // relaxed ordering suffices (avoids an unnecessary LDADDAL barrier on ARM64).
        // Inspect the prior value directly to avoid signed overflow UB on INT32_MAX + 1.
        const int32_t prev = persistent_ref_count.fetch_add(1, std::memory_order_relaxed);
        if (prev <= 0 || prev == (std::numeric_limits<int32_t>::max)()) [[unlikely]]
        {
            rundown_fatal_panic("Persistent reference count overflow");
        }
    }

    // ------------------------------------------------------------------------
    // destroy_and_deallocate
    // Purpose:
    //   Destroys the payload and control block and routes the underlying
    //   storage back to its NUMA-affinitized slot/node cache or
    //   reference-counted NUMA slab.
    // Parameters:
    //   self - Pointer to the RundownControlBlock<T> instance to destroy and free.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    static FORCE_INLINE void destroy_and_deallocate(RundownControlBlock<T>* self) noexcept
    {
        NumaSlabHeader* const slab = self->slab_owner;
        const uint32_t        node = self->numa_node;

        self->~RundownControlBlock();

        constexpr size_t allocation_alignment  = get_allocation_alignment();
        constexpr size_t total_allocation_size = get_payload_offset() + sizeof(T);

        if (g_rundown_gc_alive.load(std::memory_order_acquire) != 0) [[likely]]
        {
            SharedPtrGarbageCollector::get_instance().deallocate_control_block(static_cast<void*>(self),
                                                                               total_allocation_size,
                                                                               allocation_alignment,
                                                                               slab,
                                                                               node);
        }
        else
        {
            SharedPtrGarbageCollector::free_raw_control_block(static_cast<void*>(self),
                                                              allocation_alignment,
                                                              slab);
        }
    }

    // ------------------------------------------------------------------------
    // release_persistent_reference
    // Purpose:
    //   Atomically decrements the persistent reference count and destroys both
    //   the payload and control block when the count reaches zero, with
    //   underflow protection.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void release_persistent_reference() noexcept
    {
        const int32_t prev = persistent_ref_count.fetch_sub(1, std::memory_order_acq_rel);
        if (prev <= 0) [[unlikely]]
        {
            rundown_fatal_panic("Double free detected: persistent reference count dropped below zero");
        }

        if (prev == 1)
        {
            destroy_and_deallocate(this);
        }
    }

    // ------------------------------------------------------------------------
    // destroy_control_block_callback
    // Purpose:
    //   Type-erased static callback invoked by the garbage collector after
    //   epoch rundown completes to release the global anchor's baseline
    //   persistent reference (fast-pathing destruction when the count is 1).
    // Parameters:
    //   base_block - Pointer to the RetiredBlock base of the control block.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    static void destroy_control_block_callback(RetiredBlock* base_block) noexcept
    {
        RundownControlBlock<T>* control_block = static_cast<RundownControlBlock<T>*>(base_block);
        
        // The Garbage Collection Pipeline is dropping the Global Anchor's baseline reference.
        // Fast path: once the epoch grace period has completed, no reader can promote this
        // block via load_persistent(). If persistent_ref_count is 1, the anchor holds the
        // sole remaining reference. Using memory_order_acquire synchronizes with any prior
        // PersistentSharedPtr destructor fetch_sub(acq_rel) that dropped the count from 2 -> 1.
        if (control_block->persistent_ref_count.load(std::memory_order_acquire) == 1) [[likely]]
        {
            destroy_and_deallocate(control_block);
        }
        else
        {
            control_block->release_persistent_reference();
        }
    }

    // ------------------------------------------------------------------------
    // RundownControlBlock (Constructor)
    // Purpose:
    //   Initializes the control block with a baseline persistent reference
    //   count of 1, binds the type-erased GC deletion callback, records the
    //   owning NUMA slab and node ID, and sets anchor_state to Unanchored (0).
    // Parameters:
    //   slab - Pointer to the owning NumaSlabHeader, or nullptr if heap-allocated.
    //   node - Physical NUMA node ID where this control block resides.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    explicit RundownControlBlock(NumaSlabHeader* slab = nullptr,
                                 uint32_t        node = 0) noexcept : persistent_ref_count(1) // Factory/Anchor takes immediate baseline ownership
    {
        this->next            = nullptr;
        this->delete_callback = destroy_control_block_callback;
        this->slab_owner      = slab;
        this->anchor_state.store(0, std::memory_order_relaxed);
        this->numa_node       = node;
    }

    // ------------------------------------------------------------------------
    // ~RundownControlBlock (Destructor)
    // Purpose:
    //   Explicitly invokes the destructor of the placement-new'd payload
    //   object prior to recycling or freeing the underlying memory block.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    ~RundownControlBlock() noexcept
    {
        // Explicitly invoke payload destructor since memory was placement-new'd
        get_payload()->~T();
    }

    RundownControlBlock(const RundownControlBlock&) = delete;
    RundownControlBlock& operator=(const RundownControlBlock&) = delete;
    void operator delete(void*) = delete;
};

// ============================================================================
// PersistentSharedPtr
// The Persistent Accessor. Uses standard atomic counters. Use this for 
// stream contexts or pended callbacks. It completely escapes the Rundown pipeline.
// ============================================================================
template <typename T>
class PersistentSharedPtr
{
    // Pointer to the shared control block anchoring the promoted persistent object state.
    RundownControlBlock<T>* control_block;

    friend class RundownSharedPtr<T>;

    template <typename U, typename... Args>
    friend PersistentSharedPtr<U> MakeRundownSharedNuma(uint32_t preferred_node, Args&&... args);
    
    template <typename U, typename... Args>
    friend PersistentSharedPtr<U> MakeRundownShared(Args&&... args);

    // Internal constructor utilized by MakeRundownShared factory (adopt_ownership == true)
    // or RundownSharedPtr::load_persistent() (adopt_ownership == false).
    explicit PersistentSharedPtr(RundownControlBlock<T>* block,
                                 bool                    adopt_ownership) noexcept : control_block(block)
    {
        if (!adopt_ownership && control_block)
        {
            control_block->add_persistent_reference();
        }
    }

public:
    // ------------------------------------------------------------------------
    // PersistentSharedPtr (Default Constructor)
    // Purpose:
    //   Constructs an empty persistent shared pointer managing no control block.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    constexpr PersistentSharedPtr() noexcept : control_block(nullptr) 
    {
    }
    
    // ------------------------------------------------------------------------
    // PersistentSharedPtr (nullptr Constructor)
    // Purpose:
    //   Constructs an empty persistent shared pointer from nullptr.
    // Parameters:
    //   nullptr - Null pointer literal.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    constexpr PersistentSharedPtr(decltype(nullptr)) noexcept : control_block(nullptr) 
    {
    }

    // ------------------------------------------------------------------------
    // ~PersistentSharedPtr (Destructor)
    // Purpose:
    //   Releases the managed persistent reference count if non-empty.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    ~PersistentSharedPtr() noexcept
    {
        if (control_block)
        {
            control_block->release_persistent_reference();
        }
    }

    // ------------------------------------------------------------------------
    // PersistentSharedPtr (Copy Constructor)
    // Purpose:
    //   Shares persistent ownership of the control block managed by other by
    //   atomically incrementing its persistent reference count.
    // Parameters:
    //   other - Source persistent shared pointer to copy.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    PersistentSharedPtr(const PersistentSharedPtr& other) noexcept : control_block(other.control_block)
    {
        if (control_block)
        {
            control_block->add_persistent_reference();
        }
    }

    // ------------------------------------------------------------------------
    // operator= (Copy Assignment)
    // Purpose:
    //   Replaces the managed object with the one managed by other using
    //   copy-and-swap, safely releasing any previously held reference.
    // Parameters:
    //   other - Source persistent shared pointer to copy from.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------
    PersistentSharedPtr& operator=(const PersistentSharedPtr& other) noexcept
    {
        if (this != &other)
        {
            PersistentSharedPtr temp(other);
            swap(temp);
        }
        
        return *this;
    }

    // ------------------------------------------------------------------------
    // PersistentSharedPtr (Move Constructor)
    // Purpose:
    //   Transfers persistent ownership from other to *this without modifying
    //   atomic reference counts, leaving other empty.
    // Parameters:
    //   other - Rvalue reference to the source pointer to move from.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    PersistentSharedPtr(PersistentSharedPtr&& other) noexcept : control_block(other.control_block)
    {
        other.control_block = nullptr;
    }

    // ------------------------------------------------------------------------
    // operator= (Move Assignment)
    // Purpose:
    //   Releases any currently managed reference and transfers persistent
    //   ownership from other without extra atomic increments.
    // Parameters:
    //   other - Rvalue reference to the source pointer to move from.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------
    PersistentSharedPtr& operator=(PersistentSharedPtr&& other) noexcept
    {
        if (this != &other)
        {
            // Move-and-swap before releasing the previous control block so self-referential
            // moves (e.g., head = std::move(head->next)) never access a destroyed `other`.
            PersistentSharedPtr temp(std::move(other));
            swap(temp);
        }
        
        return *this;
    }

    // ------------------------------------------------------------------------
    // operator= (nullptr Assignment)
    // Purpose:
    //   Releases the currently managed persistent reference and resets *this
    //   to empty.
    // Parameters:
    //   nullptr - Null pointer literal.
    // Return Value:
    //   Reference to *this.
    // ------------------------------------------------------------------------
    PersistentSharedPtr& operator=(decltype(nullptr)) noexcept
    {
        reset();
        return *this;
    }

    // ------------------------------------------------------------------------
    // swap
    // Purpose:
    //   Exchanges the underlying control block pointers of *this and other
    //   without atomic reference count operations.
    // Parameters:
    //   other - Persistent shared pointer to swap state with.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    void swap(PersistentSharedPtr& other) noexcept
    {
        RundownControlBlock<T>* temp_block = control_block;
        control_block                      = other.control_block;
        other.control_block                = temp_block;
    }

    // ------------------------------------------------------------------------
    // reset
    // Purpose:
    //   Releases ownership of the managed object (if any) and sets *this to empty.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    void reset() noexcept
    {
        PersistentSharedPtr().swap(*this);
    }

    // ------------------------------------------------------------------------
    // get
    // Purpose:
    //   Returns a raw pointer to the managed payload object via compile-time
    //   offset arithmetic, or nullptr if empty.
    // Parameters:
    //   None.
    // Return Value:
    //   Pointer to the managed payload object T, or nullptr.
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE T* get() const noexcept 
    { 
        return control_block ? control_block->get_payload() : nullptr; 
    }
    
    // ------------------------------------------------------------------------
    // operator->
    // Purpose:
    //   Provides member access to the managed payload object T.
    // Parameters:
    //   None.
    // Return Value:
    //   Pointer to the managed payload object T.
    // ------------------------------------------------------------------------
    FORCE_INLINE T* operator->() const noexcept 
    { 
        return get(); 
    }
    
    // ------------------------------------------------------------------------
    // operator*
    // Purpose:
    //   Dereferences the managed payload object T. Caller must ensure the
    //   pointer is non-empty.
    // Parameters:
    //   None.
    // Return Value:
    //   Reference to the managed payload object T.
    // ------------------------------------------------------------------------
    FORCE_INLINE T& operator*() const noexcept 
    { 
        return *get(); 
    }
    
    // ------------------------------------------------------------------------
    // operator bool
    // Purpose:
    //   Checks whether *this manages a valid control block and payload object.
    // Parameters:
    //   None.
    // Return Value:
    //   true if *this owns a non-null control block; false otherwise.
    // ------------------------------------------------------------------------
    FORCE_INLINE explicit operator bool() const noexcept 
    { 
        return control_block != nullptr; 
    }

private:
    // Detaches and returns the underlying control block pointer without decrementing
    // its persistent reference count, transferring baseline ownership to the caller.
    [[nodiscard]] RundownControlBlock<T>* release() noexcept
    {
        RundownControlBlock<T>* block = control_block;
        control_block                 = nullptr;
        
        return block;
    }
};

// ============================================================================
// Scoped Rundown RAII Guard
// Constructor is private (friend of RundownSharedPtr) so client code cannot 
// instantiate or cache a rundown guard inside a heap structure or stream context.
// ============================================================================
class ScopedRundownGuard
{
    // The explicit thread-hashed rundown primitive successfully acquired by this reader.
    // Bit 0 encodes whether the slot was acquired in shared overflow mode (1) or
    // exclusive single-writer mode (0). Safe because RundownRef is cache-line aligned.
    uintptr_t encoded_rundown_ref;

    template <typename U> friend class RundownSharedPtr;
    template <typename U> friend class BatchedRundownGuard;

    // Initializes an unattached scoped rundown guard.
    explicit ScopedRundownGuard() noexcept : encoded_rundown_ref(0)
    {
    }

public:
    // ------------------------------------------------------------------------
    // attach
    // Purpose:
    //   Binds an acquired per-slot RundownRef and its acquisition mode
    //   (exclusive vs. shared) to this RAII guard for automatic release on
    //   scope exit.
    // Parameters:
    //   ref       - Pointer to the acquired cache-aligned RundownRef slot.
    //   is_shared - true if acquired via acquire_shared(); false if acquired
    //               via acquire_exclusive().
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE void attach(RundownRef* ref, bool is_shared = false) noexcept
    {
        encoded_rundown_ref = reinterpret_cast<uintptr_t>(ref) | (is_shared ? uintptr_t(1) : uintptr_t(0));
    }

    // ------------------------------------------------------------------------
    // ~ScopedRundownGuard (Destructor)
    // Purpose:
    //   Releases the attached RundownRef slot using either release_exclusive()
    //   or release_shared() according to the low tag bit.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE ~ScopedRundownGuard() noexcept
    {
        if (encoded_rundown_ref) [[likely]]
        {
            if ((encoded_rundown_ref & uintptr_t(1)) == 0) [[likely]]
            {
                reinterpret_cast<RundownRef*>(encoded_rundown_ref)->release_exclusive();
            }
            else
            {
                reinterpret_cast<RundownRef*>(encoded_rundown_ref & ~uintptr_t(1))->release_shared();
            }
        }
    }

    ScopedRundownGuard(const ScopedRundownGuard&) = delete;
    ScopedRundownGuard& operator=(const ScopedRundownGuard&) = delete;
    static void* operator new(size_t)     = delete;
    static void* operator new[](size_t)   = delete;
    static void  operator delete(void*)   = delete;
    static void  operator delete[](void*) = delete;
};

// ============================================================================
// Batched Rundown Guard
// Retains a transient execution slot exclusively for the guard's lifetime 
// allowing loop boundaries and burst tasks to rapidly execute thousands of reads
// natively without re-acquiring the GC Epoch lock continuously. 
// ============================================================================
template <typename T>
class BatchedRundownGuard
{
    // Scoped RAII guard holding the acquired epoch rundown slot for the lifetime of this batch guard.
    ScopedRundownGuard rundown_guard;

    // Pointer to the validated control block pinned by rundown_guard (or nullptr if empty).
    RundownControlBlock<T>* safe_block;

    friend class RundownSharedPtr<T>;

    // Binds the acquired non-null RundownRef slot and validated control block to this batched guard.
    FORCE_INLINE explicit BatchedRundownGuard(RundownRef*             slot_ref, 
                                              RundownControlBlock<T>* block,
                                              bool                    is_shared = false) noexcept : safe_block(block)
    {
        rundown_guard.attach(slot_ref, is_shared);
    }

public:
    // ------------------------------------------------------------------------
    // BatchedRundownGuard (Default Constructor)
    // Purpose:
    //   Allow empty initialization for failure paths or empty anchor checkouts.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    FORCE_INLINE BatchedRundownGuard() noexcept : safe_block(nullptr) 
    {
    }

    BatchedRundownGuard(const BatchedRundownGuard&) = delete;
    BatchedRundownGuard& operator=(const BatchedRundownGuard&) = delete;
    static void* operator new(size_t)     = delete;
    static void* operator new[](size_t)   = delete;
    static void  operator delete(void*)   = delete;
    static void  operator delete[](void*) = delete;

    // ------------------------------------------------------------------------
    // read
    // Purpose:
    //   Invokes the caller-supplied lambda with a raw T* pointer to the pinned
    //   payload (or nullptr if empty) without re-acquiring epoch rundown
    //   protection. Enforces at compile time that raw pointers or references
    //   cannot be returned from the lambda.
    // Parameters:
    //   func - Callable accepting (T* payload). Must not return a raw pointer
    //          or reference.
    // Return Value:
    //   Forwards the by-value return result of func.
    // ------------------------------------------------------------------------
    template <typename F>
    FORCE_INLINE decltype(auto) read(F&& func) const noexcept(std::is_nothrow_invocable_v<F, T*>)
    {
        using ReturnType = decltype(std::forward<F>(func)(static_cast<T*>(nullptr)));

        // Compile-time enforcement: Forbid smuggling raw pointers or references out of read().
        static_assert(!std::is_pointer_v<ReturnType> && !std::is_reference_v<ReturnType>,
                      "Anti-pattern blocked: Do not return raw pointers or references from read(). "
                      "Call load_persistent() to obtain a PersistentSharedPtr for caching.");

        return std::forward<F>(func)(safe_block ? safe_block->get_payload() : nullptr);
    }
};

// ============================================================================
// RundownSharedPtr
// The global anchor. Facilitates the Execute-Around Lambda Access pattern.
// Displaced pointers are pushed to the central global GC pipeline.
// ============================================================================
template <typename T>
class RundownSharedPtr
{
    // The globally visible, highly contended atomic baseline pointer governing read access.
    alignas(CACHE_LINE_SIZE) std::atomic<RundownControlBlock<T>*> active_control_block;

    // Loads the currently anchored control block pointer with the specified memory ordering.
    FORCE_INLINE RundownControlBlock<T>* get_atomic_control_block(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        return active_control_block.load(order);
    }

public:
    RundownSharedPtr(const RundownSharedPtr&) = delete;
    RundownSharedPtr& operator=(const RundownSharedPtr&) = delete;

    // ------------------------------------------------------------------------
    // RundownSharedPtr (Default Constructor)
    // Purpose:
    //   Initializes an empty anchor structure and ensures the global
    //   SharedPtrGarbageCollector singleton is constructed.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    RundownSharedPtr() noexcept : active_control_block(nullptr)
    {
        (void)SharedPtrGarbageCollector::get_instance();
    }

    // ------------------------------------------------------------------------
    // RundownSharedPtr (Persistent Checkout Constructor)
    // Purpose:
    //   Initializes the anchor and naturally acquires baseline ownership from
    //   the provided PersistentSharedPtr.
    // Parameters:
    //   desired - PersistentSharedPtr<T> whose ownership is transferred into
    //             the anchor.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    explicit RundownSharedPtr(PersistentSharedPtr<T> desired) noexcept : active_control_block(nullptr)
    {
        (void)SharedPtrGarbageCollector::get_instance();
        (void)store(std::move(desired));
    }

    // ------------------------------------------------------------------------
    // ~RundownSharedPtr (Destructor)
    // Purpose:
    //   Displaces the current object (if any) into the GC pipeline for safe
    //   epoch-based reclamation.
    // Parameters:
    //   None.
    // Return Value:
    //   None.
    // ------------------------------------------------------------------------
    ~RundownSharedPtr() noexcept
    {
        (void)store(PersistentSharedPtr<T>());
    }

    // ------------------------------------------------------------------------
    // acquire_batched
    // Purpose:
    //   Secures a structural guarantee within the current epoch timeline to allow 
    //   the calling thread to sequentially read the pointer infinitely without
    //   paying the atomic checkout overhead per iteration. Relies on RVO 
    //   (Guaranteed Copy Elision) to bypass the un-movable ScopedRundownGuard.
    // Parameters:
    //   None.
    // Return Value:
    //   A BatchedRundownGuard<T> pinning the active epoch slot and control block
    //   (or an empty guard if the anchor is currently null).
    // ------------------------------------------------------------------------
    [[nodiscard]] FORCE_INLINE BatchedRundownGuard<T> acquire_batched() const noexcept
    {
        if (!get_atomic_control_block(std::memory_order_relaxed)) [[unlikely]]
        {
            return BatchedRundownGuard<T>();
        }

        SharedPtrGarbageCollector& gc_instance = SharedPtrGarbageCollector::get_instance();
        const uint32_t             raw_slot    = gc_instance.get_thread_reader_slot_raw();
        const bool                 is_shared   = (raw_slot & SharedPtrGarbageCollector::SLOT_SHARED_FLAG) != 0;
        const uint32_t             slot        = raw_slot & SharedPtrGarbageCollector::SLOT_INDEX_MASK;

        // Acquire the optimistic epoch routing hint; the validation barrier will naturally
        // trigger the retry loop via the secondary seq_cst load if the cache was stale.
        int64_t epoch = gc_instance.get_active_epoch_relaxed();

        while (true)
        {
            CacheAwareRundownRef* ref      = gc_instance.get_epoch_rundown_ref(epoch);
            RundownRef*           slot_ref = ref->get_slot(slot);

            // Wait-free slot acquisition followed by seq_cst epoch verification prior to loading active_control_block.
            slot_ref->acquire(is_shared);

            const int64_t current_epoch = gc_instance.get_active_epoch();
            if (current_epoch != epoch) [[unlikely]]
            {
                slot_ref->release(is_shared);
                epoch = current_epoch;
                continue;
            }

            RundownControlBlock<T>* safe_block = get_atomic_control_block(std::memory_order_acquire);

            if (!safe_block) [[unlikely]]
            {
                slot_ref->release(is_shared);
                return BatchedRundownGuard<T>();
            }

            return BatchedRundownGuard<T>(slot_ref, safe_block, is_shared);
        }
    }

    // ------------------------------------------------------------------------
    // read
    // Purpose:
    //   The Execute-Around access method. Binds the rundown protection strictly 
    //   to the execution scope of the lambda. Eliminates stream caching bugs.
    //   Conditionally noexcept so that if 'func' throws an exception, the stack
    //   unwinds cleanly and BatchedRundownGuard releases the rundown slot.
    // Parameters:
    //   func - Callable accepting (T* payload). Must not return a raw pointer
    //          or reference.
    // Return Value:
    //   Forwards the by-value return result of func.
    // ------------------------------------------------------------------------
    template <typename F>
    FORCE_INLINE decltype(auto) read(F&& func) const noexcept(std::is_nothrow_invocable_v<F, T*>)
    {
        return acquire_batched().read(std::forward<F>(func));
    }

    // ------------------------------------------------------------------------
    // load_persistent
    // Purpose:
    //   Execute this path to securely establish a Persistent Stream Context 
    //   reference that explicitly bypasses the fast-path Rundown garbage collector.
    // Parameters:
    //   None.
    // Return Value:
    //   A PersistentSharedPtr<T> owning a persistent reference to the active
    //   object, or an empty PersistentSharedPtr<T> if the anchor is empty.
    // ------------------------------------------------------------------------
    [[nodiscard]] PersistentSharedPtr<T> load_persistent() const noexcept
    {
        // Delegate epoch slot acquisition and validation to acquire_batched().
        // RVO constructs the returned PersistentSharedPtr (incrementing persistent_ref_count)
        // before guard's destructor releases the rundown slot.
        BatchedRundownGuard<T> guard = acquire_batched();
        return PersistentSharedPtr<T>(guard.safe_block, false);
    }

    // ------------------------------------------------------------------------
    // store
    // Purpose:
    //   Replaces the active anchor pointer. Transfers the persistent reference 
    //   from 'desired' to establish the new baseline ownership and pushes any
    //   displaced block into the global GC pipeline.
    // Parameters:
    //   desired - New PersistentSharedPtr<T> to install (or an empty
    //             PersistentSharedPtr<T> to clear the anchor).
    // Return Value:
    //   Returns true if the store succeeded, or false if 'desired' was rejected
    //   because its control block is already anchored or pending GC reclamation.
    // ------------------------------------------------------------------------
    bool store(PersistentSharedPtr<T> desired) noexcept
    {
        RundownControlBlock<T>* new_control_block = desired.release();

        if (new_control_block)
        {
            // Fast path: if the caller holds the sole reference (persistent_ref_count == 1
            // with acquire ordering synchronizing with any prior owner's release) and
            // anchor_state is 0, set anchor_state = 1 without an extra LOCK CMPXCHG.
            // The subsequent active_control_block.exchange(acq_rel) publishes the store.
            if (new_control_block->persistent_ref_count.load(std::memory_order_acquire) == 1 &&
                new_control_block->anchor_state.load(std::memory_order_relaxed) == 0) [[likely]]
            {
                new_control_block->anchor_state.store(1, std::memory_order_relaxed);
            }
            else
            {
                // Shared reference path: transition anchor_state from Unanchored (0) -> Anchored (1).
                // Rejects self-stores, concurrent multi-anchor sharing, or re-storing
                // a block that is still awaiting rundown reclamation in the GC pipeline.
                int32_t expected = 0;
                if (!new_control_block->anchor_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel))
                {
                    new_control_block->release_persistent_reference();
                    return false;
                }
            }
        }
        else if (!get_atomic_control_block(std::memory_order_relaxed))
        {
            // Fast-path no-op: Avoid dirtying the cache line when clearing an already empty anchor.
            return true;
        }

        RundownControlBlock<T>* old_control_block = active_control_block.exchange(new_control_block, std::memory_order_acq_rel);

        if (old_control_block)
        {
            // Direct the displaced pointer to the central global GC queue for aging
            SharedPtrGarbageCollector::get_instance().push(old_control_block);
        }

        return true;
    }
};

// ============================================================================
// Single-Allocation NUMA-Aware Factories
// ============================================================================

// ----------------------------------------------------------------------------
// MakeRundownSharedNuma
// Purpose:
//   Creates a PersistentSharedPtr in a single NUMA-affinitized allocation block
//   on the specified physical NUMA node (or the calling processor's local NUMA
//   node when preferred_node == NumaAllocator::ANY_NODE_OK). Places the
//   RundownControlBlock at the front and embeds the payload object directly
//   adjacent to it at a cache-aligned offset, ensuring strong data locality,
//   avoiding heap fragmentation, and isolating the payload from control-block
//   atomic mutations.
// Parameters:
//   preferred_node - Preferred physical NUMA node ID (or NumaAllocator::ANY_NODE_OK).
//   args           - Constructor arguments perfectly forwarded to T.
// Return Value:
//   A PersistentSharedPtr<T> managing the newly constructed object, or an
//   empty PersistentSharedPtr<T> if allocation fails.
// ----------------------------------------------------------------------------
template <typename T, typename... Args>
[[nodiscard]] PersistentSharedPtr<T> MakeRundownSharedNuma(uint32_t preferred_node, Args&&... args)
{
    SharedPtrGarbageCollector& gc_instance = SharedPtrGarbageCollector::get_instance();

    constexpr size_t allocation_alignment  = RundownControlBlock<T>::get_allocation_alignment();
    constexpr size_t payload_offset        = RundownControlBlock<T>::get_payload_offset();
    constexpr size_t total_allocation_size = payload_offset + sizeof(T);

    NumaSlabHeader* slab        = nullptr;
    uint32_t        actual_node = 0;

    // Pull from the per-slot / per-NUMA-node aligned control-block cache
    // (falling back to a 64 KB NumaAllocator slab or std::nothrow ::operator new)
    void* mem = gc_instance.allocate_control_block(total_allocation_size,
                                                   allocation_alignment,
                                                   preferred_node,
                                                   slab,
                                                   actual_node);
    if (!mem)
    {
        return PersistentSharedPtr<T>();
    }

    // Cast payload address to void* for placement new to support const T assignments
    void* payload_mem = static_cast<char*>(mem) + payload_offset;

    // Guard placement construction of T so that if T's constructor throws an exception,
    // the raw aligned memory block is properly returned to its NUMA cache/slab before propagating.
    try
    {
        ::new (payload_mem) T(std::forward<Args>(args)...);
    }
    catch (...)
    {
        gc_instance.deallocate_control_block(mem,
                                             total_allocation_size,
                                             allocation_alignment,
                                             slab,
                                             actual_node);
        throw;
    }

    RundownControlBlock<T>* control_block = ::new (mem) RundownControlBlock<T>(slab, actual_node);

    // The factory internally adopts the initial persistent reference (persistent_ref_count = 1) 
    // generated securely during the RundownControlBlock's constructor phase.
    return PersistentSharedPtr<T>(control_block, true);
}

// ----------------------------------------------------------------------------
// MakeRundownShared
// Purpose:
//   Creates a PersistentSharedPtr in a single allocation block on the calling
//   processor's local NUMA node (NumaAllocator::ANY_NODE_OK). Places the
//   RundownControlBlock at the front and embeds the payload object directly  
//   adjacent to it, ensuring strong data locality and avoiding fragmentation. 
// Parameters:
//   args - Constructor arguments perfectly forwarded to T.
// Return Value:
//   A PersistentSharedPtr<T> managing the newly constructed object, or an
//   empty PersistentSharedPtr<T> if allocation fails.
// ----------------------------------------------------------------------------
template <typename T, typename... Args>
[[nodiscard]] PersistentSharedPtr<T> MakeRundownShared(Args&&... args)
{
    return MakeRundownSharedNuma<T>(NumaAllocator::ANY_NODE_OK, std::forward<Args>(args)...);
}
