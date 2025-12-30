#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Forward declaration
struct llama_context;

namespace cortex {

// KV cache statistics
struct KVCacheStats {
    size_t total_cells = 0;
    size_t used_cells = 0;
    size_t max_seq_len = 0;
    size_t memory_bytes = 0;
    float usage_ratio = 0.0f;
};

// KV cache configuration
struct KVCacheConfig {
    int n_ctx = 4096;           // Context size
    int n_batch = 512;          // Batch size
    bool use_cache = true;      // Enable KV cache
    float defrag_threshold = 0.8f;  // When to defragment
};

class KVCache {
public:
    KVCache();
    ~KVCache();
    
    // Non-copyable
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    // Initialization
    bool initialize(llama_context* ctx, const KVCacheConfig& config);
    void shutdown();
    
    // Cache operations
    void clear();
    bool removeTokens(int start_pos, int end_pos);
    bool shiftTokens(int start_pos, int delta);
    
    // Sequence management
    bool sequenceCopy(int src_seq, int dst_seq, int start_pos, int end_pos);
    bool sequenceRemove(int seq_id, int start_pos, int end_pos);
    void sequenceKeep(int seq_id);
    
    // Defragmentation
    void defragment();
    bool needsDefragmentation() const;
    
    // Statistics
    KVCacheStats getStats() const;
    size_t getUsedCells() const;
    size_t getTotalCells() const;
    float getUsageRatio() const;
    
    // Memory estimation
    static size_t estimateMemory(int n_ctx, int n_embd, int n_layer, int n_head);
    
private:
    llama_context* ctx_ = nullptr;
    KVCacheConfig config_;
    bool initialized_ = false;
};

} // namespace cortex
