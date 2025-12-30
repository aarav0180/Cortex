#include "kv_cache.h"
#include "llama.h"
#ifdef __ANDROID__
#include <android/log.h>
#else
#include <iostream>
#endif

#define LOG_TAG "CortexKVCache"
#ifdef __ANDROID__
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) printf("INFO: " __VA_ARGS__); printf("\n")
#define LOGD(...) printf("DEBUG: " __VA_ARGS__); printf("\n")
#define LOGW(...) printf("WARN: " __VA_ARGS__); printf("\n")
#define LOGE(...) printf("ERROR: " __VA_ARGS__); printf("\n")
#endif

namespace cortex {

KVCache::KVCache() {
    LOGD("KVCache created");
}

KVCache::~KVCache() {
    shutdown();
    LOGD("KVCache destroyed");
}

bool KVCache::initialize(llama_context* ctx, const KVCacheConfig& config) {
    if (ctx == nullptr) {
        LOGE("Cannot initialize KV cache: context is null");
        return false;
    }
    
    ctx_ = ctx;
    config_ = config;
    initialized_ = true;
    
    LOGI("KV cache initialized with %d context size", config.n_ctx);
    return true;
}

void KVCache::shutdown() {
    if (initialized_) {
        clear();
        ctx_ = nullptr;
        initialized_ = false;
        LOGI("KV cache shutdown");
    }
}

void KVCache::clear() {
    if (!initialized_ || ctx_ == nullptr) return;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem != nullptr) {
        llama_memory_clear(mem, true);
    }
    LOGD("KV cache cleared");
}

bool KVCache::removeTokens(int start_pos, int end_pos) {
    if (!initialized_ || ctx_ == nullptr) return false;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return false;
    
    // Use sequence remove for all sequences (-1)
    bool success = llama_memory_seq_rm(mem, -1, start_pos, end_pos);
    
    if (success) {
        LOGD("Removed tokens from pos %d to %d", start_pos, end_pos);
    } else {
        LOGW("Failed to remove tokens from pos %d to %d", start_pos, end_pos);
    }
    
    return success;
}

bool KVCache::shiftTokens(int start_pos, int delta) {
    if (!initialized_ || ctx_ == nullptr) return false;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return false;
    
    // Shift all sequences (-1)
    llama_memory_seq_add(mem, -1, start_pos, -1, delta);
    LOGD("Shifted tokens from pos %d by delta %d", start_pos, delta);
    
    return true;
}

bool KVCache::sequenceCopy(int src_seq, int dst_seq, int start_pos, int end_pos) {
    if (!initialized_ || ctx_ == nullptr) return false;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return false;
    
    llama_memory_seq_cp(mem, src_seq, dst_seq, start_pos, end_pos);
    LOGD("Copied sequence %d to %d (pos %d-%d)", src_seq, dst_seq, start_pos, end_pos);
    
    return true;
}

bool KVCache::sequenceRemove(int seq_id, int start_pos, int end_pos) {
    if (!initialized_ || ctx_ == nullptr) return false;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return false;
    
    bool success = llama_memory_seq_rm(mem, seq_id, start_pos, end_pos);
    
    if (success) {
        LOGD("Removed sequence %d (pos %d-%d)", seq_id, start_pos, end_pos);
    }
    
    return success;
}

void KVCache::sequenceKeep(int seq_id) {
    if (!initialized_ || ctx_ == nullptr) return;
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return;
    
    llama_memory_seq_keep(mem, seq_id);
    LOGD("Keeping only sequence %d", seq_id);
}

void KVCache::defragment() {
    if (!initialized_ || ctx_ == nullptr) return;
    
    // Note: llama_memory doesn't have a direct defrag call in new API
    // Defragmentation is handled internally or via clearing
    LOGI("KV cache defragment requested (handled internally)");
}

bool KVCache::needsDefragmentation() const {
    if (!initialized_) return false;
    
    float usage = getUsageRatio();
    return usage > config_.defrag_threshold;
}

KVCacheStats KVCache::getStats() const {
    KVCacheStats stats;
    
    if (!initialized_ || ctx_ == nullptr) {
        return stats;
    }
    
    stats.used_cells = getUsedCells();
    stats.total_cells = getTotalCells();
    stats.usage_ratio = getUsageRatio();
    
    // Rough memory estimate (actual depends on model architecture)
    // KV cache typically uses 2 * n_layer * n_embd * sizeof(float) per token
    // We can't easily get these values without model reference, so estimate
    stats.memory_bytes = stats.used_cells * 4096;  // Rough estimate
    
    return stats;
}

size_t KVCache::getUsedCells() const {
    if (!initialized_ || ctx_ == nullptr) return 0;
    
    // Use llama_memory_seq_pos_max to estimate used cells
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem == nullptr) return 0;
    
    llama_pos max_pos = llama_memory_seq_pos_max(mem, 0);
    return max_pos > 0 ? static_cast<size_t>(max_pos) : 0;
}

size_t KVCache::getTotalCells() const {
    if (!initialized_ || ctx_ == nullptr) return 0;
    return llama_n_ctx(ctx_);
}

float KVCache::getUsageRatio() const {
    size_t total = getTotalCells();
    if (total == 0) return 0.0f;
    return static_cast<float>(getUsedCells()) / static_cast<float>(total);
}

size_t KVCache::estimateMemory(int n_ctx, int n_embd, int n_layer, int n_head) {
    // KV cache stores key and value tensors for each layer
    // Each has size: n_ctx * n_embd * sizeof(float)
    // Total: 2 * n_layer * n_ctx * n_embd * sizeof(float)
    
    size_t kv_size = 2 * static_cast<size_t>(n_layer) * 
                     static_cast<size_t>(n_ctx) * 
                     static_cast<size_t>(n_embd) * 
                     sizeof(float);
    
    // Add some overhead for metadata and alignment
    size_t overhead = kv_size / 10;  // 10% overhead
    
    return kv_size + overhead;
}

} // namespace cortex
