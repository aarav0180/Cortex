#include "memory_manager.h"
#include <fstream>
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "CortexMemory"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <iostream>
#define LOGI(...) printf(__VA_ARGS__); printf("\n")
#define LOGW(...) printf("WARN: "); printf(__VA_ARGS__); printf("\n")
#define LOGE(...) printf("ERROR: "); printf(__VA_ARGS__); printf("\n")
#endif

namespace cortex {

// Memory thresholds (in bytes)
constexpr size_t MEMORY_LOW_THRESHOLD = 512 * 1024 * 1024;      // 512 MB
constexpr size_t MEMORY_MEDIUM_THRESHOLD = 256 * 1024 * 1024;   // 256 MB
constexpr size_t MEMORY_HIGH_THRESHOLD = 128 * 1024 * 1024;     // 128 MB
constexpr size_t MEMORY_CRITICAL_THRESHOLD = 64 * 1024 * 1024;  // 64 MB

// Safety margins
constexpr size_t ALLOCATION_SAFETY_MARGIN = 100 * 1024 * 1024;  // 100 MB
constexpr double MAX_MEMORY_USAGE_RATIO = 0.90;  // Use at most 90% of available (mobile can handle more)

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() {
    LOGI("MemoryManager initialized");
    LOGI("Total memory: %zu MB", getTotalMemory() / (1024 * 1024));
    LOGI("Available memory: %zu MB", getAvailableMemory() / (1024 * 1024));
}

size_t MemoryManager::readMemInfo(const char* field) const {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        LOGE("Failed to open /proc/meminfo");
        return 0;
    }
    
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find(field) == 0) {
            std::istringstream iss(line);
            std::string name;
            size_t value;
            std::string unit;
            iss >> name >> value >> unit;
            // Value is in kB, convert to bytes
            return value * 1024;
        }
    }
    
    return 0;
}

size_t MemoryManager::getTotalMemory() const {
    return readMemInfo("MemTotal:");
}

size_t MemoryManager::getAvailableMemory() const {
    // Try MemAvailable first (more accurate on newer kernels)
    size_t available = readMemInfo("MemAvailable:");
    if (available > 0) {
        return available;
    }
    
    // Fallback to MemFree + Buffers + Cached
    size_t free = readMemInfo("MemFree:");
    size_t buffers = readMemInfo("Buffers:");
    size_t cached = readMemInfo("Cached:");
    
    return free + buffers + cached;
}

MemoryStats MemoryManager::getMemoryStats() const {
    MemoryStats stats;
    stats.total_memory = getTotalMemory();
    stats.available_memory = getAvailableMemory();
    stats.used_memory = stats.total_memory - stats.available_memory;
    stats.model_memory = model_memory_;
    stats.context_memory = context_memory_;
    stats.pressure = getMemoryPressure();
    return stats;
}

MemoryPressure MemoryManager::getMemoryPressure() const {
    size_t available = getAvailableMemory();
    
    if (available < MEMORY_CRITICAL_THRESHOLD) {
        return MemoryPressure::Critical;
    } else if (available < MEMORY_HIGH_THRESHOLD) {
        return MemoryPressure::High;
    } else if (available < MEMORY_MEDIUM_THRESHOLD) {
        return MemoryPressure::Medium;
    }
    
    return MemoryPressure::Low;
}

bool MemoryManager::canAllocate(size_t bytes) const {
    size_t available = getAvailableMemory();
    size_t needed = bytes + ALLOCATION_SAFETY_MARGIN;
    
    // Primary check: do we have enough available memory?
    if (available < needed) {
        LOGW("Cannot allocate %zu bytes: only %zu MB available", bytes, available / (1024 * 1024));
        return false;
    }
    
    // Secondary check: ensure we leave at least 500MB free after allocation
    constexpr size_t MIN_FREE_AFTER_ALLOC = 500 * 1024 * 1024;  // 500 MB
    if (available - bytes < MIN_FREE_AFTER_ALLOC) {
        LOGW("Allocation would leave less than 500MB free");
        return false;
    }
    
    LOGI("Memory check passed: %zu MB needed, %zu MB available", 
         bytes / (1024 * 1024), available / (1024 * 1024));
    return true;
}

size_t MemoryManager::getRecommendedContextSize() const {
    size_t available = getAvailableMemory();
    
    // Reserve memory for model and system
    size_t usable = available > ALLOCATION_SAFETY_MARGIN ? 
                    available - ALLOCATION_SAFETY_MARGIN : 0;
    
    // Rough estimate: each context token uses ~4KB for KV cache
    // This varies significantly based on model architecture
    constexpr size_t BYTES_PER_TOKEN = 4 * 1024;
    
    size_t max_tokens = usable / BYTES_PER_TOKEN;
    
    // Clamp to reasonable range
    max_tokens = std::max(max_tokens, static_cast<size_t>(512));
    max_tokens = std::min(max_tokens, static_cast<size_t>(32768));
    
    // Round to power of 2 for efficiency
    size_t result = 512;
    while (result < max_tokens && result < 32768) {
        result *= 2;
    }
    if (result > max_tokens) {
        result /= 2;
    }
    
    LOGI("Recommended context size: %zu tokens", result);
    return result;
}

size_t MemoryManager::getMaxModelSize() const {
    size_t available = getAvailableMemory();
    
    // Use at most 60% of available memory for model
    // (need to leave room for context and runtime allocations)
    size_t max_size = static_cast<size_t>(available * 0.6);
    
    LOGI("Max model size: %zu MB", max_size / (1024 * 1024));
    return max_size;
}

void MemoryManager::setMemoryPressureCallback(MemoryPressureCallback callback) {
    pressure_callback_ = std::move(callback);
}

void MemoryManager::checkMemoryPressure() {
    MemoryPressure pressure = getMemoryPressure();
    
    if (pressure != MemoryPressure::Low && pressure_callback_) {
        LOGW("Memory pressure detected: %d", static_cast<int>(pressure));
        pressure_callback_(pressure);
    }
}

void MemoryManager::requestMemoryCleanup() {
    LOGI("Memory cleanup requested");
    
    // Memory cleanup - let garbage collection handle it
    // Direct memory manipulation APIs vary by Android version
    
    LOGI("After cleanup - Available: %zu MB", getAvailableMemory() / (1024 * 1024));
}

void MemoryManager::registerModelMemory(size_t bytes) {
    model_memory_ += bytes;
    LOGI("Model memory registered: %zu MB (total: %zu MB)", 
         bytes / (1024 * 1024), model_memory_ / (1024 * 1024));
    checkMemoryPressure();
}

void MemoryManager::unregisterModelMemory(size_t bytes) {
    if (bytes > model_memory_) {
        model_memory_ = 0;
    } else {
        model_memory_ -= bytes;
    }
    LOGI("Model memory unregistered: %zu MB (total: %zu MB)", 
         bytes / (1024 * 1024), model_memory_ / (1024 * 1024));
}

void MemoryManager::registerContextMemory(size_t bytes) {
    context_memory_ += bytes;
    LOGI("Context memory registered: %zu MB (total: %zu MB)", 
         bytes / (1024 * 1024), context_memory_ / (1024 * 1024));
    checkMemoryPressure();
}

void MemoryManager::unregisterContextMemory(size_t bytes) {
    if (bytes > context_memory_) {
        context_memory_ = 0;
    } else {
        context_memory_ -= bytes;
    }
    LOGI("Context memory unregistered: %zu MB (total: %zu MB)", 
         bytes / (1024 * 1024), context_memory_ / (1024 * 1024));
}

} // namespace cortex
