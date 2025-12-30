#pragma once

#include <cstddef>
#include <string>
#include <functional>

namespace cortex {

// Memory pressure levels
enum class MemoryPressure {
    Low,
    Medium,
    High,
    Critical
};

// Memory statistics
struct MemoryStats {
    size_t total_memory = 0;
    size_t available_memory = 0;
    size_t used_memory = 0;
    size_t model_memory = 0;
    size_t context_memory = 0;
    MemoryPressure pressure = MemoryPressure::Low;
};

// Memory callback for pressure notifications
using MemoryPressureCallback = std::function<void(MemoryPressure level)>;

class MemoryManager {
public:
    static MemoryManager& getInstance();
    
    // Delete copy/move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    // Memory queries
    MemoryStats getMemoryStats() const;
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;
    MemoryPressure getMemoryPressure() const;
    
    // Memory management
    bool canAllocate(size_t bytes) const;
    size_t getRecommendedContextSize() const;
    size_t getMaxModelSize() const;
    
    // Memory pressure handling
    void setMemoryPressureCallback(MemoryPressureCallback callback);
    void checkMemoryPressure();
    void requestMemoryCleanup();
    
    // Model memory tracking
    void registerModelMemory(size_t bytes);
    void unregisterModelMemory(size_t bytes);
    void registerContextMemory(size_t bytes);
    void unregisterContextMemory(size_t bytes);
    
private:
    MemoryManager();
    ~MemoryManager() = default;
    
    size_t model_memory_ = 0;
    size_t context_memory_ = 0;
    MemoryPressureCallback pressure_callback_;
    
    size_t readMemInfo(const char* field) const;
};

} // namespace cortex
