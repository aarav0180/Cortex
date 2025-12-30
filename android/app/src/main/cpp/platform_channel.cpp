#include "inference_engine.h"
#include "memory_manager.h"
#include <thread>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "CortexChannel"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <iostream>
#define LOGI(...) printf(__VA_ARGS__); printf("\n")
#define LOGE(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#endif

namespace cortex {

// Global inference engine instance
static std::unique_ptr<InferenceEngine> g_engine;

// Get or create the inference engine
InferenceEngine* getEngine() {
    if (!g_engine) {
        g_engine = std::make_unique<InferenceEngine>();
    }
    return g_engine.get();
}

// create optimal config for mobile
InferenceConfig createMobileConfig() {
    InferenceConfig config;
    
    MemoryManager& memMgr = MemoryManager::getInstance();
    size_t availableMemory = memMgr.getAvailableMemory();
    
    // use n_cores - 1 to leave one for os/ui
    int hardware_threads = std::thread::hardware_concurrency();
    config.threads = std::max(2, hardware_threads - 1);
    
    config.context_length = 256;
    config.batch_size = 32;
    config.max_tokens = 256;
    
    config.use_mmap = true;
    config.use_mlock = false;
    config.gpu_layers = 0;
    
    config.temperature = 0.7f;
    config.top_p = 0.9f;
    config.top_k = 40;
    config.repeat_penalty = 1.1f;
    config.repeat_last_n = 64;
    
    LOGI("config: ctx=%d batch=%d threads=%d flash_attn=on kv=f16", 
         config.context_length, config.batch_size, config.threads);
    
    return config;
}

// Platform channel functions (called from JNI)

bool loadModel(const std::string& modelPath) {
    LOGI("loading: %s", modelPath.c_str());
    
    InferenceEngine* engine = getEngine();
    InferenceConfig config = createMobileConfig();
    
    // Check if we have enough memory based on available RAM
    MemoryManager& memMgr = MemoryManager::getInstance();
    // Estimate memory needed: small Q4 models need ~700MB, larger ones need more
    // For now, require at least 600MB available (will be refined once model is loaded)
    size_t estimated_mem = 600 * 1024 * 1024;  // 600 MB minimum
    if (!memMgr.canAllocate(estimated_mem)) {
        LOGE("Not enough memory to load model (need ~600MB)");
        return false;
    }
    
    bool success = engine->loadModel(modelPath, config);
    
    if (success) {
        // Register memory usage
        memMgr.registerModelMemory(engine->getModelMemoryUsage());
        memMgr.registerContextMemory(engine->getContextMemoryUsage());
    }
    
    return success;
}

void unloadModel() {
    if (g_engine) {
        MemoryManager& memMgr = MemoryManager::getInstance();
        memMgr.unregisterModelMemory(g_engine->getModelMemoryUsage());
        memMgr.unregisterContextMemory(g_engine->getContextMemoryUsage());
        
        g_engine->unloadModel();
    }
}

bool isModelLoaded() {
    return g_engine && g_engine->isModelLoaded();
}

std::string getModelInfo() {
    if (!g_engine) return "No model loaded";
    return g_engine->getModelInfo();
}

bool startGeneration(const std::string& prompt, float temperature, float top_p, 
                     int top_k, int max_tokens) {
    if (!g_engine || !g_engine->isModelLoaded()) {
        LOGE("Cannot generate: model not loaded");
        return false;
    }
    
    InferenceConfig config = createMobileConfig();
    
    // Override with user settings
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;
    config.max_tokens = max_tokens;
    
    LOGI("generation: temp=%.2f top_p=%.2f top_k=%d",
         temperature, top_p, top_k);
    
    return g_engine->startInference(prompt, config);
}

bool startGenerationIncremental(const std::string& prompt, float temperature, float top_p, 
                                int top_k, int max_tokens) {
    if (!g_engine || !g_engine->isModelLoaded()) {
        LOGE("model not loaded");
        return false;
    }
    
    InferenceConfig config = createMobileConfig();
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;
    config.max_tokens = max_tokens;
    
    return g_engine->startInferenceIncremental(prompt, config);
}

bool startGenerationTurbo(const std::string& prompt) {
    if (!g_engine || !g_engine->isModelLoaded()) {
        LOGE("model not loaded");
        return false;
    }
    
    InferenceConfig config = createMobileConfig();
    config.temperature = 0.7f;
    config.top_k = 40;
    config.top_p = 0.9f;
    config.repeat_penalty = 1.1f;
    config.repeat_last_n = 64;
    config.max_tokens = 256;
    
    return g_engine->startInferenceIncremental(prompt, config);
}

std::string getNextTokensBatch(int count) {
    // OPTIMIZATION: Return concatenated string instead of vector
    // This reduces JNI overhead by avoiding jobjectArray creation
    if (!g_engine) return "";
    
    std::vector<std::string> tokens = g_engine->getNextTokens(count);
    std::string result;
    result.reserve(count * 8);  // Pre-allocate
    for (const auto& token : tokens) {
        result += token;
    }
    return result;
}

void clearCache() {
    if (g_engine) {
        g_engine->clearCache();
    }
}

int getCachedTokenCount() {
    if (!g_engine) return 0;
    return g_engine->getCachedTokenCount();
}

std::string getNextToken() {
    if (!g_engine) return "";
    return g_engine->getNextToken();
}

std::vector<std::string> getNextTokens(int count) {
    if (!g_engine) return {};
    return g_engine->getNextTokens(count);
}

bool isGenerating() {
    return g_engine && g_engine->isGenerating();
}

void stopGeneration() {
    if (g_engine) {
        g_engine->stopGeneration();
    }
}

std::string getStats() {
    if (!g_engine) return "{}";
    
    GenerationStats stats = g_engine->getStats();
    
    // Return as simple JSON
    char buffer[512];
    snprintf(buffer, sizeof(buffer),
        "{\"prompt_tokens\":%lld,\"generated_tokens\":%lld,"
        "\"prompt_time_ms\":%.2f,\"eval_time_ms\":%.2f,"
        "\"tokens_per_second\":%.2f}",
        static_cast<long long>(stats.prompt_tokens),
        static_cast<long long>(stats.generated_tokens),
        stats.prompt_eval_time_ms,
        stats.eval_time_ms,
        stats.tokens_per_second);
    
    return std::string(buffer);
}

std::string getMemoryInfo() {
    MemoryManager& memMgr = MemoryManager::getInstance();
    MemoryStats stats = memMgr.getMemoryStats();
    
    const char* pressureStr;
    switch (stats.pressure) {
        case MemoryPressure::Low: pressureStr = "low"; break;
        case MemoryPressure::Medium: pressureStr = "medium"; break;
        case MemoryPressure::High: pressureStr = "high"; break;
        case MemoryPressure::Critical: pressureStr = "critical"; break;
        default: pressureStr = "unknown";
    }
    
    char buffer[512];
    snprintf(buffer, sizeof(buffer),
        "{\"total_mb\":%zu,\"available_mb\":%zu,\"used_mb\":%zu,"
        "\"model_mb\":%zu,\"context_mb\":%zu,\"pressure\":\"%s\"}",
        stats.total_memory / (1024 * 1024),
        stats.available_memory / (1024 * 1024),
        stats.used_memory / (1024 * 1024),
        stats.model_memory / (1024 * 1024),
        stats.context_memory / (1024 * 1024),
        pressureStr);
    
    return std::string(buffer);
}

long getMemoryUsage() {
    MemoryManager& memMgr = MemoryManager::getInstance();
    MemoryStats stats = memMgr.getMemoryStats();
    return static_cast<long>(stats.model_memory + stats.context_memory);
}

void resetStats() {
    if (g_engine) {
        g_engine->resetStats();
    }
}

bool startGenerationThreaded(const std::string& prompt, float temperature, float top_p,
                             int top_k, int max_tokens) {
    if (!g_engine || !g_engine->isModelLoaded()) {
        LOGE("model not loaded");
        return false;
    }
    
    InferenceConfig config = createMobileConfig();
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;
    config.max_tokens = max_tokens;
    
    return g_engine->startInferenceThreaded(prompt, config);
}

std::string getBufferedTokens() {
    if (!g_engine) return "";
    return g_engine->popTokenFromQueue();
}

void setTokenCallback(std::function<void(const std::string&)> callback) {
    if (g_engine) {
        g_engine->setTokenCallback(callback);
    }
}

void clearTokenCallback() {
    if (g_engine) {
        g_engine->clearTokenCallback();
    }
}

} // namespace cortex
