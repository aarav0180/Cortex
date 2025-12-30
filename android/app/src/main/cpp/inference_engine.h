#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>

// llama.cpp headers
#include "llama.h"
#include "ggml.h"

// JNI callback for push-based token delivery
#ifdef __ANDROID__
#include <jni.h>
#endif

namespace cortex {

// Configuration for the inference engine
struct InferenceConfig {
    int context_length = 4096;
    int batch_size = 512;
    int max_tokens = 2048;
    int threads = 4;
    bool use_mmap = true;
    bool use_mlock = false;
    
    // Sampling parameters
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    
    // GPU offload (for future use)
    int gpu_layers = 0;
};

// Statistics about generation
struct GenerationStats {
    int64_t total_tokens = 0;
    int64_t prompt_tokens = 0;
    int64_t generated_tokens = 0;
    double prompt_eval_time_ms = 0;
    double eval_time_ms = 0;
    double tokens_per_second = 0;
};

// Token callback for streaming
using TokenCallback = std::function<bool(const std::string& token, bool is_final)>;

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    // Model management
    bool loadModel(const std::string& model_path, const InferenceConfig& config);
    void unloadModel();
    bool isModelLoaded() const;
    std::string getModelInfo() const;
    
    // Inference
    bool startInference(const std::string& prompt, const InferenceConfig& config);
    bool startInferenceIncremental(const std::string& prompt, const InferenceConfig& config);  // KV cache reuse
    bool startInferenceThreaded(const std::string& prompt, const InferenceConfig& config);  // Multi-threaded generation
    std::string getNextToken();
    std::vector<std::string> getNextTokens(int count = 4);  // Batch token decoding
    bool isGenerating() const;
    void stopGeneration();
    
    // Threaded generation control
    void setTokenCallback(std::function<void(const std::string&)> callback);
    void clearTokenCallback();
    std::string popTokenFromQueue();  // Non-blocking pop for polling mode
    
    // Context management
    void shiftContext(int keep_tokens = 64);  // Context shifting for infinite generation
    void clearCache();  // Clear KV cache for new conversation
    int getCachedTokenCount() const { return n_past_; }  // How many tokens are cached
    
    // Streaming inference with callback
    bool generateWithCallback(const std::string& prompt, 
                              const InferenceConfig& config,
                              TokenCallback callback);
    
    // Statistics
    GenerationStats getStats() const;
    void resetStats();
    
    // Memory info
    size_t getModelMemoryUsage() const;
    size_t getContextMemoryUsage() const;
    
private:
    // llama.cpp structures
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    
    // State
    std::atomic<bool> is_generating_{false};
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex mutex_;
    
    // Threaded generation
    std::thread generation_thread_;
    std::thread processor_thread_;
    std::queue<llama_token> token_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> generation_complete_{false};
    std::function<void(const std::string&)> token_callback_;
    std::mutex callback_mutex_;
    
    // Output buffer for batched delivery
    std::string output_buffer_;
    std::mutex buffer_mutex_;
    static constexpr int BATCH_SIZE = 4;  // Tokens to batch before callback
    
    // Token state for getNextToken()
    std::vector<llama_token> tokens_;
    int current_pos_ = 0;
    int n_past_ = 0;
    
    // Stats
    GenerationStats stats_;
    int64_t eval_start_time_ = 0;
    
    // Configuration
    InferenceConfig current_config_;
    std::string model_path_;
    
    // Internal methods
    bool tokenizePrompt(const std::string& prompt, std::vector<llama_token>& tokens);
    bool evaluateTokens(const std::vector<llama_token>& tokens, int n_past, int n_tokens);
    llama_token sampleNextToken();
    std::string tokenToString(llama_token token);
    void initSampler(const InferenceConfig& config);
    void freeSampler();
    int64_t getCurrentTimeMs() const;
    
    // Thread worker functions
    void generationThreadFunc();
    void processorThreadFunc();
    void stopThreads();
};

} // namespace cortex
