#include "inference_engine.h"
#include <chrono>
#include <thread>

#ifdef __ANDROID__
    #include <android/log.h>
    #include <sched.h>
    #include <pthread.h>
    #define LOG_TAG "CortexInference"
    #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
    #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
    #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
    #define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#else
    #include <iostream>
    #define LOG_TAG "CortexInference"
    #define LOGI(...) printf("[INFO] " __VA_ARGS__); printf("\n")
    #define LOGE(...) printf("[ERROR] " __VA_ARGS__); printf("\n")
    #define LOGD(...) printf("[DEBUG] " __VA_ARGS__); printf("\n")
    #define LOGW(...) printf("[WARN] " __VA_ARGS__); printf("\n")
#endif

namespace cortex {

InferenceEngine::InferenceEngine() {
    llama_backend_init();
}

InferenceEngine::~InferenceEngine() {
    unloadModel();
    llama_backend_free();
}

bool InferenceEngine::loadModel(const std::string& model_path, const InferenceConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Unload any existing model
    if (model_ != nullptr) {
        unloadModel();
    }
    
    LOGI("loading: %s", model_path.c_str());
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.gpu_layers;
    model_params.use_mmap = config.use_mmap;
    model_params.use_mlock = config.use_mlock;
    
    // Load the model
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model_ == nullptr) {
        LOGE("failed to load: %s", model_path.c_str());
        return false;
    }
    
    // context params with optimizations
    llama_context_params ctx_params = llama_context_default_params();
    
    ctx_params.n_ctx = config.context_length;
    ctx_params.n_batch = config.batch_size;
    
    int n_cores = std::thread::hardware_concurrency();
    ctx_params.n_threads = std::max(1, n_cores - 1);
    ctx_params.n_threads_batch = n_cores;
    
    // flash attention + f16 kv cache
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;
    ctx_params.n_ubatch = 32;
    ctx_params.embeddings = false;
    ctx_params.no_perf = true;
    
    // create context
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        LOGE("failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }
    
    // Initialize sampler
    initSampler(config);
    
    // Store config and path
    current_config_ = config;
    model_path_ = model_path;
    
    return true;
}

void InferenceEngine::unloadModel() {
    stop_requested_ = true;
    
    // Wait for generation to stop
    while (is_generating_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    freeSampler();
    
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    
    tokens_.clear();
    current_pos_ = 0;
    n_past_ = 0;
    stop_requested_ = false;
}

bool InferenceEngine::isModelLoaded() const {
    return model_ != nullptr && ctx_ != nullptr;
}

std::string InferenceEngine::getModelInfo() const {
    if (!isModelLoaded()) {
        return "No model loaded";
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    std::string info;
    info += "Model: " + model_path_ + "\n";
    info += "Context size: " + std::to_string(llama_n_ctx(ctx_)) + "\n";
    info += "Vocab size: " + std::to_string(llama_vocab_n_tokens(vocab)) + "\n";
    info += "Embedding size: " + std::to_string(llama_model_n_embd(model_)) + "\n";
    
    return info;
}

void InferenceEngine::initSampler(const InferenceConfig& config) {
    freeSampler();
    
    // Create sampler chain
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(chain_params);
    
    // Add samplers to the chain
    // Repetition penalty (new API: only 4 args)
    llama_sampler_chain_add(sampler_, 
        llama_sampler_init_penalties(
            config.repeat_last_n,       // penalty_last_n
            config.repeat_penalty,      // penalty_repeat
            0.0f,                        // penalty_freq
            0.0f                         // penalty_present
        )
    );
    
    // Top-K sampling
    llama_sampler_chain_add(sampler_, 
        llama_sampler_init_top_k(config.top_k)
    );
    
    // Top-P (nucleus) sampling
    llama_sampler_chain_add(sampler_, 
        llama_sampler_init_top_p(config.top_p, 1)
    );
    
    // Temperature
    llama_sampler_chain_add(sampler_, 
        llama_sampler_init_temp(config.temperature)
    );
    
    // Distribution sampler (final selection)
    llama_sampler_chain_add(sampler_, 
        llama_sampler_init_dist(LLAMA_DEFAULT_SEED)
    );
    
    LOGD("Sampler initialized: temp=%.2f, top_k=%d, top_p=%.2f, repeat_penalty=%.2f",
         config.temperature, config.top_k, config.top_p, config.repeat_penalty);
}

void InferenceEngine::freeSampler() {
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
}

bool InferenceEngine::startInference(const std::string& prompt, const InferenceConfig& config) {
    if (!isModelLoaded()) {
        LOGE("Cannot start inference: model not loaded");
        return false;
    }
    
    // If already generating, stop it first (allows recovery from stuck state)
    if (is_generating_) {
        LOGW("Previous generation still marked as active, forcing stop");
        stop_requested_ = true;
        is_generating_ = false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Reset state - clear everything for fresh start
    stop_requested_ = false;
    tokens_.clear();
    current_pos_ = 0;
    n_past_ = 0;
    
    // Update sampler with new config if needed
    current_config_ = config;
    initSampler(config);
    
    // Tokenize the prompt
    if (!tokenizePrompt(prompt, tokens_)) {
        LOGE("tokenize failed");
        return false;
    }
    
    // Clear the memory (KV cache)
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem != nullptr) {
        llama_memory_clear(mem, true);
    }
    
    // Record start time
    eval_start_time_ = getCurrentTimeMs();
    stats_.prompt_tokens = tokens_.size();
    stats_.generated_tokens = 0;
    
    // Evaluate prompt tokens
    if (!evaluateTokens(tokens_, 0, tokens_.size())) {
        LOGE("prompt eval failed");
        return false;
    }
    
    stats_.prompt_eval_time_ms = getCurrentTimeMs() - eval_start_time_;
    
    n_past_ = tokens_.size();
    current_pos_ = tokens_.size();
    is_generating_ = true;
    
    return true;
}

bool InferenceEngine::startInferenceIncremental(const std::string& prompt, const InferenceConfig& config) {
    if (!isModelLoaded()) {
        LOGE("model not loaded");
        return false;
    }
    
    if (is_generating_) {
        stop_requested_ = true;
        is_generating_ = false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    stop_requested_ = false;
    current_config_ = config;
    initSampler(config);
    
    std::vector<llama_token> new_tokens;
    if (!tokenizePrompt(prompt, new_tokens)) {
        LOGE("tokenize failed");
        return false;
    }
    
    int available_space = current_config_.context_length - n_past_ - 32;
    if (static_cast<int>(new_tokens.size()) > available_space) {
        shiftContext(64);
    }
    
    eval_start_time_ = getCurrentTimeMs();
    stats_.prompt_tokens = new_tokens.size();
    stats_.generated_tokens = 0;
    
    if (!evaluateTokens(new_tokens, n_past_, new_tokens.size())) {
        LOGE("eval failed");
        return false;
    }
    
    tokens_.insert(tokens_.end(), new_tokens.begin(), new_tokens.end());
    stats_.prompt_eval_time_ms = getCurrentTimeMs() - eval_start_time_;
    
    n_past_ += new_tokens.size();
    current_pos_ = tokens_.size();
    is_generating_ = true;
    
    return true;
}

void InferenceEngine::clearCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem != nullptr) {
        llama_memory_clear(mem, true);
    }
    
    tokens_.clear();
    n_past_ = 0;
    current_pos_ = 0;
}

std::string InferenceEngine::getNextToken() {
    if (!is_generating_ || stop_requested_) {
        is_generating_ = false;
        return "";
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (n_past_ >= current_config_.context_length - 1) {
        is_generating_ = false;
        return "";
    }
    
    if (stats_.generated_tokens >= current_config_.max_tokens) {
        is_generating_ = false;
        return "";
    }
    
    // Sample next token
    llama_token new_token = sampleNextToken();
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    if (llama_vocab_is_eog(vocab, new_token)) {
        is_generating_ = false;
        return "";
    }
    
    // Convert token to text
    std::string token_text = tokenToString(new_token);
    
    // Add token to sequence
    tokens_.push_back(new_token);
    
    // Evaluate the new token
    std::vector<llama_token> single_token = {new_token};
    if (!evaluateTokens(single_token, n_past_, 1)) {
        LOGE("Failed to evaluate token");
        is_generating_ = false;
        return "";
    }
    
    n_past_++;
    stats_.generated_tokens++;
    current_pos_++;
    
    // Update stats
    double total_time = getCurrentTimeMs() - eval_start_time_;
    stats_.eval_time_ms = total_time - stats_.prompt_eval_time_ms;
    if (stats_.eval_time_ms > 0) {
        stats_.tokens_per_second = (stats_.generated_tokens * 1000.0) / stats_.eval_time_ms;
    }
    
    return token_text;
}

std::vector<std::string> InferenceEngine::getNextTokens(int count) {
    std::vector<std::string> result;
    result.reserve(count);
    
    for (int i = 0; i < count && is_generating_ && !stop_requested_; i++) {
        if (n_past_ >= current_config_.context_length - 16) {
            shiftContext(64);
        }
        
        std::string token = getNextToken();
        if (token.empty() && !is_generating_) {
            break;  // End of generation
        }
        result.push_back(token);
    }
    
    return result;
}

void InferenceEngine::shiftContext(int keep_tokens) {
    if (n_past_ <= keep_tokens) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    int shift_amount = n_past_ - keep_tokens;
    
    // Shift the KV cache using the new memory API
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem != nullptr) {
        // Remove old tokens from cache and shift remaining
        llama_memory_seq_rm(mem, 0, 0, shift_amount);
        llama_memory_seq_add(mem, 0, shift_amount, n_past_, -shift_amount);
    }
    
    // Update position
    n_past_ = keep_tokens;
    
    // Keep only recent tokens
    if (tokens_.size() > static_cast<size_t>(keep_tokens)) {
        tokens_.erase(tokens_.begin(), tokens_.end() - keep_tokens);
    }
    
    current_pos_ = tokens_.size();
}

bool InferenceEngine::isGenerating() const {
    return is_generating_;
}

void InferenceEngine::stopGeneration() {
    stop_requested_ = true;
    is_generating_ = false;
}

bool InferenceEngine::generateWithCallback(const std::string& prompt,
                                           const InferenceConfig& config,
                                           TokenCallback callback) {
    if (!startInference(prompt, config)) {
        return false;
    }
    
    while (is_generating_ && !stop_requested_) {
        std::string token = getNextToken();
        
        if (token.empty() && !is_generating_) {
            // Generation ended
            if (callback) {
                callback("", true);
            }
            break;
        }
        
        if (callback) {
            bool should_continue = callback(token, false);
            if (!should_continue) {
                stopGeneration();
                break;
            }
        }
    }
    
    is_generating_ = false;
    stats_.total_tokens = stats_.prompt_tokens + stats_.generated_tokens;
    
    return true;
}

bool InferenceEngine::tokenizePrompt(const std::string& prompt, std::vector<llama_token>& tokens) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    // Allocate buffer for tokens (prompt length + some extra)
    int max_tokens = prompt.length() + 32;
    tokens.resize(max_tokens);
    
    // Tokenize using vocabulary
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.length(),
        tokens.data(),
        max_tokens,
        true,   // add_special (BOS token)
        false   // parse_special
    );
    
    if (n_tokens < 0) {
        // Buffer too small, resize and retry
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.length(),
            tokens.data(),
            -n_tokens,
            true,
            false
        );
    }
    
    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return false;
    }
    
    tokens.resize(n_tokens);
    return true;
}

bool InferenceEngine::evaluateTokens(const std::vector<llama_token>& tokens, int n_past, int n_tokens) {
    // OPTIMIZATION: Use llama_batch_get_one for single token (most common case)
    // This avoids batch allocation overhead
    if (n_tokens == 1) {
        llama_token token = tokens[0];
        llama_batch batch = llama_batch_get_one(&token, 1);
        
        // Set position for single token
        // Note: llama_batch_get_one sets pos[0] = 0 by default, we need to override
        // Actually, llama_decode handles position based on n_past in the KV cache
        
        if (llama_decode(ctx_, batch) != 0) {
            LOGE("llama_decode failed for single token");
            return false;
        }
        return true;
    }
    
    // For multiple tokens (prompt evaluation), use batched processing
    llama_batch batch = llama_batch_init(current_config_.batch_size, 0, 1);
    
    // Process tokens in batches
    for (int i = 0; i < n_tokens; i += current_config_.batch_size) {
        int n_eval = std::min(current_config_.batch_size, n_tokens - i);
        
        // Clear batch
        batch.n_tokens = 0;
        
        // Add tokens to batch
        for (int j = 0; j < n_eval; j++) {
            batch.token[batch.n_tokens] = tokens[i + j];
            batch.pos[batch.n_tokens] = n_past + i + j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            // Only compute logits for the last token
            batch.logits[batch.n_tokens] = (i + j == n_tokens - 1);
            batch.n_tokens++;
        }
        
        // Decode batch
        if (llama_decode(ctx_, batch) != 0) {
            LOGE("llama_decode failed");
            llama_batch_free(batch);
            return false;
        }
    }
    
    llama_batch_free(batch);
    return true;
}

llama_token InferenceEngine::sampleNextToken() {
    // Get logits from the last token
    llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);
    
    // Accept the token (updates sampler state)
    llama_sampler_accept(sampler_, new_token);
    
    return new_token;
}

std::string InferenceEngine::tokenToString(llama_token token) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    // Get the token text
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    
    if (n < 0) {
        // Buffer too small (shouldn't happen for single tokens)
        return "";
    }
    
    return std::string(buf, n);
}

GenerationStats InferenceEngine::getStats() const {
    return stats_;
}

void InferenceEngine::resetStats() {
    stats_ = GenerationStats();
}

size_t InferenceEngine::getModelMemoryUsage() const {
    if (model_ == nullptr) return 0;
    return llama_model_size(model_);
}

size_t InferenceEngine::getContextMemoryUsage() const {
    if (ctx_ == nullptr) return 0;
    // Approximate context memory usage
    return llama_n_ctx(ctx_) * llama_model_n_embd(model_) * sizeof(float) * 2;
}

int64_t InferenceEngine::getCurrentTimeMs() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

// ============================================================================
// MULTI-THREADED GENERATION
// ============================================================================

void InferenceEngine::setTokenCallback(std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    token_callback_ = callback;
}

void InferenceEngine::clearTokenCallback() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    token_callback_ = nullptr;
}

std::string InferenceEngine::popTokenFromQueue() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (output_buffer_.empty()) {
        return "";
    }
    std::string result = output_buffer_;
    output_buffer_.clear();
    return result;
}

void InferenceEngine::stopThreads() {
    stop_requested_ = true;
    generation_complete_ = true;
    
    // Wake up the processor thread
    queue_cv_.notify_all();
    
    // Wait for threads to finish
    if (generation_thread_.joinable()) {
        generation_thread_.join();
    }
    if (processor_thread_.joinable()) {
        processor_thread_.join();
    }
}

void InferenceEngine::generationThreadFunc() {
#ifdef __ANDROID__
    // Set thread priority to real-time for lower latency
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    // Try to pin to performance cores (cores 4-7 on most big.LITTLE devices)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int num_cores = std::thread::hardware_concurrency();
    // Use upper half cores (performance cores on big.LITTLE)
    for (int i = num_cores / 2; i < num_cores; i++) {
        CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    while (!stop_requested_ && is_generating_) {
        // Check limits
        if (n_past_ >= current_config_.context_length - 1 ||
            stats_.generated_tokens >= current_config_.max_tokens) {
            break;
        }
        
        // Sample next token - this is fast
        llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);
        llama_sampler_accept(sampler_, new_token);
        
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        
        // Push token to queue (producer side)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            token_queue_.push(new_token);
        }
        queue_cv_.notify_one();
        
        // Evaluate the token - THE SLOW PART (llama_decode)
        // This is where 90%+ of time is spent
        std::vector<llama_token> single_token = {new_token};
        tokens_.push_back(new_token);
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!evaluateTokens(single_token, n_past_, 1)) {
                LOGE("Failed to evaluate token");
                break;
            }
        }
        
        n_past_++;
        stats_.generated_tokens++;
    }
    
    // Mark generation as complete
    generation_complete_ = true;
    is_generating_ = false;
    queue_cv_.notify_all();
    
    // Update final stats
    double total_time = getCurrentTimeMs() - eval_start_time_;
    stats_.eval_time_ms = total_time - stats_.prompt_eval_time_ms;
    if (stats_.eval_time_ms > 0) {
        stats_.tokens_per_second = (stats_.generated_tokens * 1000.0) / stats_.eval_time_ms;
    }
}

void InferenceEngine::processorThreadFunc() {
    
    std::string batch_buffer;
    batch_buffer.reserve(256);
    int batch_count = 0;
    auto last_flush = std::chrono::steady_clock::now();
    
    while (!generation_complete_ || !token_queue_.empty()) {
        llama_token token;
        bool got_token = false;
        
        // Wait for token with timeout
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (queue_cv_.wait_for(lock, std::chrono::milliseconds(10), 
                [this] { return !token_queue_.empty() || generation_complete_; })) {
                if (!token_queue_.empty()) {
                    token = token_queue_.front();
                    token_queue_.pop();
                    got_token = true;
                }
            }
        }
        
        if (got_token) {
            // Convert token to string (moved OFF the main generation thread)
            std::string token_text = tokenToString(token);
            batch_buffer += token_text;
            batch_count++;
        }
        
        // Flush batch when we have BATCH_SIZE tokens or timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush);
        
        bool should_flush = (batch_count >= BATCH_SIZE) || 
                           (elapsed.count() > 50 && batch_count > 0) ||
                           (generation_complete_ && batch_count > 0);
        
        if (should_flush && !batch_buffer.empty()) {
            // Store in output buffer for polling
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                output_buffer_ += batch_buffer;
            }
            
            // Call callback if registered
            {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (token_callback_) {
                    token_callback_(batch_buffer);
                }
            }
            
            batch_buffer.clear();
            batch_count = 0;
            last_flush = now;
        }
    }
    
    // Flush any remaining tokens
    if (!batch_buffer.empty()) {
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            output_buffer_ += batch_buffer;
        }
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (token_callback_) {
                token_callback_(batch_buffer);
            }
        }
    }
}

bool InferenceEngine::startInferenceThreaded(const std::string& prompt, const InferenceConfig& config) {
    if (!isModelLoaded()) {
        LOGE("model not loaded");
        return false;
    }
    
    if (is_generating_) {
        stopThreads();
    }
    
    // Reset state
    stop_requested_ = false;
    generation_complete_ = false;
    
    // Clear queues and buffers
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!token_queue_.empty()) token_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        output_buffer_.clear();
    }
    
    // Standard setup (same as startInferenceIncremental)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        current_config_ = config;
        initSampler(config);
        
        // Tokenize the new prompt
        std::vector<llama_token> new_tokens;
        if (!tokenizePrompt(prompt, new_tokens)) {
            LOGE("Failed to tokenize prompt");
            return false;
        }
        
        int available_space = current_config_.context_length - n_past_ - 32;
        if (static_cast<int>(new_tokens.size()) > available_space) {
            shiftContext(64);
        }
        
        eval_start_time_ = getCurrentTimeMs();
        stats_.prompt_tokens = new_tokens.size();
        stats_.generated_tokens = 0;
        
        if (!evaluateTokens(new_tokens, n_past_, new_tokens.size())) {
            LOGE("eval failed");
            return false;
        }
        
        tokens_.insert(tokens_.end(), new_tokens.begin(), new_tokens.end());
        stats_.prompt_eval_time_ms = getCurrentTimeMs() - eval_start_time_;
        
        n_past_ += new_tokens.size();
    }
    
    is_generating_ = true;
    
    // Start worker threads
    generation_thread_ = std::thread(&InferenceEngine::generationThreadFunc, this);
    processor_thread_ = std::thread(&InferenceEngine::processorThreadFunc, this);
    
    return true;
}

} // namespace cortex
