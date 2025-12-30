#pragma once

#include <string>
#include <vector>
#include <functional>

namespace cortex {

// Model loading
bool loadModel(const std::string& modelPath);
void unloadModel();
bool isModelLoaded();
std::string getModelInfo();

// Text generation
bool startGeneration(const std::string& prompt, float temperature, float top_p, 
                     int top_k, int max_tokens);
bool startGenerationIncremental(const std::string& prompt, float temperature, float top_p, 
                                int top_k, int max_tokens);  // KV cache reuse
bool startGenerationTurbo(const std::string& prompt);  // Multi-threaded with quality sampling
bool startGenerationThreaded(const std::string& prompt, float temperature, float top_p,
                             int top_k, int max_tokens);  // Multi-threaded generation
std::string getNextToken();
std::vector<std::string> getNextTokens(int count = 4);  // Batch token decoding
std::string getNextTokensBatch(int count);  // Returns concatenated string for speed
std::string getBufferedTokens();  // Get tokens from threaded generation buffer
bool isGenerating();
void stopGeneration();

// Threaded generation callbacks
void setTokenCallback(std::function<void(const std::string&)> callback);
void clearTokenCallback();

// Cache management
void clearCache();
int getCachedTokenCount();

// Statistics
std::string getStats();
void resetStats();
std::string getMemoryInfo();
long getMemoryUsage();

} // namespace cortex
