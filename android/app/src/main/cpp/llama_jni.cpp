#ifdef __ANDROID__
#include <jni.h>
#include <android/log.h>
#define LOG_TAG "CortexJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
typedef void* JNIEnv;
typedef void* jobject;
typedef void* jstring;
typedef void* JavaVM;
typedef void* jclass;
typedef void* jobjectArray;
typedef int jint;
typedef long jlong;
typedef float jfloat;
typedef bool jboolean;
#define JNI_TRUE true
#define JNI_FALSE false
#define JNI_VERSION_1_6 0x00010006
#define JNIEXPORT
#define JNICALL
#define LOGI(...)
#define LOGE(...)
#endif
#include <string>
#include "platform_channel.h"

// Helper to convert jstring to std::string
static std::string jstringToString(JNIEnv* env, jstring jstr) {
#ifdef __ANDROID__
    if (jstr == nullptr) return "";
    
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (chars == nullptr) return "";
    
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
#else
    return ""; // Non-Android stub
#endif
}

// Helper to convert std::string to jstring
static jstring stringToJstring(JNIEnv* env, const std::string& str) {
#ifdef __ANDROID__
    return env->NewStringUTF(str.c_str());
#else
    return nullptr; // Non-Android stub
#endif
}

extern "C" {

// JNI_OnLoad - called when library is loaded
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("llama_jni library loaded");
    return JNI_VERSION_1_6;
}

// Load a model from the given path
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_loadModelNative(
    JNIEnv* env,
    jobject thiz,
    jstring model_path
) {
    std::string path = jstringToString(env, model_path);
    LOGI("JNI loadModel: %s", path.c_str());
    
    bool result = cortex::loadModel(path);
    return result ? JNI_TRUE : JNI_FALSE;
}

// Unload the current model
JNIEXPORT void JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_unloadModelNative(
    JNIEnv* env,
    jobject thiz
) {
    LOGI("JNI unloadModel");
    cortex::unloadModel();
}

// Check if a model is loaded
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_isModelLoadedNative(
    JNIEnv* env,
    jobject thiz
) {
    return cortex::isModelLoaded() ? JNI_TRUE : JNI_FALSE;
}

// Get model information
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getModelInfoNative(
    JNIEnv* env,
    jobject thiz
) {
    std::string info = cortex::getModelInfo();
    return stringToJstring(env, info);
}

// Start text generation
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_startGenerationNative(
    JNIEnv* env,
    jobject thiz,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jint max_tokens
) {
    std::string promptStr = jstringToString(env, prompt);
    LOGI("JNI startGeneration: prompt length=%zu", promptStr.length());
    
    bool result = cortex::startGeneration(
        promptStr,
        static_cast<float>(temperature),
        static_cast<float>(top_p),
        static_cast<int>(top_k),
        static_cast<int>(max_tokens)
    );
    
    return result ? JNI_TRUE : JNI_FALSE;
}

// Start incremental generation (KV cache reuse for faster prompt eval)
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_startGenerationIncrementalNative(
    JNIEnv* env,
    jobject thiz,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jint max_tokens
) {
    std::string promptStr = jstringToString(env, prompt);
    LOGI("JNI startGenerationIncremental: prompt length=%zu", promptStr.length());
    
    bool result = cortex::startGenerationIncremental(
        promptStr,
        static_cast<float>(temperature),
        static_cast<float>(top_p),
        static_cast<int>(top_k),
        static_cast<int>(max_tokens)
    );
    
    return result ? JNI_TRUE : JNI_FALSE;
}

// Clear KV cache
JNIEXPORT void JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_clearCacheNative(
    JNIEnv* env,
    jobject thiz
) {
    LOGI("JNI clearCache");
    cortex::clearCache();
}

// Get cached token count
JNIEXPORT jint JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getCachedTokenCountNative(
    JNIEnv* env,
    jobject thiz
) {
    return static_cast<jint>(cortex::getCachedTokenCount());
}

// Get the next generated token
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getNextTokenNative(
    JNIEnv* env,
    jobject thiz
) {
    std::string token = cortex::getNextToken();
    return stringToJstring(env, token);
}

// Get batch of next tokens (reduces JNI overhead)
JNIEXPORT jobjectArray JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getNextTokensNative(
    JNIEnv* env,
    jobject thiz,
    jint count
) {
    std::vector<std::string> tokens = cortex::getNextTokens(static_cast<int>(count));
    
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(tokens.size(), stringClass, nullptr);
    
    for (size_t i = 0; i < tokens.size(); i++) {
        jstring jstr = stringToJstring(env, tokens[i]);
        env->SetObjectArrayElement(result, i, jstr);
        env->DeleteLocalRef(jstr);
    }
    
    return result;
}

// Check if generation is in progress
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_isGeneratingNative(
    JNIEnv* env,
    jobject thiz
) {
    return cortex::isGenerating() ? JNI_TRUE : JNI_FALSE;
}

// Stop ongoing generation
JNIEXPORT void JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_stopGenerationNative(
    JNIEnv* env,
    jobject thiz
) {
    LOGI("JNI stopGeneration");
    cortex::stopGeneration();
}

// Get generation statistics
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getStatsNative(
    JNIEnv* env,
    jobject thiz
) {
    std::string stats = cortex::getStats();
    return stringToJstring(env, stats);
}

// Get memory information
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getMemoryInfoNative(
    JNIEnv* env,
    jobject thiz
) {
    std::string info = cortex::getMemoryInfo();
    return stringToJstring(env, info);
}

// Reset generation statistics
JNIEXPORT void JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_resetStatsNative(
    JNIEnv* env,
    jobject thiz
) {
    LOGI("JNI resetStats");
    cortex::resetStats();
}

// Start generation in TURBO mode (maximum speed)
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_startGenerationTurboNative(
    JNIEnv* env,
    jobject thiz,
    jstring prompt
) {
    const char* promptChars = env->GetStringUTFChars(prompt, nullptr);
    LOGI("JNI startGenerationTurbo: prompt length=%zu", strlen(promptChars));
    
    bool result = cortex::startGenerationTurbo(promptChars);
    env->ReleaseStringUTFChars(prompt, promptChars);
    return result;
}

// Get batch of tokens as concatenated string (faster than array)
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getNextTokensBatchNative(
    JNIEnv* env,
    jobject thiz,
    jint count
) {
    std::string tokens = cortex::getNextTokensBatch(count);
    return stringToJstring(env, tokens);
}

// Get memory usage in bytes
JNIEXPORT jlong JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getMemoryUsageNative(
    JNIEnv* env,
    jobject thiz
) {
    return cortex::getMemoryUsage();
}

// Start threaded generation with custom parameters
JNIEXPORT jboolean JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_startGenerationThreadedNative(
    JNIEnv* env,
    jobject thiz,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jint max_tokens
) {
    std::string promptStr = jstringToString(env, prompt);
    LOGI("JNI startGenerationThreaded: prompt length=%zu", promptStr.length());
    
    bool result = cortex::startGenerationThreaded(
        promptStr,
        static_cast<float>(temperature),
        static_cast<float>(top_p),
        static_cast<int>(top_k),
        static_cast<int>(max_tokens)
    );
    
    return result ? JNI_TRUE : JNI_FALSE;
}

// Get tokens from threaded generation buffer
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_getBufferedTokensNative(
    JNIEnv* env,
    jobject thiz
) {
    std::string tokens = cortex::getBufferedTokens();
    return stringToJstring(env, tokens);
}

// Benchmark function for testing
JNIEXPORT jstring JNICALL
Java_com_aarav_cortex_cortex2_InferenceEnginePlugin_runBenchmarkNative(
    JNIEnv* env,
    jobject thiz,
    jint num_tokens
) {
    LOGI("JNI runBenchmark: %d tokens", num_tokens);
    
    if (!cortex::isModelLoaded()) {
        return stringToJstring(env, "{\"error\":\"No model loaded\"}");
    }
    
    // Simple benchmark: generate tokens and measure speed
    std::string prompt = "Hello, my name is";
    bool started = cortex::startGeneration(prompt, 0.7f, 0.9f, 40, num_tokens);
    
    if (!started) {
        return stringToJstring(env, "{\"error\":\"Failed to start generation\"}");
    }
    
    int tokens_generated = 0;
    while (cortex::isGenerating() && tokens_generated < num_tokens) {
        std::string token = cortex::getNextToken();
        if (!token.empty()) {
            tokens_generated++;
        }
    }
    
    std::string stats = cortex::getStats();
    return stringToJstring(env, stats);
}

} // extern "C"
