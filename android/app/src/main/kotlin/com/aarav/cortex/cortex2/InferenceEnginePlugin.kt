package com.aarav.cortex.cortex2

import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.*
import android.os.Handler
import android.os.Looper

class InferenceEnginePlugin : FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    
    private lateinit var channel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var eventSink: EventChannel.EventSink? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    
    // Token streaming job
    private var streamingJob: Job? = null
    
    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "inference_engine")
        channel.setMethodCallHandler(this)
        
        // EventChannel for push-based token streaming
        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, "inference_engine/tokens")
        eventChannel.setStreamHandler(this)
    }
    
    // EventChannel.StreamHandler implementation
    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }
    
    override fun onCancel(arguments: Any?) {
        eventSink = null
        streamingJob?.cancel()
        streamingJob = null
    }
    
    // Push tokens to EventChannel from native callback or polling
    private fun pushTokens(tokens: String) {
        if (tokens.isNotEmpty()) {
            mainHandler.post {
                eventSink?.success(tokens)
            }
        }
    }
    
    // Start polling for tokens (used when EventChannel is active)
    private fun startTokenPolling() {
        streamingJob?.cancel()
        streamingJob = scope.launch {
            while (isActive && isGeneratingNative()) {
                val tokens = getBufferedTokensNative()
                if (tokens.isNotEmpty()) {
                    pushTokens(tokens)
                }
                delay(20) // Poll every 20ms
            }
            // Final flush
            val remaining = getBufferedTokensNative()
            if (remaining.isNotEmpty()) {
                pushTokens(remaining)
            }
            // Signal end of stream
            mainHandler.post {
                eventSink?.endOfStream()
            }
        }
    }
    
    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "loadModel" -> {
                val modelPath = call.argument<String>("modelPath")
                if (modelPath != null) {
                    scope.launch {
                        val success = loadModelNative(modelPath)
                        withContext(Dispatchers.Main) {
                            result.success(success)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Model path is required", null)
                }
            }
            
            "unloadModel" -> {
                // Run on background thread to avoid blocking while waiting for generation to stop
                scope.launch {
                    unloadModelNative()
                    withContext(Dispatchers.Main) {
                        result.success(true)
                    }
                }
            }
            
            "isModelLoaded" -> {
                result.success(isModelLoadedNative())
            }
            
            "getModelInfo" -> {
                result.success(getModelInfoNative())
            }
            
            "startInference" -> {
                val prompt = call.argument<String>("prompt")
                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.7f
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.9f
                val topK = call.argument<Int>("topK") ?: 40
                val maxTokens = call.argument<Int>("maxTokens") ?: 2048
                
                if (prompt != null) {
                    // Run on background thread - prompt evaluation can take seconds
                    scope.launch {
                        val success = startGenerationNative(prompt, temperature, topP, topK, maxTokens)
                        withContext(Dispatchers.Main) {
                            result.success(success)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Prompt is required", null)
                }
            }
            
            "startInferenceIncremental" -> {
                // OPTIMIZATION: Use incremental inference with KV cache reuse
                val prompt = call.argument<String>("prompt")
                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.7f
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.9f
                val topK = call.argument<Int>("topK") ?: 40
                val maxTokens = call.argument<Int>("maxTokens") ?: 2048
                
                if (prompt != null) {
                    scope.launch {
                        val success = startGenerationIncrementalNative(prompt, temperature, topP, topK, maxTokens)
                        withContext(Dispatchers.Main) {
                            result.success(success)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Prompt is required", null)
                }
            }
            
            "startInferenceTurbo" -> {
                // TURBO MODE: Multi-threaded with quality sampling
                val prompt = call.argument<String>("prompt")
                if (prompt != null) {
                    scope.launch {
                        val success = startGenerationTurboNative(prompt)
                        if (success && eventSink != null) {
                            // Start pushing tokens via EventChannel
                            startTokenPolling()
                        }
                        withContext(Dispatchers.Main) {
                            result.success(success)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Prompt is required", null)
                }
            }
            
            "startInferenceThreaded" -> {
                // Multi-threaded generation with custom parameters
                val prompt = call.argument<String>("prompt")
                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.7f
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.9f
                val topK = call.argument<Int>("topK") ?: 40
                val maxTokens = call.argument<Int>("maxTokens") ?: 2048
                
                if (prompt != null) {
                    scope.launch {
                        val success = startGenerationThreadedNative(prompt, temperature, topP, topK, maxTokens)
                        if (success && eventSink != null) {
                            startTokenPolling()
                        }
                        withContext(Dispatchers.Main) {
                            result.success(success)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Prompt is required", null)
                }
            }
            
            "getBufferedTokens" -> {
                // Get tokens from threaded generation buffer
                scope.launch {
                    val tokens = getBufferedTokensNative()
                    withContext(Dispatchers.Main) {
                        result.success(tokens)
                    }
                }
            }
            
            "clearCache" -> {
                clearCacheNative()
                result.success(true)
            }
            
            "getCachedTokenCount" -> {
                result.success(getCachedTokenCountNative())
            }
            
            "getNextToken" -> {
                // Run on background thread - token generation blocks
                scope.launch {
                    val token = getNextTokenNative()
                    withContext(Dispatchers.Main) {
                        result.success(token)
                    }
                }
            }
            
            "getNextTokens" -> {
                // Batch token retrieval for reduced JNI overhead
                val count = call.argument<Int>("count") ?: 4
                scope.launch {
                    val tokens = getNextTokensNative(count)
                    withContext(Dispatchers.Main) {
                        result.success(tokens.toList())
                    }
                }
            }
            
            "getNextTokensBatch" -> {
                // TURBO: Get batch as single concatenated string (faster)
                val count = call.argument<Int>("count") ?: 8
                scope.launch {
                    val tokens = getNextTokensBatchNative(count)
                    withContext(Dispatchers.Main) {
                        result.success(tokens)
                    }
                }
            }
            
            "isGenerating" -> {
                result.success(isGeneratingNative())
            }
            
            "stopGeneration" -> {
                // Run on background thread to avoid blocking
                scope.launch {
                    stopGenerationNative()
                    withContext(Dispatchers.Main) {
                        result.success(true)
                    }
                }
            }
            
            "getStats" -> {
                result.success(getStatsNative())
            }
            
            "resetStats" -> {
                resetStatsNative()
                result.success(true)
            }
            
            "getMemoryInfo" -> {
                result.success(getMemoryInfoNative())
            }
            
            "getMemoryUsage" -> {
                // Return memory usage as integer (bytes)
                result.success(getMemoryUsageNative())
            }
            
            "runBenchmark" -> {
                val numTokens = call.argument<Int>("numTokens") ?: 100
                scope.launch {
                    val stats = runBenchmarkNative(numTokens)
                    withContext(Dispatchers.Main) {
                        result.success(stats)
                    }
                }
            }
            
            else -> {
                result.notImplemented()
            }
        }
    }
    
    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        streamingJob?.cancel()
        scope.cancel()
        unloadModelNative()
    }
    
    companion object {
        init {
            System.loadLibrary("llama_jni")
        }
    }
    
    // Native method declarations - must match JNI function names
    private external fun loadModelNative(modelPath: String): Boolean
    private external fun unloadModelNative()
    private external fun isModelLoadedNative(): Boolean
    private external fun getModelInfoNative(): String
    private external fun startGenerationNative(prompt: String, temperature: Float, topP: Float, topK: Int, maxTokens: Int): Boolean
    private external fun startGenerationIncrementalNative(prompt: String, temperature: Float, topP: Float, topK: Int, maxTokens: Int): Boolean
    private external fun startGenerationThreadedNative(prompt: String, temperature: Float, topP: Float, topK: Int, maxTokens: Int): Boolean
    private external fun clearCacheNative()
    private external fun getCachedTokenCountNative(): Int
    private external fun getNextTokenNative(): String
    private external fun getNextTokensNative(count: Int): Array<String>
    private external fun getNextTokensBatchNative(count: Int): String
    private external fun getBufferedTokensNative(): String
    private external fun isGeneratingNative(): Boolean
    private external fun stopGenerationNative()
    private external fun getStatsNative(): String
    private external fun resetStatsNative()
    private external fun getMemoryInfoNative(): String
    private external fun getMemoryUsageNative(): Long
    private external fun runBenchmarkNative(numTokens: Int): String
    private external fun startGenerationTurboNative(prompt: String): Boolean
}