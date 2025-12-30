package com.aarav.cortex.cortex2

/**
 * JNI wrapper for llama.cpp
 * Provides Kotlin interface to native C++ functions
 */
class LlamaJNI {
    companion object {
        init {
            System.loadLibrary("llama")
        }
    }

    external fun nativeInit(): Int
    external fun nativeLoadModel(modelPath: String): Long
    external fun nativeCreateContext(modelPtr: Long, nCtx: Int): Long
    external fun nativeTokenize(ctxPtr: Long, text: String): IntArray
    external fun nativeDecode(ctxPtr: Long, token: Int): Int
    external fun nativeSample(ctxPtr: Long, temperature: Double): Int
    external fun nativeFreeModel(modelPtr: Long)
    external fun nativeFreeContext(ctxPtr: Long)
}

