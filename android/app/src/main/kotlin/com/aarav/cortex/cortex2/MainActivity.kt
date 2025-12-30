package com.aarav.cortex.cortex2

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        // Register the inference engine plugin
        flutterEngine.plugins.add(InferenceEnginePlugin())
    }
}
