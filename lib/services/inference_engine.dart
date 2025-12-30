import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:convert';

class InferenceEngine {
  static const MethodChannel _channel = MethodChannel('inference_engine');

  static Future<bool> loadModel(String modelPath) async {
    final result = await _channel.invokeMethod('loadModel', {
      'modelPath': modelPath,
    });
    print('model loaded: $modelPath');
    return result == true;
  }

  static Future<void> unloadModel() async {
    await _channel.invokeMethod('unloadModel');
    print('model unloaded');
  }

  static Future<bool> startInference(String prompt) async {
    final result = await _channel.invokeMethod('startInference', {
      'prompt': prompt,
    });
    print('starting inference: ${prompt.length} chars');
    return result == true;
  }

  static Future<bool> startInferenceIncremental(String prompt) async {
    final result = await _channel.invokeMethod('startInferenceIncremental', {
      'prompt': prompt,
    });
    print('incremental inference: ${prompt.length} chars');
    return result == true;
  }

  static Future<bool> startInferenceTurbo(String prompt) async {
    final result = await _channel.invokeMethod('startInferenceTurbo', {
      'prompt': prompt,
    });
    print('optimized inference: ${prompt.length} chars');
    return result == true;
  }

  static Future<String> getBufferedTokens() async {
    final result = await _channel.invokeMethod('getBufferedTokens');
    return result?.toString() ?? '';
  }

  static Future<void> clearCache() async {
    await _channel.invokeMethod('clearCache');
  }

  static Future<int> getCachedTokenCount() async {
    final result = await _channel.invokeMethod('getCachedTokenCount');
    return result ?? 0;
  }

  static Future<String> getNextToken() async {
    final result = await _channel.invokeMethod('getNextToken');
    return result?.toString() ?? '';
  }

  static Future<List<String>> getNextTokens({int count = 4}) async {
    final result = await _channel.invokeMethod('getNextTokens', {
      'count': count,
    });
    if (result is List) {
      return result.map((e) => e?.toString() ?? '').toList();
    }
    return [];
  }

  static Future<String> getNextTokensBatch({int count = 8}) async {
    final result = await _channel.invokeMethod('getNextTokensBatch', {
      'count': count,
    });
    return result?.toString() ?? '';
  }

  static Future<bool> isGenerating() async {
    final result = await _channel.invokeMethod('isGenerating');
    return result == true;
  }

  static Future<void> stopGeneration() async {
    await _channel.invokeMethod('stopGeneration');
  }

  static Future<int> getMemoryUsage() async {
    final result = await _channel.invokeMethod('getMemoryUsage');
    return result?.toInt() ?? 0;
  }

  static Future<Map<String, dynamic>> getStats() async {
    final result = await _channel.invokeMethod('getStats');
    if (result is String && result.isNotEmpty) {
      try {
        return Map<String, dynamic>.from(
          const JsonDecoder().convert(result) as Map,
        );
      } catch (e) {
        print('failed to parse stats: $e');
      }
    }
    return {};
  }

  static Future<Map<String, dynamic>> getMemoryInfo() async {
    final result = await _channel.invokeMethod('getMemoryInfo');
    if (result is String && result.isNotEmpty) {
      try {
        return Map<String, dynamic>.from(
          const JsonDecoder().convert(result) as Map,
        );
      } catch (e) {
        print('failed to parse memory info: $e');
      }
    }
    return {};
  }

  static Future<void> resetStats() async {
    await _channel.invokeMethod('resetStats');
  }

  static Stream<String> streamTokens() async* {
    try {
      bool isActive = await isGenerating();
      int tokenCount = 0;
      
      while (isActive) {
        if (tokenCount < 3) {
          final token = await getNextToken();
          if (token.isNotEmpty) {
            yield token;
            tokenCount++;
          }
        } else {
          final tokens = await getNextTokens(count: 4);
          if (tokens.isNotEmpty) {
            for (final token in tokens) {
              if (token.isNotEmpty) {
                yield token;
                tokenCount++;
              }
            }
          }
        }
        isActive = await isGenerating();
      }
    } catch (e) {
      print('stream error: $e');
    }
  }

  static Stream<String> streamTokensThreaded() async* {
    try {
      bool isActive = await isGenerating();
      
      while (isActive) {
        await Future.delayed(const Duration(milliseconds: 25));
        final tokens = await getBufferedTokens();
        if (tokens.isNotEmpty) {
          yield tokens;
        }
        isActive = await isGenerating();
      }
      
      final remaining = await getBufferedTokens();
      if (remaining.isNotEmpty) {
        yield remaining;
      }
    } catch (e) {
      print('stream error: $e');
    }
  }
}
