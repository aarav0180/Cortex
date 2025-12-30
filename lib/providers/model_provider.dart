import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:async';
import 'package:http/http.dart' as http;
import '../models/app_models.dart';
import '../services/inference_engine.dart';

class ModelProvider extends ChangeNotifier {
  final List<Model> _models = [
    // === TINY MODELS (< 200M params) - Fastest ===
    Model(
      id: 'smollm-135m-q8',
      name: 'SmolLM 135M (Q8)',
      description: 'HuggingFace SmolLM 135M - Ultra fast, 8-bit',
      sizeGB: 0.15,
      downloadUrl: 'https://huggingface.co/QuantFactory/SmolLM-135M-Instruct-GGUF/resolve/main/SmolLM-135M-Instruct.Q8_0.gguf',
    ),
    Model(
      id: 'qwen2-0.5b-q4',
      name: 'Qwen2 0.5B (Q4)',
      description: 'Alibaba Qwen2 0.5B - Very fast',
      sizeGB: 0.35,
      downloadUrl: 'https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf',
    ),
    Model(
      id: 'qwen-1.5-0.5b-q4',
      name: 'Qwen 1.5 0.5B (Q4)',
      description: 'Alibaba Qwen 1.5 0.5B - Fast and capable',
      sizeGB: 0.4,
      downloadUrl: 'https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_0.gguf',
    ),
    // === SMALL MODELS (1-2B params) - Balanced ===
    Model(
      id: 'tinyllama-1.1b-q4',
      name: 'TinyLlama 1.1B (Q4)',
      description: 'TinyLlama Chat - Good balance of speed/quality',
      sizeGB: 0.6,
      downloadUrl: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf',
    ),
    Model(
      id: 'llama-3.2-1b-q4',
      name: 'Llama 3.2 1B (Q4)',
      description: 'Meta Llama 3.2 1B - Latest and capable',
      sizeGB: 0.7,
      downloadUrl: 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf',
    ),
    Model(
      id: 'stablelm-2-1.6b-q4',
      name: 'StableLM 2 1.6B (Q4)',
      description: 'Stability AI StableLM 2 - Efficient',
      sizeGB: 0.9,
      downloadUrl: 'https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b/resolve/main/stablelm-2-zephyr-1_6b-Q4_0.gguf',
    ),
    // === MEDIUM MODELS (2-3B params) - Best Quality ===
    Model(
      id: 'phi-2-q4',
      name: 'Phi-2 2.7B (Q4)', 
      description: 'Microsoft Phi-2 - Excellent reasoning',
      sizeGB: 1.6,
      downloadUrl: 'https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf',
    ),
    Model(
      id: 'gemma-2b-q4',
      name: 'Gemma 2B (Q4)',
      description: 'Google Gemma 2B - High quality', 
      sizeGB: 1.4,
      downloadUrl: 'https://huggingface.co/google/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.gguf',
    ),
  ];

  Model? _selectedModel;
  String? _loadedModelPath;
  int _memoryUsage = 0;
  Timer? _memoryTimer;

  List<Model> get models => List.unmodifiable(_models);
  Model? get selectedModel => _selectedModel;
  bool get hasSelectedModel => _selectedModel != null;
  bool get hasLoadedModel => _loadedModelPath != null;
  int get memoryUsageMB => (_memoryUsage / 1024 / 1024).round();

  ModelProvider() {
    _startMemoryMonitoring();
  }

  @override
  void dispose() {
    _memoryTimer?.cancel();
    super.dispose();
  }

  void _startMemoryMonitoring() {
    _memoryTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      _memoryUsage = await InferenceEngine.getMemoryUsage();
      notifyListeners();
    });
  }

  /// Download a model from HuggingFace
  Future<void> downloadModel(String modelId) async {
    final modelIndex = _models.indexWhere((m) => m.id == modelId);
    if (modelIndex == -1) return;

    final model = _models[modelIndex];
    if (model.status == ModelStatus.downloaded) return;

    try {
      // Update status to downloading
      _models[modelIndex] = model.copyWith(
        status: ModelStatus.downloading,
        downloadProgress: 0.0,
      );
      notifyListeners();

      // Get app documents directory for storing models
      final appDir = await getApplicationDocumentsDirectory();
      final modelsDir = Directory('${appDir.path}/models');
      if (!modelsDir.existsSync()) {
        modelsDir.createSync(recursive: true);
      }

      final modelFile = File('${modelsDir.path}/${model.id}.gguf');
      
      // Download the model file from HuggingFace with progress tracking
      if (model.downloadUrl == null) {
        throw Exception('Download URL not available');
      }
      
      print('starting download: ${model.downloadUrl}');
      
      final request = http.Request('GET', Uri.parse(model.downloadUrl!));
      final response = await request.send();
      
      if (response.statusCode != 200) {
        throw Exception('Failed to download model: ${response.statusCode} ${response.reasonPhrase}');
      }

      final contentLength = response.contentLength ?? 0;
      print('model size: ${(contentLength / 1024 / 1024).toStringAsFixed(1)} MB');
      
      final sink = modelFile.openWrite();
      int downloaded = 0;
      
      await response.stream.listen((chunk) {
        sink.add(chunk);
        downloaded += chunk.length;
        
        final progress = contentLength > 0 ? downloaded / contentLength : 0.0;
        _models[modelIndex] = model.copyWith(
          status: ModelStatus.downloading,
          downloadProgress: progress,
        );
        notifyListeners();
      }).asFuture();
      
      await sink.close();
      print('model downloaded: ${modelFile.path}');
      
      // Update status to downloaded
      _models[modelIndex] = model.copyWith(
        status: ModelStatus.downloaded,
        downloadProgress: 1.0,
        localPath: modelFile.path,
      );

    } catch (e) {
      print('download failed: $e');
      
      // Update status to error
      _models[modelIndex] = model.copyWith(
        status: ModelStatus.error,
        downloadProgress: 0.0,
      );
    }
    
    notifyListeners();
  }

  /// Load a downloaded model into the inference engine
  Future<bool> loadModel(String modelId) async {
    final model = _models.firstWhere((m) => m.id == modelId);
    
    if (model.status != ModelStatus.downloaded || model.localPath == null) {
      print('model not downloaded: $modelId');
      return false;
    }

    try {
      // Unload any existing model first
      if (_loadedModelPath != null) {
        print('unloading previous model');
        await unloadModel();
      }
      
      print('loading model: ${model.localPath}');
      
      final success = await InferenceEngine.loadModel(model.localPath!);
      
      if (success) {
        _selectedModel = model;
        _loadedModelPath = model.localPath;
        print('model loaded: $modelId');
        notifyListeners();
        return true;
      } else {
        print('native engine failed to load: $modelId');
        _selectedModel = null;
        _loadedModelPath = null;
        notifyListeners();
        return false;
      }
      
    } catch (e) {
      print('exception loading model: $e');
      _selectedModel = null;
      _loadedModelPath = null;
      notifyListeners();
      return false;
    }
  }

  /// Unload the current model
  Future<void> unloadModel() async {
    await InferenceEngine.stopGeneration();
    await InferenceEngine.unloadModel();
    _selectedModel = null;
    _loadedModelPath = null;
    notifyListeners();
  }

  /// Delete a downloaded model
  Future<void> deleteModel(String modelId) async {
    final modelIndex = _models.indexWhere((m) => m.id == modelId);
    if (modelIndex == -1) return;

    final model = _models[modelIndex];
    
    // Unload if this is the currently loaded model
    if (_selectedModel?.id == modelId) {
      await unloadModel();
    }

    // Delete the file
    if (model.localPath != null) {
      final file = File(model.localPath!);
      if (file.existsSync()) {
        await file.delete();
      }
    }

    // Update status
    _models[modelIndex] = model.copyWith(
      status: ModelStatus.notDownloaded,
      downloadProgress: 0.0,
      localPath: null,
    );

    notifyListeners();
  }

  /// Check which models are already downloaded
  Future<void> checkDownloadedModels() async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory('${appDir.path}/models');
    
    if (!modelsDir.existsSync()) return;

    for (int i = 0; i < _models.length; i++) {
      final model = _models[i];
      final modelFile = File('${modelsDir.path}/${model.id}.gguf');
      
      if (modelFile.existsSync()) {
        _models[i] = model.copyWith(
          status: ModelStatus.downloaded,
          localPath: modelFile.path,
        );
      }
    }
    
    notifyListeners();
  }
}