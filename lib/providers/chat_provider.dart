import 'package:flutter/material.dart';
import 'dart:async';
import '../models/app_models.dart';
import '../services/inference_engine.dart';
import 'model_provider.dart';
import 'package:provider/provider.dart';

/// Chat template formats for different model families
enum ChatTemplate {
  chatml,     // Qwen, many instruct models
  llama,      // Llama 2/3, TinyLlama  
  alpaca,     // Alpaca-style models
  phi,        // Microsoft Phi models
  gemma,      // Google Gemma
  simple,     // Fallback simple format
}

class ChatProvider extends ChangeNotifier {
  final List<ChatMessage> _messages = [];
  bool _isGenerating = false;
  bool _lastGenerationComplete = false;
  StreamSubscription<String>? _tokenSubscription;
  String? _currentModelId;
  bool _isFirstMessage = true;  // Track if this is the first message (no cache)
  
  // OPTIMIZATION Max messages to keep in context (aggressive trimming)
  static const int _maxContextMessages = 4;  // Keep last 4 messages (2 turns)
  
  // Suggested questions for new users
  static const List<String> suggestions = [
    "What is 2+2?",
    "Say hello in 3 languages",
    "Write a haiku about coding",
    "Explain AI in one sentence",
    "Give me a fun fact",
  ];

  List<ChatMessage> get messages => List.unmodifiable(_messages);
  bool get isGenerating => _isGenerating;
  bool get hasMessages => _messages.isNotEmpty;
  bool get lastGenerationComplete => _lastGenerationComplete;
  
  /// Set the current model for template selection
  /// When model changes, we must reset the KV cache since it's model-specific
  Future<void> setCurrentModel(String? modelId, {bool clearChat = false}) async {
    if (_currentModelId != modelId) {
      // Model changed - KV cache is invalid for new model
      _currentModelId = modelId;
      _isFirstMessage = true;
      
      // Clear the KV cache since it belongs to the old model
      try {
        await InferenceEngine.clearCache();
      } catch (e) {
        print('Warning: Could not clear cache: $e');
      }
      
      if (clearChat) {
        _messages.clear();
        _lastGenerationComplete = false;
      }
      
      notifyListeners();
    }
  }
  
  /// Clear cache when starting new conversation
  Future<void> _resetConversation() async {
    await InferenceEngine.clearCache();
    _isFirstMessage = true;
  }
  
  /// Detect chat template based on model ID
  ChatTemplate _detectTemplate(String? modelId) {
    if (modelId == null) return ChatTemplate.simple;
    
    final id = modelId.toLowerCase();
    
    if (id.contains('qwen')) return ChatTemplate.chatml;
    if (id.contains('llama') || id.contains('tinyllama')) return ChatTemplate.llama;
    if (id.contains('phi')) return ChatTemplate.phi;
    if (id.contains('gemma')) return ChatTemplate.gemma;
    if (id.contains('alpaca')) return ChatTemplate.alpaca;
    if (id.contains('smol')) return ChatTemplate.chatml;  // SmolLM uses ChatML
    if (id.contains('stable')) return ChatTemplate.chatml; // StableLM uses ChatML
    
    return ChatTemplate.simple;
  }

  /// Format prompt with chat template for instruction-tuned models
  /// OPTIMIZATION 2: Chunk the prompt - system context separate from user input
  /// OPTIMIZATION 3: Trim context aggressively - only include recent messages
  String _formatPrompt(String userMessage, {bool includeHistory = false}) {
    final template = _detectTemplate(_currentModelId);
    
    // Build context from recent messages (OPTIMIZATION 3: aggressive trimming)
    String contextPrefix = '';
    if (includeHistory && _messages.length > 1) {
      // Get last N messages for context (excluding the just-added user message)
      final recentMessages = _messages
          .take((_messages.length - 1).clamp(0, _maxContextMessages))
          .toList();
      
      for (final msg in recentMessages) {
        if (msg.isUser) {
          contextPrefix += _formatUserMessage(msg.content, template);
        } else if (msg.content.isNotEmpty && 
                   !msg.content.startsWith('error') && 
                   !msg.content.startsWith('failed') &&
                   !msg.content.startsWith('model generated empty')) {
          contextPrefix += _formatAssistantMessage(msg.content, template);
        }
      }
    }
    
    // Format just the new user message
    return contextPrefix + _formatUserMessage(userMessage, template) + _getAssistantPrefix(template);
  }
  
  String _formatUserMessage(String content, ChatTemplate template) {
    switch (template) {
      case ChatTemplate.chatml:
        return '<|im_start|>user\n$content<|im_end|>\n';
      case ChatTemplate.llama:
        return '[INST] $content [/INST]';
      case ChatTemplate.phi:
        return 'Instruct: $content\n';
      case ChatTemplate.gemma:
        return '<start_of_turn>user\n$content<end_of_turn>\n';
      case ChatTemplate.alpaca:
        return '### Instruction:\n$content\n\n';
      case ChatTemplate.simple:
      default:
        return 'Q: $content\n';
    }
  }
  
  String _formatAssistantMessage(String content, ChatTemplate template) {
    switch (template) {
      case ChatTemplate.chatml:
        return '<|im_start|>assistant\n$content<|im_end|>\n';
      case ChatTemplate.llama:
        return '$content</s>';
      case ChatTemplate.phi:
        return 'Output: $content\n';
      case ChatTemplate.gemma:
        return '<start_of_turn>model\n$content<end_of_turn>\n';
      case ChatTemplate.alpaca:
        return '### Response:\n$content\n\n';
      case ChatTemplate.simple:
      default:
        return 'A: $content\n';
    }
  }
  
  String _getAssistantPrefix(ChatTemplate template) {
    switch (template) {
      case ChatTemplate.chatml:
        return '<|im_start|>assistant\n';
      case ChatTemplate.llama:
        return '';  // No prefix needed for llama
      case ChatTemplate.phi:
        return 'Output:';
      case ChatTemplate.gemma:
        return '<start_of_turn>model\n';
      case ChatTemplate.alpaca:
        return '### Response:\n';
      case ChatTemplate.simple:
      default:
        return 'A:';
    }
  }

  Future<void> sendMessage(String content) async {
    if (content.trim().isEmpty) return;
    
    // If still generating, stop the current generation first
    if (_isGenerating) {
      await stopGeneration();
    }

    // Add user message
    final userMessage = ChatMessage(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      content: content,
      isUser: true,
      timestamp: DateTime.now(),
    );
    
    _messages.add(userMessage);
    notifyListeners();

    // Start AI response generation
    await _generateAIResponse(content);
  }


  Future<void> _generateAIResponse(String userMessage) async {
    // Cancel any existing subscription first
    if (_tokenSubscription != null) {
      await _tokenSubscription?.cancel();
      _tokenSubscription = null;
    }

    // If still generating, stop it first
    if (_isGenerating) {
      await InferenceEngine.stopGeneration();
      _isGenerating = false;
    }

    _isGenerating = true;
    _lastGenerationComplete = false;
    notifyListeners();

    // Enforce strict alternation: user, AI, user, AI...
    String aiMessageId;
    if (_messages.isEmpty || _messages.last.isUser) {
      // Last is user or list is empty: add new AI placeholder
      aiMessageId = DateTime.now().millisecondsSinceEpoch.toString();
      final aiMessage = ChatMessage(
        id: aiMessageId,
        content: '',
        isUser: false,
        timestamp: DateTime.now(),
      );
      _messages.add(aiMessage);
      notifyListeners();
    } else {
      // Last is AI: update it
      aiMessageId = _messages.last.id;
    }

    try {
      // OPTIMIZATION 1 & 2: Use incremental inference for subsequent messages
      // First message: full prompt with context (clears cache)
      // Subsequent messages: only new tokens (reuses KV cache)
      bool success;
      final formattedPrompt = _formatPrompt(userMessage, includeHistory: false);
      
      if (_isFirstMessage) {
        // First message: format with full context, use regular inference
        success = await InferenceEngine.startInference(formattedPrompt);
        if (success) _isFirstMessage = false;
      } else {
        // OPTIMIZATION 1: Reuse KV cache - only send new user message
        // The model's previous response is already in the cache!
        success = await InferenceEngine.startInferenceIncremental(formattedPrompt);
        
        // Fallback: if incremental fails (cache issue), try regular inference
        if (!success) {
          // print removed
          success = await InferenceEngine.startInference(formattedPrompt);
          _isFirstMessage = false;  // Reset for next time
        }
      }
      
      if (!success) {
        _updateAIMessage(aiMessageId, 'failed to start inference. please load a model first.');
        _isGenerating = false;
        _lastGenerationComplete = true;
        notifyListeners();
        return;
      }

      // OPTIMIZATION 4: Stream tokens immediately
      StringBuffer responseBuffer = StringBuffer();
      int emptyCount = 0;
      int tokenCount = 0;
      DateTime lastUpdate = DateTime.now();
      
      // Use single token mode for first few tokens (immediate display)
      // Then switch to batch mode for efficiency
      _tokenSubscription = InferenceEngine.streamTokens().listen(
        (token) {
          if (token.isNotEmpty) {
            // Filter out special tokens that shouldn't be displayed
            String cleanToken = token
                .replaceAll('<|im_end|>', '')
                .replaceAll('<|im_start|>', '')
                .replaceAll('<|endoftext|>', '');
            
            if (cleanToken.isNotEmpty) {
              responseBuffer.write(cleanToken);
              tokenCount++;
              
              // Throttle UI updates to prevent buffer overflow
              // Update every 3 tokens or every 100ms, whichever comes first
              final now = DateTime.now();
              if (tokenCount % 3 == 0 || now.difference(lastUpdate).inMilliseconds > 100) {
                _updateAIMessage(aiMessageId, responseBuffer.toString());
                lastUpdate = now;
              }
              emptyCount = 0;
            }
          } else {
            emptyCount++;
            // If we get many empty tokens in a row, generation may have ended
            if (emptyCount > 5) {
              _tokenSubscription?.cancel();
              _tokenSubscription = null;
              // Ensure native side stops too
              InferenceEngine.stopGeneration();
              if (responseBuffer.isEmpty) {
                _updateAIMessage(aiMessageId, 'model generated empty response. try rephrasing your question.');
              } else {
                // Final update with all remaining content
                _updateAIMessage(aiMessageId, responseBuffer.toString());
              }
              _isGenerating = false;
              _lastGenerationComplete = true;
              notifyListeners();
              return; // Exit early to avoid duplicate processing
            }
          }
        },
        onDone: () {
          _tokenSubscription = null;
          if (responseBuffer.isEmpty) {
            _updateAIMessage(aiMessageId, 'model generated empty response. try rephrasing your question.');
          } else {
            // Final update with all content
            _updateAIMessage(aiMessageId, responseBuffer.toString());
          }
          _isGenerating = false;
          _lastGenerationComplete = true;
          notifyListeners();
        },
        onError: (error) {
          _tokenSubscription = null;
          _updateAIMessage(aiMessageId, 'error during generation: $error');
          _isGenerating = false;
          _lastGenerationComplete = true;
          notifyListeners();
        },
      );

    } catch (e) {
      _tokenSubscription?.cancel();
      _tokenSubscription = null;
      _updateAIMessage(aiMessageId, 'error: $e');
      _isGenerating = false;
      _lastGenerationComplete = true;
      notifyListeners();
    }
  }

  void _updateAIMessage(String messageId, String content) {
    // Only update the last AI message
    if (_messages.isNotEmpty && !_messages.last.isUser && _messages.last.id == messageId) {
      _messages[_messages.length - 1] = ChatMessage(
        id: messageId,
        content: content,
        isUser: false,
        timestamp: _messages.last.timestamp,
      );
      notifyListeners();
    }
  }

  Future<void> stopGeneration() async {
    if (_isGenerating || _tokenSubscription != null) {
      await InferenceEngine.stopGeneration();
      await _tokenSubscription?.cancel();
      _tokenSubscription = null;
      _isGenerating = false;
      _lastGenerationComplete = true;
      
      // Remove any empty AI message placeholder that was left behind
      final before = _messages.length;
      _messages.removeWhere((m) => !m.isUser && m.content.isEmpty);
      // print removed
      
      notifyListeners();
    }
  }

  Future<void> clearChat() async {
    await stopGeneration();
    await _resetConversation();  // Clear KV cache for new conversation
    _messages.clear();
    _lastGenerationComplete = false;
    notifyListeners();
  }

  @override
  void dispose() {
    _tokenSubscription?.cancel();
    _tokenSubscription = null;
    super.dispose();
  }
}