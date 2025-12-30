/// Chat template formats for different model families
enum ChatTemplate {
  chatml,     // Qwen, many instruct models
  llama,      // Llama 2/3, TinyLlama  
  alpaca,     // Alpaca-style models
  phi,        // Microsoft Phi models
  gemma,      // Google Gemma
  simple,     // Fallback simple format
}

/// Utility class for formatting prompts with chat templates
class ChatTemplateFormatter {
  /// Detect chat template based on model ID
  static ChatTemplate detectTemplate(String? modelId) {
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

  /// Format a user message with the appropriate template
  static String formatUserMessage(String content, ChatTemplate template) {
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
        return 'Q: $content\n';
    }
  }
  
  /// Format an assistant message with the appropriate template
  static String formatAssistantMessage(String content, ChatTemplate template) {
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
        return 'A: $content\n';
    }
  }
  
  /// Get the assistant prefix (what comes before assistant's response)
  static String getAssistantPrefix(ChatTemplate template) {
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
        return 'A:';
    }
  }

  /// Format a simple prompt (user message + assistant prefix) for benchmarking
  static String formatSimplePrompt(String userMessage, String? modelId) {
    final template = detectTemplate(modelId);
    return formatUserMessage(userMessage, template) + getAssistantPrefix(template);
  }
}
