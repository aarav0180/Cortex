import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/chat_provider.dart';
import '../providers/model_provider.dart';
import '../shared/widgets/empty_state.dart';
import '../shared/widgets/message_bubble.dart';
import '../shared/widgets/typing_indicator.dart';
import '../shared/widgets/streaming_bubble.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer2<ChatProvider, ModelProvider>(
      builder: (context, chatProvider, modelProvider, _) {
        return Scaffold(
          appBar: AppBar(
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Chat'),
                Text(
                  modelProvider.selectedModel?.name ?? 'No model selected',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ],
            ),
            actions: [
              if (chatProvider.hasMessages)
                IconButton(
                  onPressed: chatProvider.clearChat,
                  icon: const Icon(Icons.clear_all),
                  tooltip: 'Clear chat',
                ),
            ],
          ),
          body: modelProvider.selectedModel == null
              ? EmptyState(
                  icon: Icons.smart_toy_outlined,
                  title: 'No Model Selected',
                  subtitle: 'Select a model from the Models tab to start chatting',
                  action: FilledButton(
                    onPressed: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Switch to Models tab to select a model'),
                        ),
                      );
                    },
                    child: const Text('Select Model'),
                  ),
                )
              : _buildChatInterface(context, chatProvider),
        );
      },
    );
  }

  Widget _buildChatInterface(BuildContext context, ChatProvider chatProvider) {
    _scrollToBottom();

    return Column(
      children: [
        Expanded(
          child: chatProvider.hasMessages
              ? _buildMessagesList(chatProvider)
              : _buildSuggestions(context, chatProvider),
        ),
        // Generation status indicator
        if (chatProvider.isGenerating || chatProvider.lastGenerationComplete)
          _buildStatusBar(context, chatProvider),
        _buildInputArea(context, chatProvider),
      ],
    );
  }

  Widget _buildStatusBar(BuildContext context, ChatProvider chatProvider) {
    final isGenerating = chatProvider.isGenerating;
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        border: Border(
          top: BorderSide(
            color: Theme.of(context).dividerColor.withOpacity(0.1),
          ),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (isGenerating) ...[
            SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              'Generating...',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          ] else ...[
            Icon(
              Icons.check_circle,
              size: 14,
              color: Theme.of(context).colorScheme.primary.withOpacity(0.7),
            ),
            const SizedBox(width: 6),
            Text(
              'Completed',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSuggestions(BuildContext context, ChatProvider chatProvider) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 32),
          Center(
            child: Icon(
              Icons.auto_awesome,
              size: 64,
              color: Theme.of(context).colorScheme.primary.withOpacity(0.5),
            ),
          ),
          const SizedBox(height: 16),
          Center(
            child: Text(
              'What would you like to know?',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
          ),
          const SizedBox(height: 8),
          Center(
            child: Text(
              'Try one of these suggestions or ask anything',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
              ),
            ),
          ),
          const SizedBox(height: 32),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children: ChatProvider.suggestions.map((suggestion) {
              return _buildSuggestionChip(context, chatProvider, suggestion);
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildSuggestionChip(BuildContext context, ChatProvider chatProvider, String suggestion) {
    return ActionChip(
      label: Text(
        suggestion,
        style: Theme.of(context).textTheme.bodySmall,
      ),
      avatar: Icon(
        Icons.lightbulb_outline,
        size: 16,
        color: Theme.of(context).colorScheme.primary,
      ),
      backgroundColor: Theme.of(context).colorScheme.primaryContainer.withOpacity(0.3),
      side: BorderSide(
        color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
      ),
      onPressed: chatProvider.isGenerating ? null : () {
        chatProvider.sendMessage(suggestion);
      },
    );
  }

  Widget _buildMessagesList(ChatProvider chatProvider) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.all(16),
      itemCount: chatProvider.messages.length,
      itemBuilder: (context, index) {
        final message = chatProvider.messages[index];
        final isLastMessage = index == chatProvider.messages.length - 1;
        
        // Show streaming bubble for the last AI message during generation
        if (!message.isUser && isLastMessage && chatProvider.isGenerating) {
          // Show typing indicator if content is empty, otherwise show streaming content
          if (message.content.isEmpty) {
            return const TypingIndicator();
          }
          return StreamingBubble(
            content: message.content,
            timestamp: message.timestamp,
          );
        }
        
        // For non-last messages or completed messages, skip empty AI messages
        if (!message.isUser && message.content.isEmpty) {
          return const SizedBox.shrink();  // Hide empty AI messages that aren't actively generating
        }
        
        return MessageBubble(
          message: message.content,
          isUser: message.isUser,
          timestamp: message.timestamp,
        );
      },
    );
  }

  Widget _buildInputArea(BuildContext context, ChatProvider chatProvider) {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: const InputDecoration(
                hintText: 'Type your message...',
              ),
              maxLines: null,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => _sendMessage(chatProvider),
              enabled: !chatProvider.isGenerating,
            ),
          ),
          const SizedBox(width: 8),
          if (chatProvider.isGenerating)
            IconButton.filled(
              onPressed: () => chatProvider.stopGeneration(),
              style: IconButton.styleFrom(
                backgroundColor: Theme.of(context).colorScheme.error,
              ),
              icon: const Icon(Icons.stop),
              tooltip: 'Stop generating',
            )
          else
            IconButton.filled(
              onPressed: () => _sendMessage(chatProvider),
              icon: const Icon(Icons.send),
            ),
        ],
      ),
    );
  }

  void _sendMessage(ChatProvider chatProvider) {
    final text = _controller.text.trim();
    if (text.isEmpty || chatProvider.isGenerating) return;

    chatProvider.sendMessage(text);
    _controller.clear();
  }
}