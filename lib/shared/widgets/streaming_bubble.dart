import 'package:flutter/material.dart';

/// A message bubble that shows streaming content with a cursor indicator
class StreamingBubble extends StatefulWidget {
  final String content;
  final DateTime timestamp;

  const StreamingBubble({
    super.key,
    required this.content,
    required this.timestamp,
  });

  @override
  State<StreamingBubble> createState() => _StreamingBubbleState();
}

class _StreamingBubbleState extends State<StreamingBubble>
    with SingleTickerProviderStateMixin {
  late AnimationController _cursorController;

  @override
  void initState() {
    super.initState();
    _cursorController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _cursorController.dispose();
    super.dispose();
  }

  String _formatTime(DateTime dateTime) {
    return '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final hasContent = widget.content.isNotEmpty;

    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(16),
        ),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (hasContent)
              // Show content with blinking cursor
              Row(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Flexible(
                    child: Text(
                      widget.content,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                  // Blinking cursor
                  FadeTransition(
                    opacity: _cursorController,
                    child: Container(
                      width: 2,
                      height: 16,
                      margin: const EdgeInsets.only(left: 2),
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                ],
              )
            else
              // Show skeleton loading dots when no content yet
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildLoadingDot(context, 0),
                  const SizedBox(width: 4),
                  _buildLoadingDot(context, 1),
                  const SizedBox(width: 4),
                  _buildLoadingDot(context, 2),
                ],
              ),
            const SizedBox(height: 4),
            Text(
              _formatTime(widget.timestamp),
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Theme.of(context)
                        .colorScheme
                        .onSurfaceVariant
                        .withOpacity(0.6),
                  ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingDot(BuildContext context, int index) {
    return AnimatedBuilder(
      animation: _cursorController,
      builder: (context, child) {
        // Staggered animation for each dot
        final phase = (_cursorController.value + (index * 0.3)) % 1.0;
        final scale = 0.5 + (phase * 0.5);
        return Transform.scale(
          scale: scale,
          child: Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: Theme.of(context)
                  .colorScheme
                  .primary
                  .withOpacity(0.4 + (phase * 0.4)),
              shape: BoxShape.circle,
            ),
          ),
        );
      },
    );
  }
}
