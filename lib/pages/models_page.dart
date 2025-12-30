import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/model_provider.dart';
import '../providers/chat_provider.dart';
import '../models/app_models.dart';
import '../shared/widgets/empty_state.dart';

class ModelsPage extends StatefulWidget {
  const ModelsPage({super.key});

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  @override
  void initState() {
    super.initState();
    // Check for existing downloaded models
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ModelProvider>().checkDownloadedModels();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ModelProvider>(
      builder: (context, provider, _) {
        return Scaffold(
          appBar: AppBar(
            title: const Text('Available Models'),
            actions: [
              // Memory usage indicator
              if (provider.memoryUsageMB > 0)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Chip(
                    label: Text('${provider.memoryUsageMB}MB'),
                    backgroundColor: Theme.of(context).colorScheme.surfaceContainer,
                    labelStyle: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              // Currently loaded model
              if (provider.hasSelectedModel)
                Padding(
                  padding: const EdgeInsets.only(right: 16),
                  child: Chip(
                    label: Text('🟢 ${provider.selectedModel!.name}'),
                    backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                  ),
                ),
            ],
          ),
          body: _buildBody(context, provider),
        );
      },
    );
  }

  Widget _buildBody(BuildContext context, ModelProvider provider) {
    if (provider.models.isEmpty) {
      return const EmptyState(
        icon: Icons.download_outlined,
        title: 'No Models Available',
        subtitle: 'No models are currently available for download',
      );
    }
    return _buildModelsList(context, provider);
  }

  Widget _buildModelsList(BuildContext context, ModelProvider provider) {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: provider.models.length,
      itemBuilder: (context, index) {
        final model = provider.models[index];
        final isSelected = provider.selectedModel?.id == model.id;

        return Card(
          child: ListTile(
            leading: _buildLeadingIcon(context, model, isSelected),
            title: Text(model.name),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${model.description} • ${model.sizeGB} GB'),
                if (model.status == ModelStatus.downloading) ...[
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: model.downloadProgress,
                    backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${(model.downloadProgress * 100).toInt()}%',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ],
            ),
            trailing: _buildTrailingWidget(context, model, provider),
            onTap: model.status == ModelStatus.downloaded 
                ? () => _onModelTap(context, model, provider)
                : null,
          ),
        );
      },
    );
  }

  Widget _buildLeadingIcon(BuildContext context, Model model, bool isSelected) {
    IconData iconData;
    Color? iconColor;

    switch (model.status) {
      case ModelStatus.downloaded:
        iconData = isSelected ? Icons.check_circle : Icons.memory;
        iconColor = isSelected ? Theme.of(context).colorScheme.primary : null;
        break;
      case ModelStatus.downloading:
        iconData = Icons.download;
        iconColor = Theme.of(context).colorScheme.primary;
        break;
      case ModelStatus.error:
        iconData = Icons.error;
        iconColor = Theme.of(context).colorScheme.error;
        break;
      case ModelStatus.notDownloaded:
        iconData = Icons.download_outlined;
        break;
    }

    return Icon(iconData, color: iconColor);
  }

  Widget _buildTrailingWidget(BuildContext context, Model model, ModelProvider provider) {
    switch (model.status) {
      case ModelStatus.downloaded:
        bool isLoaded = provider.selectedModel?.id == model.id;
        
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (!isLoaded)
              IconButton(
                onPressed: () => _loadModel(model, provider),
                icon: const Icon(Icons.play_arrow),
                tooltip: 'Load model',
              )
            else
              IconButton(
                onPressed: () => _unloadModel(provider),
                icon: const Icon(Icons.stop),
                tooltip: 'Unload model',
              ),
            IconButton(
              onPressed: () => _showDeleteDialog(context, model, provider),
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Delete model',
            ),
          ],
        );
      case ModelStatus.downloading:
        return const SizedBox(
          width: 20,
          height: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        );
      case ModelStatus.error:
        return IconButton(
          onPressed: () => provider.downloadModel(model.id),
          icon: const Icon(Icons.refresh),
          tooltip: 'Retry download',
        );
      case ModelStatus.notDownloaded:
        return IconButton(
          onPressed: () => provider.downloadModel(model.id),
          icon: const Icon(Icons.download),
          tooltip: 'Download model',
        );
    }
  }

  void _onModelTap(BuildContext context, Model model, ModelProvider provider) {
    // Model tap can load the model if it's downloaded
    if (model.status == ModelStatus.downloaded) {
      _loadModel(model, provider);
    }
  }

  void _loadModel(Model model, ModelProvider provider) async {
    final scaffold = ScaffoldMessenger.of(context);
    
    scaffold.showSnackBar(
      SnackBar(content: Text('Loading ${model.name}...')),
    );

    final success = await provider.loadModel(model.id);
    
    if (success) {
      // Update chat provider with current model ID for template detection
      // Clear chat when switching models since context is model-specific
      if (mounted) {
        await context.read<ChatProvider>().setCurrentModel(model.id, clearChat: true);
      }
      
      scaffold.showSnackBar(
        SnackBar(content: Text('${model.name} loaded')),
      );
    } else {
      scaffold.showSnackBar(
        SnackBar(
          content: Text('failed to load ${model.name}'),
          backgroundColor: Theme.of(context).colorScheme.error,
        ),
      );
    }
  }

  void _unloadModel(ModelProvider provider) async {
    await provider.unloadModel();
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model unloaded')),
      );
    }
  }

  void _showDeleteDialog(BuildContext context, Model model, ModelProvider provider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Model'),
        content: Text('Are you sure you want to delete ${model.name}?\n\nThis will free up ${model.sizeGB.toStringAsFixed(1)} GB of storage.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.of(context).pop();
              await provider.deleteModel(model.id);
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Deleted ${model.name}')),
                );
              }
            },
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}