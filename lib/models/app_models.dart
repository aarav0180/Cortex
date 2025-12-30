class ChatMessage {
  final String id;
  final String content;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.id,
    required this.content,
    required this.isUser,
    required this.timestamp,
  });
}

class Model {
  final String id;
  final String name;
  final String description;
  final double sizeGB;
  final ModelStatus status;
  final double downloadProgress;
  final String? downloadUrl;
  final String? localPath;

  Model({
    required this.id,
    required this.name,
    required this.description,
    required this.sizeGB,
    this.status = ModelStatus.notDownloaded,
    this.downloadProgress = 0.0,
    this.downloadUrl,
    this.localPath,
  });

  Model copyWith({
    String? id,
    String? name,
    String? description,
    double? sizeGB,
    ModelStatus? status,
    double? downloadProgress,
    String? downloadUrl,
    String? localPath,
  }) {
    return Model(
      id: id ?? this.id,
      name: name ?? this.name,
      description: description ?? this.description,
      sizeGB: sizeGB ?? this.sizeGB,
      status: status ?? this.status,
      downloadProgress: downloadProgress ?? this.downloadProgress,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      localPath: localPath ?? this.localPath,
    );
  }
}

enum ModelStatus { 
  notDownloaded, 
  downloading, 
  downloaded, 
  error 
}