import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../shared/theme/app_theme.dart';
import '../providers/model_provider.dart';
import '../providers/chat_provider.dart';
import 'app_navigation.dart';

class ModelRunnerApp extends StatelessWidget {
  const ModelRunnerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider<ModelProvider>(
          create: (_) => ModelProvider(),
        ),
        ChangeNotifierProvider<ChatProvider>(
          create: (_) => ChatProvider(),
        ),
      ],
      child: MaterialApp(
        title: 'Model Runner',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeMode.system,
        home: const AppNavigation(),
      ),
    );
  }
}