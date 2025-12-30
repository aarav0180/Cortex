samples, guidance on mobile development, and a full API reference.

# Cortex2

Cortex2 is a fast, efficient AI model runner for mobile devices, built with Flutter. It allows you to run local AI models, chat with them, and experiment with different model families—all optimized for performance and usability.

## Features

| Feature | Description |
|---------|-------------|
| Local Model Runner | Run AI models directly on your device, no cloud required |
| Chat Interface | Clean, modern chat UI for interacting with models |
| Model Family Support | Supports multiple chat template formats (Qwen, Llama, Alpaca, Phi, Gemma, and more) |
| Streaming Responses | See model responses as they are generated, with incremental token streaming |
| Model Switching | Easily switch between models; context and cache are managed automatically |
| Suggestions | Quick-start suggested prompts for new users |
| Mobile Optimized | Designed for low memory usage and fast inference on mobile hardware |

## Optimizations

| Optimization | Details |
|--------------|---------|
| Aggressive Context Trimming | Only the last 4 messages (2 turns) are kept in context to minimize memory and speed up inference |
| KV Cache Management | When switching models, the key-value cache is cleared to avoid cross-model contamination |
| Incremental Inference | After the first message, only new user input is sent to the model, reusing cached context for faster responses |
| Streaming Token Updates | UI updates every few tokens or every 100ms for smooth, real-time feedback |
| Placeholder Management | Ensures strict alternation between user and AI messages, with only one AI placeholder per turn |

## Architecture

- **Flutter Frontend:** Handles chat UI, message management, and user interaction.
- **Provider State Management:** Uses Provider for clean separation of chat/model logic and UI.
- **FFI Backend:** Communicates with native inference engine via Dart FFI for high performance.
- **Extensible Model Support:** Easily add new model templates and formats.

## File Structure

- `lib/main.dart` — App entry point.
- `lib/pages/chat_page.dart` — Main chat UI.
- `lib/providers/chat_provider.dart` — Chat logic, message management, optimizations.
- `lib/providers/model_provider.dart` — Model selection and management(also the available models).
- `lib/services/inference_engine.dart` — FFI bridge to native inference.

## Getting Started

1. **Install Flutter:**  
	[Flutter installation guide](https://docs.flutter.dev/get-started/install)

2. **Clone the repository:**  
	```
	git clone <your-repo-url>
	cd cortex2
	```

3. **Install dependencies:**  
	```
	flutter pub get
	```

4. **Run the app:**  
	```
	flutter run
	```

5. **Add Models:**  
	Place your supported model files in the appropriate directory as described in `lib/providers/model_provider.dart`.

## Usage

- Select a model from the Models tab.
- Start chatting using the suggestions or your own prompts.
- Stop or clear chat at any time.
- Switch models as needed; context and cache are managed automatically.

## Supported Model Templates

- **Qwen (ChatML)**
- **Llama (Llama 2/3, TinyLlama)**
- **Alpaca**
- **Phi**
- **Gemma**
- **Simple (Fallback)**

## Contributing

Pull requests and issues are welcome! Please follow standard Flutter/Dart best practices.

## License

This project is licensed under the MIT License.
