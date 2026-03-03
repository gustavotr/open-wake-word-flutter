# open_wake_word

A Flutter FFI plugin for [openWakeWord](https://github.com/dscripka/openWakeWord), providing efficient on-device wake word detection using ONNX Runtime.

## Features

- **On-device Detection**: All processing happens locally on the device for privacy and speed.
- **High Accuracy**: Uses the power of openWakeWord's models.
- **Cross-Platform**: Supports Android and iOS (with support for more platforms in progress).
- **Fast Execution**: Implemented via Dart FFI for near-native performance.

## Getting Started

### Prerequisites

You will need the following ONNX models from the [openWakeWord](https://github.com/dscripka/openWakeWord) project:
1. `melspectrogram.onnx`
2. `embedding_model.onnx`
3. A wake word model (e.g., `hey_jarvis.onnx`)

### Installation

Add `open_wake_word` to your `pubspec.yaml`:

```yaml
dependencies:
  open_wake_word: ^0.1.0
```

### Asset Setup

Include the ONNX models in your `pubspec.yaml` assets:

```yaml
flutter:
  assets:
    - assets/models/melspectrogram.onnx
    - assets/models/embedding_model.onnx
    - assets/models/hey_jarvis.onnx
```

## Usage

### Initialization

Initialize the engine by providing the asset paths to your models:

```dart
import 'package:open_wake_word/open_wake_word.dart';

bool success = await OpenWakeWord.init(
  melModelAssetPath: 'assets/models/melspectrogram.onnx',
  embModelAssetPath: 'assets/models/embedding_model.onnx',
  wwModelAssetPath: 'assets/models/hey_jarvis.onnx',
);
```

### Processing Audio

Feed 16kHz PCM audio data (int16) to the engine. For optimal results, use chunks of 1280 samples (80ms).

```dart
OpenWakeWord.processAudio(audioData);

if (OpenWakeWord.isActivated()) {
  print("Wake word detected!");
}

double probability = OpenWakeWord.getProbability();
print("Current probability: $probability");
```

### Cleanup

Don't forget to destroy the engine when it's no longer needed:

```dart
OpenWakeWord.destroy();
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
It uses models and concepts from the original [openWakeWord](https://github.com/dscripka/openWakeWord) project by Kevin Ahrens.
