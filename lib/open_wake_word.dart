import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'open_wake_word_bindings_generated.dart';
import 'package:ffi/ffi.dart';

const String _libName = 'open_wake_word';

final DynamicLibrary _dylib = () {
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.process();
  }
  if (Platform.isAndroid || Platform.isLinux) {
    try {
      return DynamicLibrary.open('lib$_libName.so');
    } catch (e) {
      // Fallback for some android environments
      return DynamicLibrary.process();
    }
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

final OpenWakeWordBindings _bindings = OpenWakeWordBindings(_dylib);

class OpenWakeWord {
  /// Initialize the OpenWakeWord engine with the given models.
  /// Ensure the models correctly exist in your flutter asset bundle.
  static Future<bool> init({
    required String melModelAssetPath,
    required String embModelAssetPath,
    required String wwModelAssetPath,
  }) async {
    try {
      final melPath = await _extractAsset(melModelAssetPath);
      final embPath = await _extractAsset(embModelAssetPath);
      final wwPath = await _extractAsset(wwModelAssetPath);

      final melPtr = melPath.toNativeUtf8();
      final embPtr = embPath.toNativeUtf8();
      final wwPtr = wwPath.toNativeUtf8();

      final result = _bindings.oww_init(
        melPtr.cast<Char>(),
        embPtr.cast<Char>(),
        wwPtr.cast<Char>(),
      );

      calloc.free(melPtr);
      calloc.free(embPtr);
      calloc.free(wwPtr);

      return result == 0;
    } catch (e) {
      print('Failed to init OpenWakeWord: $e');
      return false;
    }
  }

  /// Process a chunk of audio (should be 16kHz PCM, optimally 1280 samples / 80ms)
  static void processAudio(Int16List audioData) {
    final ptr = calloc<Int16>(audioData.length);
    final nativeList = ptr.asTypedList(audioData.length);
    nativeList.setAll(0, audioData);
    
    _bindings.oww_process_audio(ptr, audioData.length);
    calloc.free(ptr);
  }

  /// Get the latest wake word probability.
  static double getProbability() {
    return _bindings.oww_get_probability();
  }

  /// Returns true if the wake word threshold has been crossed (activation state).
  static bool isActivated() {
    return _bindings.oww_is_activated();
  }

  /// Free all resources consumed by the engine.
  static void destroy() {
    _bindings.oww_destroy();
  }

  static Future<String> _extractAsset(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    final Directory dir = await getApplicationDocumentsDirectory();
    final String filename = assetPath.split('/').last;
    final File file = File('${dir.path}/$filename');

    if (!await file.exists()) {
      await file.writeAsBytes(bytes, flush: true);
    }
    return file.path;
  }
}
