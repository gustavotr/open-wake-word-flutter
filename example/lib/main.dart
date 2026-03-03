import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:open_wake_word/open_wake_word.dart';
import 'package:record/record.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isInitialized = false;
  bool _isListening = false;
  bool _isActivated = false;
  double _probability = 0.0;

  final AudioRecorder _audioRecorder = AudioRecorder();
  StreamSubscription<Uint8List>? _audioStreamSubscription;
  Timer? _pollingTimer;

  @override
  void initState() {
    super.initState();
    _initEngine();
  }

  Future<void> _initEngine() async {
    final success = await OpenWakeWord.init(
      melModelAssetPath: 'assets/models/melspectrogram.onnx',
      embModelAssetPath: 'assets/models/embedding_model.onnx',
      wwModelAssetPath: 'assets/models/hey_jarvis_v0.1.onnx',
    );

    setState(() {
      _isInitialized = success;
    });

    if (success) {
      _startPolling();
    }
  }

  void _startPolling() {
    _pollingTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (!_isInitialized) return;

      final prob = OpenWakeWord.getProbability();
      final activated = OpenWakeWord.isActivated();

      if (prob != _probability || activated != _isActivated) {
        setState(() {
          _probability = prob;
          _isActivated = activated;
        });
      }
    });
  }

  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if (await _audioRecorder.hasPermission()) {
      // record 6.x: startStream returns Stream<Uint8List>
      final stream = await _audioRecorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ));

      _audioStreamSubscription = stream.listen((Uint8List bytes) {
        // Convert raw PCM bytes (little-endian int16) to Int16List
        final int16List = Int16List(bytes.length ~/ 2);
        for (int i = 0; i < int16List.length; i++) {
          int16List[i] = (bytes[i * 2] & 0xff) | ((bytes[i * 2 + 1] & 0xff) << 8);
          // Treat as signed 16-bit:
          if (int16List[i] >= 0x8000) {
            int16List[i] = int16List[i] - 0x10000;
          }
        }

        OpenWakeWord.processAudio(int16List);
      });

      setState(() {
        _isListening = true;
      });
    }
  }

  Future<void> _stopListening() async {
    await _audioStreamSubscription?.cancel();
    _audioStreamSubscription = null;
    await _audioRecorder.stop();
    setState(() {
      _isListening = false;
    });
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    _audioStreamSubscription?.cancel();
    _audioRecorder.dispose();
    OpenWakeWord.destroy();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OpenWakeWord FFI Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: Scaffold(
        appBar: AppBar(
          title: const Text('OpenWakeWord FFI Demo'),
          backgroundColor: Colors.deepPurple,
          foregroundColor: Colors.white,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                _isInitialized ? Icons.check_circle : Icons.error_outline,
                color: _isInitialized ? Colors.green : Colors.red,
                size: 40,
              ),
              const SizedBox(height: 8),
              Text(
                _isInitialized ? 'Engine ready' : 'Engine not initialized',
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _isInitialized ? _toggleListening : null,
                icon: Icon(_isListening ? Icons.mic_off : Icons.mic),
                label: Text(_isListening ? 'Stop Listening' : 'Start Listening'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                ),
              ),
              const SizedBox(height: 48),
              AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 200,
                height: 200,
                decoration: BoxDecoration(
                  color: _isActivated ? Colors.green : Colors.grey.shade300,
                  shape: BoxShape.circle,
                  boxShadow: _isActivated
                      ? [BoxShadow(color: Colors.green.withOpacity(0.5), blurRadius: 30, spreadRadius: 10)]
                      : [],
                ),
                child: Center(
                  child: Text(
                    _isActivated ? '✓' : 'Listening...',
                    style: TextStyle(
                      color: _isActivated ? Colors.white : Colors.grey,
                      fontSize: _isActivated ? 48 : 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Probability: ${_probability.toStringAsFixed(4)}',
                style: const TextStyle(fontSize: 18, color: Colors.black54),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
