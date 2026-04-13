#ifndef OPEN_WAKE_WORD_H
#define OPEN_WAKE_WORD_H

#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
#define OWW_EXPORT __declspec(dllexport)
#elif defined(__APPLE__)
#define OWW_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#else
#define OWW_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the OpenWakeWord engine with the given model paths.
// Returns 0 on success, non-zero on error.
OWW_EXPORT int oww_init(const char* mel_model_path, const char* emb_model_path, const char* ww_model_path);

// Process a chunk of 16kHz PCM audio data.
// Length is the number of int16_t samples (not bytes).
// Note: Chunk size should ideally match the engine's internal step frame size (e.g., 1280 samples = 80ms)
OWW_EXPORT void oww_process_audio(const int16_t* audio_data, int length);

// Get the latest probability for the wake word model.
OWW_EXPORT float oww_get_probability();

// Return the current boolean activation state (threshold triggered).
OWW_EXPORT bool oww_is_activated();

// Clean up and release the engine resources.
OWW_EXPORT void oww_destroy();

#ifdef __cplusplus
}
#endif

#endif // OPEN_WAKE_WORD_H
