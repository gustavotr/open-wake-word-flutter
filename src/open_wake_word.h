#ifndef OPEN_WAKE_WORD_H
#define OPEN_WAKE_WORD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the OpenWakeWord engine with the given model paths.
// Returns 0 on success, non-zero on error.
int oww_init(const char* mel_model_path, const char* emb_model_path, const char* ww_model_path);

// Process a chunk of 16kHz PCM audio data.
// Length is the number of int16_t samples (not bytes).
// Note: Chunk size should ideally match the engine's internal step frame size (e.g., 1280 samples = 80ms)
void oww_process_audio(const int16_t* audio_data, int length);

// Get the latest probability for the wake word model.
float oww_get_probability();

// Return the current boolean activation state (threshold triggered).
bool oww_is_activated();

// Clean up and release the engine resources.
void oww_destroy();

#ifdef __cplusplus
}
#endif

#endif // OPEN_WAKE_WORD_H
