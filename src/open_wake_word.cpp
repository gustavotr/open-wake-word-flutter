#include "open_wake_word.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include <onnxruntime_cxx_api.h>

using namespace std;

// Model settings constants
const string instanceName = "openWakeWordFFI";
const size_t chunkSamples = 1280; // 80 ms
const size_t numMels = 32;
const size_t embWindowSize = 76; // 775 ms
const size_t embStepSize = 8;    // 80 ms
const size_t embFeatures = 96;
const size_t wwFeatures = 16;
const size_t frameSize = 1280;

struct Settings {
  string melModelPath;
  string embModelPath;
  string wwModelPath;

  float threshold = 0.5f;
  Ort::SessionOptions options;
};

struct EngineState {
  Ort::Env env;

  mutex mutFeatures;
  condition_variable cvFeatures;
  bool featuresExhausted = false;
  bool featuresReady = false;

  bool samplesExhausted = false;
  bool melsExhausted = false;
  bool samplesReady = false;
  bool melsReady = false;

  mutex mutSamples, mutMels, mutReady, mutOutput;
  condition_variable cvSamples, cvMels, cvReady;
  
  size_t numReady = 0;

  std::atomic<float> latestProbability{0.0f};
  std::atomic<bool> isActivated{false};

  EngineState() {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    env.DisableTelemetryEvents();
  }
};

static Settings* g_settings = nullptr;
static EngineState* g_state = nullptr;

static vector<float> g_floatSamples;
static vector<float> g_mels;
static vector<float> g_features;

static thread* g_melThread = nullptr;
static thread* g_featuresThread = nullptr;
static thread* g_wwThread = nullptr;

// Thread 1: Audio -> Mels
void audioToMels() {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto melSession = Ort::Session(g_state->env,
#ifdef _WIN32
    std::wstring(g_settings->melModelPath.begin(), g_settings->melModelPath.end()).c_str(),
#else
    g_settings->melModelPath.c_str(),
#endif
  g_settings->options);

  vector<int64_t> samplesShape{1, (int64_t)frameSize};

  auto melInputName = melSession.GetInputNameAllocated(0, allocator);
  vector<const char *> melInputNames{melInputName.get()};

  auto melOutputName = melSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> melOutputNames{melOutputName.get()};

  vector<float> todoSamples;

  {
    unique_lock<mutex> lockReady(g_state->mutReady);
    g_state->numReady += 1;
    g_state->cvReady.notify_one();
  }

  Ort::RunOptions runOptions{nullptr};

  while (true) {
    {
      unique_lock<mutex> lockSamples{g_state->mutSamples};
      g_state->cvSamples.wait(lockSamples, [] { return g_state->samplesReady; });
      if (g_state->samplesExhausted && g_floatSamples.empty()) {
        break;
      }
      copy(g_floatSamples.begin(), g_floatSamples.end(), back_inserter(todoSamples));
      g_floatSamples.clear();

      if (!g_state->samplesExhausted) {
        g_state->samplesReady = false;
      }
    }

    while (todoSamples.size() >= frameSize) {
      Ort::Value melInputTensor = Ort::Value::CreateTensor<float>(
          memoryInfo, todoSamples.data(), frameSize,
          samplesShape.data(), samplesShape.size());

      auto melOutputTensors =
          melSession.Run(runOptions, melInputNames.data(),
                         &melInputTensor, 1,
                         melOutputNames.data(), melOutputNames.size());

      const auto &melOut = melOutputTensors.front();
      const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
      const auto melShape = melInfo.GetShape();

      const float *melData = melOut.GetTensorData<float>();
      size_t melCount = accumulate(melShape.begin(), melShape.end(), 1, multiplies<size_t>());

      {
        unique_lock<mutex> lockMels{g_state->mutMels};
        g_mels.reserve(g_mels.size() + melCount);
        for (size_t i = 0; i < melCount; i++) {
          g_mels.push_back((melData[i] / 10.0f) + 2.0f);
        }
        g_state->melsReady = true;
        g_state->cvMels.notify_one();
      }

      todoSamples.erase(todoSamples.begin(), todoSamples.begin() + frameSize);
    }
  }
}

// Thread 2: Mels -> Features
void melsToFeatures() {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto embSession = Ort::Session(g_state->env, 
#ifdef _WIN32
    std::wstring(g_settings->embModelPath.begin(), g_settings->embModelPath.end()).c_str(),
#else
    g_settings->embModelPath.c_str(), 
#endif
  g_settings->options);

  vector<int64_t> embShape{1, (int64_t)embWindowSize, (int64_t)numMels, 1};

  auto embInputName = embSession.GetInputNameAllocated(0, allocator);
  vector<const char *> embInputNames{embInputName.get()};

  auto embOutputName = embSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> embOutputNames{embOutputName.get()};

  vector<float> todoMels;
  size_t melFrames = 0;

  {
    unique_lock<mutex> lockReady(g_state->mutReady);
    g_state->numReady += 1;
    g_state->cvReady.notify_one();
  }

  Ort::RunOptions runOptions{nullptr};

  while (true) {
    {
      unique_lock<mutex> lockMels{g_state->mutMels};
      g_state->cvMels.wait(lockMels, [] { return g_state->melsReady; });
      if (g_state->melsExhausted && g_mels.empty()) {
        break;
      }
      copy(g_mels.begin(), g_mels.end(), back_inserter(todoMels));
      g_mels.clear();

      if (!g_state->melsExhausted) {
        g_state->melsReady = false;
      }
    }

    melFrames = todoMels.size() / numMels;
    while (melFrames >= embWindowSize) {
      Ort::Value embInputTensor = Ort::Value::CreateTensor<float>(
          memoryInfo, todoMels.data(), embWindowSize * numMels, embShape.data(),
          embShape.size());

      auto embOutputTensors =
          embSession.Run(runOptions, embInputNames.data(),
                         &embInputTensor, 1,
                         embOutputNames.data(), embOutputNames.size());

      const auto &embOut = embOutputTensors.front();
      const auto embOutInfo = embOut.GetTensorTypeAndShapeInfo();
      const auto embOutShape = embOutInfo.GetShape();

      const float *embOutData = embOut.GetTensorData<float>();
      size_t embOutCount = accumulate(embOutShape.begin(), embOutShape.end(), 1, multiplies<size_t>());

      {
        unique_lock<mutex> lockFeatures{g_state->mutFeatures};
        g_features.reserve(g_features.size() + embOutCount);
        copy(embOutData, embOutData + embOutCount, back_inserter(g_features));
        g_state->featuresReady = true;
        g_state->cvFeatures.notify_one();
      }

      todoMels.erase(todoMels.begin(), todoMels.begin() + (embStepSize * numMels));
      melFrames = todoMels.size() / numMels;
    }
  }
}

// Thread 3: Features -> Output Probability
void featuresToOutput() {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto wwSession = Ort::Session(g_state->env, 
#ifdef _WIN32
    std::wstring(g_settings->wwModelPath.begin(), g_settings->wwModelPath.end()).c_str(),
#else
    g_settings->wwModelPath.c_str(), 
#endif
  g_settings->options);

  vector<int64_t> wwShape{1, (int64_t)wwFeatures, (int64_t)embFeatures};

  auto wwInputName = wwSession.GetInputNameAllocated(0, allocator);
  vector<const char *> wwInputNames{wwInputName.get()};

  auto wwOutputName = wwSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> wwOutputNames{wwOutputName.get()};

  vector<float> todoFeatures;
  size_t numBufferedFeatures = 0;

  {
    unique_lock<mutex> lockReady(g_state->mutReady);
    g_state->numReady += 1;
    g_state->cvReady.notify_one();
  }

  Ort::RunOptions runOptions{nullptr};

  while (true) {
    {
      unique_lock<mutex> lockFeatures{g_state->mutFeatures};
      g_state->cvFeatures.wait(lockFeatures, [] { return g_state->featuresReady; });
      if (g_state->featuresExhausted && g_features.empty()) {
        break;
      }
      copy(g_features.begin(), g_features.end(), back_inserter(todoFeatures));
      g_features.clear();

      if (!g_state->featuresExhausted) {
        g_state->featuresReady = false;
      }
    }

    numBufferedFeatures = todoFeatures.size() / embFeatures;
    while (numBufferedFeatures >= wwFeatures) {
      Ort::Value wwInputTensor = Ort::Value::CreateTensor<float>(
          memoryInfo, todoFeatures.data(), wwFeatures * embFeatures,
          wwShape.data(), wwShape.size());

      auto wwOutputTensors =
          wwSession.Run(runOptions, wwInputNames.data(),
                        &wwInputTensor, 1, wwOutputNames.data(), 1);

      const auto &wwOut = wwOutputTensors.front();
      const auto wwOutInfo = wwOut.GetTensorTypeAndShapeInfo();
      const auto wwOutShape = wwOutInfo.GetShape();
      const float *wwOutData = wwOut.GetTensorData<float>();
      size_t wwOutCount = accumulate(wwOutShape.begin(), wwOutShape.end(), 1, multiplies<size_t>());

      for (size_t i = 0; i < wwOutCount; i++) {
        float probability = wwOutData[i];
        g_state->latestProbability.store(probability);
        g_state->isActivated.store(probability > g_settings->threshold);
      }

      todoFeatures.erase(todoFeatures.begin(), todoFeatures.begin() + (1 * embFeatures));
      numBufferedFeatures = todoFeatures.size() / embFeatures;
    }
  }
}

extern "C" {

int oww_init(const char* mel_model_path, const char* emb_model_path, const char* ww_model_path) {
  if (g_settings) oww_destroy();

  try {
    g_settings = new Settings();
    g_settings->melModelPath = mel_model_path;
    g_settings->embModelPath = emb_model_path;
    g_settings->wwModelPath = ww_model_path;

    g_settings->options.SetIntraOpNumThreads(1);
    g_settings->options.SetInterOpNumThreads(1);
    g_settings->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    g_state = new EngineState();

    g_melThread = new thread(audioToMels);
    g_featuresThread = new thread(melsToFeatures);
    g_wwThread = new thread(featuresToOutput);

    // Block until all 3 threads have loaded models
    {
      unique_lock<mutex> lockReady(g_state->mutReady);
      g_state->cvReady.wait(lockReady, [] {
        return g_state->numReady == 3;
      });
    }
    return 0; // Success
  } catch (const std::exception& e) {
    return -1; // Error
  }
}

void oww_process_audio(const int16_t* audio_data, int length) {
  if (!g_state) return;

  {
    unique_lock<mutex> lockSamples{g_state->mutSamples};
    g_floatSamples.reserve(g_floatSamples.size() + length);
    for(int i = 0; i < length; i++) {
      g_floatSamples.push_back((float)audio_data[i]);
    }
    g_state->samplesReady = true;
    g_state->cvSamples.notify_one();
  }
}

float oww_get_probability() {
  if (!g_state) return 0.0f;
  return g_state->latestProbability.load();
}

bool oww_is_activated() {
  if (!g_state) return false;
  return g_state->isActivated.load();
}

void oww_destroy() {
  if (!g_state) return;

  {
    unique_lock<mutex> lockSamples{g_state->mutSamples};
    g_state->samplesExhausted = true;
    g_state->samplesReady = true;
    g_state->cvSamples.notify_one();
  }
  if (g_melThread) { g_melThread->join(); delete g_melThread; g_melThread = nullptr; }

  {
    unique_lock<mutex> lockMels{g_state->mutMels};
    g_state->melsExhausted = true;
    g_state->melsReady = true;
    g_state->cvMels.notify_one();
  }
  if (g_featuresThread) { g_featuresThread->join(); delete g_featuresThread; g_featuresThread = nullptr; }

  {
    unique_lock<mutex> lockFeatures{g_state->mutFeatures};
    g_state->featuresExhausted = true;
    g_state->featuresReady = true;
    g_state->cvFeatures.notify_one();
  }
  if (g_wwThread) { g_wwThread->join(); delete g_wwThread; g_wwThread = nullptr; }

  delete g_state;
  g_state = nullptr;

  delete g_settings;
  g_settings = nullptr;
}

} // extern "C"
