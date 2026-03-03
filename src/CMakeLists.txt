# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(open_wake_word_library VERSION 0.0.1 LANGUAGES C CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(open_wake_word SHARED
  "open_wake_word.cpp"
)

set_target_properties(open_wake_word PROPERTIES
  PUBLIC_HEADER open_wake_word.h
  OUTPUT_NAME "open_wake_word"
)

if (ONNX_DIR AND ANDROID)
  include_directories("${ONNX_DIR}/headers")
  target_link_libraries(open_wake_word PRIVATE "${ONNX_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so")
endif()

target_compile_definitions(open_wake_word PUBLIC DART_SHARED_LIB)

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(open_wake_word PRIVATE "-Wl,-z,max-page-size=16384")
endif()
