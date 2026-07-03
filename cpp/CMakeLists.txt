cmake_minimum_required(VERSION 3.16)
project(HICS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hics_core
    src/topology.cpp
    src/lstm_predictor.cpp
    src/telemetry.cpp
    src/path_engine.cpp
    src/shim.cpp
)

target_include_directories(hics_core PUBLIC include)

add_executable(hics_demo src/main.cpp)
target_link_libraries(hics_demo PRIVATE hics_core)

# NCCL plugin shared library (LD_PRELOAD)
add_library(hics_nccl_plugin SHARED src/shim.cpp src/topology.cpp src/lstm_predictor.cpp
    src/telemetry.cpp src/path_engine.cpp)
target_include_directories(hics_nccl_plugin PUBLIC include)
target_compile_definitions(hics_nccl_plugin PRIVATE HICS_NCCL_PLUGIN=1)
set_target_properties(hics_nccl_plugin PROPERTIES OUTPUT_NAME "hics_nccl")

install(TARGETS hics_core hics_demo hics_nccl_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
