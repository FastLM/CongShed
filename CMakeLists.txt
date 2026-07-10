cmake_minimum_required(VERSION 3.16)
project(HICS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hics_core
    topology/topology.cpp
    lstm/lstm_predictor.cpp
    telemetry/telemetry.cpp
    path_engine/path_engine.cpp
    shim/shim.cpp
)

target_include_directories(hics_core PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(hics_demo apps/main.cpp)
target_link_libraries(hics_demo PRIVATE hics_core)

# NCCL plugin shared library (LD_PRELOAD)
add_library(hics_nccl_plugin SHARED
    shim/shim.cpp
    topology/topology.cpp
    lstm/lstm_predictor.cpp
    telemetry/telemetry.cpp
    path_engine/path_engine.cpp
)
target_include_directories(hics_nccl_plugin PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(hics_nccl_plugin PRIVATE HICS_NCCL_PLUGIN=1)
set_target_properties(hics_nccl_plugin PROPERTIES OUTPUT_NAME "hics_nccl")

install(TARGETS hics_core hics_demo hics_nccl_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
