cmake_minimum_required(VERSION 3.16)
project(HICS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HICS_HAVE_NCCL "Build against system NCCL net.h" OFF)

set(HICS_SOURCES
    topology/topology.cpp
    lstm/lstm_predictor.cpp
    telemetry/hw_counters.cpp
    telemetry/telemetry.cpp
    path_engine/path_engine.cpp
    transfer/transfer_executor.cpp
    shim/shim.cpp
    shim/ucx_transport.cpp
    shim/nccl_plugin.cpp
)

add_library(hics_core ${HICS_SOURCES})
target_include_directories(hics_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(hics_core PUBLIC pthread ${CMAKE_DL_LIBS})

if(HICS_HAVE_NCCL)
    target_compile_definitions(hics_core PUBLIC HICS_HAVE_NCCL_NET_H=1)
endif()

add_executable(hics_demo apps/main.cpp)
target_link_libraries(hics_demo PRIVATE hics_core)

# NCCL plugin shared library (NCCL_NET_PLUGIN / LD_PRELOAD)
add_library(hics_nccl_plugin SHARED ${HICS_SOURCES})
target_include_directories(hics_nccl_plugin PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(hics_nccl_plugin PUBLIC pthread ${CMAKE_DL_LIBS})
target_compile_definitions(hics_nccl_plugin PRIVATE HICS_NCCL_PLUGIN=1)
set_target_properties(hics_nccl_plugin PROPERTIES OUTPUT_NAME "hics_nccl")
if(HICS_HAVE_NCCL)
    target_compile_definitions(hics_nccl_plugin PUBLIC HICS_HAVE_NCCL_NET_H=1)
endif()

install(TARGETS hics_core hics_demo hics_nccl_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
