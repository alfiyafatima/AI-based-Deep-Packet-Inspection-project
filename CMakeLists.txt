cmake_minimum_required(VERSION 3.16)
project(PacketAnalyzer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header files location
include_directories(${CMAKE_SOURCE_DIR}/include)

# Common library source files
set(DPI_SOURCES
    src/pcap_reader.cpp
    src/packet_parser.cpp
    src/sni_extractor.cpp
    src/types.cpp
    src/rule_manager.cpp
    src/connection_tracker.cpp
    src/fast_path.cpp
    src/load_balancer.cpp
    src/dpi_engine.cpp
    src/flow_features.cpp
    src/ml_classifier.cpp
)

# Executable for simple packet analyzer
add_executable(packet_analyzer src/main.cpp ${DPI_SOURCES})

# Executable for DPI Engine
add_executable(dpi_engine src/main_dpi.cpp ${DPI_SOURCES})
