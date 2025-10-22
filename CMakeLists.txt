cmake_minimum_required(VERSION 3.15)
project(MyMPIProject LANGUAGES CXX)

# Always export compile_commands.json (useful for IDEs and static analysis)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use C++17 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimization flags for Release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Add common optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0 -march=native -DNDEBUG")

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Executable
add_executable(main main.cc)

# Include MPI headers
target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_PATH})

# Link everything (MPI + OpenMP)
target_link_libraries(main PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Optional: show compilation info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MPI found: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
