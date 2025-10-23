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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Executable for OpenMP version
add_executable(main main.cc)

# Executable for Hybrid MPI+OpenMP version
add_executable(main_hybrid main_hybrid.cc)

# Include MPI headers for both
target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_include_directories(main_hybrid PRIVATE ${MPI_CXX_INCLUDE_PATH})

# Link everything (MPI + OpenMP) for both executables
target_link_libraries(main PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_libraries(main_hybrid PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Optional: show compilation info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MPI found: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
