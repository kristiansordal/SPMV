cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)


# Set include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find MPI and METIS
find_package(MPI REQUIRED)
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

# Find Fast Matrix Market
include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

# List of all executables
set(all_executables
    dist
    shared
    seq
)

# Adding executables
foreach(executable IN LISTS all_executables)
    add_executable(${executable} 
    src/main_${executable}.cpp
    src/spmv.cpp
    src/spmv.hpp
    )
endforeach()

# Common include directories and link settings
foreach(executable IN LISTS all_executables)
    target_include_directories(${executable} PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
    target_link_libraries(${executable} PRIVATE 
        ${MPI_CXX_LIBRARIES}
        ${METIS_LIBRARIES}
        OpenMP::OpenMP_CXX
        fast_matrix_market::fast_matrix_market
    )
    target_compile_options(${executable} PRIVATE -fsanitize=address -O3)
    target_link_options(${executable} PRIVATE -fsanitize=address)
endforeach()

set(CMAKE_COLOR_DIAGNOSTICS On)
