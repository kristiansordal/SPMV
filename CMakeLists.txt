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

add_executable(1b 
    src/main.cpp
)



target_include_directories(1b PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1b PRIVATE 
    ${MPI_CXX_LIBRARIES}
    ${METIS_LIBRARIES}
    OpenMP::OpenMP_CXX
    fast_matrix_market::fast_matrix_market
    m

)
target_compile_options(1b PRIVATE -fsanitize=address -O3)
target_link_options(1b PRIVATE -fsanitize=address)

set(CMAKE_COLOR_DIAGNOSTICS On)
