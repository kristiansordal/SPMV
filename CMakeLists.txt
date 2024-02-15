cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

# Find MPI
find_package(MPI REQUIRED)

add_executable(test 
    src/matrix.h
    src/read_file.c
    src/read_file.h
    src/spmv.h
    src/spmv.c
    src/main.c
)

target_include_directories(test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(test PRIVATE ${MPI_C_LIBRARIES})
target_compile_options(test PRIVATE -fsanitize=address)
target_link_options(test PRIVATE -fsanitize=address)
