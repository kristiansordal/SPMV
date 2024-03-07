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

add_executable(1a 
    src/1a/matrix.h
    src/1a/read_file.c
    src/1a/read_file.h
    src/1a/spmv.h
    src/1a/spmv.c
    src/1a/main.c
)
add_executable(1b 
    src/1b/matrix.h
    src/1b/read_file.c
    src/1b/read_file.h
    src/1b/main.c
)


add_executable(test
    src/test/main.c
)
target_include_directories(1a PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(1a PRIVATE ${MPI_C_LIBRARIES})
target_compile_options(1a PRIVATE -fsanitize=address)
target_link_options(1a PRIVATE -fsanitize=address)

target_include_directories(1b PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(1b PRIVATE ${MPI_C_LIBRARIES})
target_compile_options(1b PRIVATE -fsanitize=address)
target_link_options(1b PRIVATE -fsanitize=address)


target_include_directories(test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(test PRIVATE ${MPI_C_LIBRARIES})
target_compile_options(test PRIVATE -fsanitize=address)
target_link_options(test PRIVATE -fsanitize=address)
