cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)


# Set include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find MPI and METIS
find_package(MPI REQUIRED)
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

# add_executable(1a 
#     src/main_1a.c
#     src/spmv.c
# )
add_executable(1b 
    src/main_1b.c
    src/spmv.c
    src/mtx.c
)


# target_include_directories(1a PRIVATE ${MPI_C_INCLUDE_PATH})
# target_link_libraries(1a PRIVATE ${MPI_C_LIBRARIES})
# target_compile_options(1a PRIVATE -fsanitize=address)
# target_link_options(1a PRIVATE -fsanitize=address)

target_include_directories(1b PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
target_link_libraries(1b PRIVATE 
    ${MPI_C_LIBRARIES}
    ${METIS_LIBRARIES}
    OpenMP::OpenMP_C
)
target_compile_options(1b PRIVATE -fsanitize=address -lm -O3)
target_link_options(1b PRIVATE -fsanitize=address)
