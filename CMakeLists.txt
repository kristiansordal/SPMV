cmake_minimum_required(VERSION 3.17)

project(SpMV
    VERSION 0.0.1
    DESCRIPTION "Sparse Matrix Multiplication"
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)

# Find MPI
find_package(MPI REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find METIS and OpenMP
find_package(METIS REQUIRED)
find_package(OpenMP REQUIRED)

# Executables
add_executable(strategySequential 
    src/strategySequential.c
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(strategyA 
    src/strategyA.c
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(strategyB 
    src/strategyB.c
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(strategyC 
    src/strategyC.c
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

add_executable(strategyD 
    src/strategyD.c
    src/mtx.c
    src/mtx.h
    src/spmv.c
    src/spmv.h
)

include_directories(${CMAKE_SOURCE_DIR}/include)

foreach(target  strategySequential strategyA strategyB strategyC strategyD)
    target_include_directories(${target} PRIVATE ${MPI_C_INCLUDE_PATH} ${METIS_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${MPI_C_LIBRARIES} ${METIS_LIBRARIES} OpenMP::OpenMP_C m)
    target_compile_options(${target} PRIVATE -O3 -march=native)
endforeach()
