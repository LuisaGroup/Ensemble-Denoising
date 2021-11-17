cmake_minimum_required(VERSION 3.17...3.20)
project(EnsembleDenoising LANGUAGES C CXX)

add_executable(EnsembleDenoising src/main.cpp src/vector.h src/blend.h src/solve.h src/filter.h)
target_compile_features(EnsembleDenoising PRIVATE cxx_std_17)
target_include_directories(EnsembleDenoising PRIVATE src)

find_package(Eigen3 REQUIRED)
target_include_directories(EnsembleDenoising PRIVATE ${EIGEN3_INCLUDE_DIR})

if (NOT MSVC)# TODO: MSVC seems to have problem with fbstab...
    # fbstab Solver
    add_subdirectory(thirdparty/fbstab)
    target_link_libraries(EnsembleDenoising PRIVATE fbstab_dense pthread)
    target_compile_definitions(EnsembleDenoising PRIVATE ENABLE_FBSTAB=1)
endif()

# cxxopts
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/cxxopts)
target_link_libraries(EnsembleDenoising PRIVATE cxxopts)

# for image I/O
find_package(OpenCV REQUIRED)
target_link_libraries(EnsembleDenoising PRIVATE ${OpenCV_LIBS})
