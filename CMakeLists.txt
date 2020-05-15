cmake_minimum_required(VERSION 2.8)

project(featuretools)

IF(NOT CMAKE_BUILD_TYPE)
#   SET(CMAKE_BUILD_TYPE Debug)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_FLAGS  "-std=c++11")

find_package( OpenCV REQUIRED )

include_directories(
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(Feattool "feat")

add_library(${Feattool}
src/featuredetection.cpp
src/featurematch.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

target_link_libraries(${Feattool}
${OpenCV_LIBS}
)

add_executable(featuretools 
demo.cpp)

target_link_libraries(featuretools 
${OpenCV_LIBS}
${Feattool})
