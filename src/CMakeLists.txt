set(BINARY ${CMAKE_PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(Eigen3)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# message(STATUS "Eigen3 library status:")
# message(STATUS "    version: ${Eigen3_VERSION}")
# message(STATUS "    include path: ${EIGEN3_INCLUDE_DIR}")

file(GLOB_RECURSE SOURCES LIST_DIRECRORIES true *.h *.cpp)
set(SOURCES ${SOURCES})
add_executable(run_${BINARY} ${SOURCES})

target_link_libraries(run_${BINARY} ${OpenCV_LIBS})  