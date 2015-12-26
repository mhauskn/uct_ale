cmake_minimum_required (VERSION 2.6)
project(uct)
include(ExternalProject)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
add_definitions(-std=c++11 -Wno-deprecated)

ExternalProject_Add(ale
  GIT_REPOSITORY "https://github.com/mgbellemare/Arcade-Learning-Environment/"
  GIT_TAG "v0.5.1"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
  UPDATE_COMMAND ""
  INSTALL_COMMAND "")
ExternalProject_Get_Property(ale SOURCE_DIR)
include_directories(${SOURCE_DIR}/src)
link_directories(${SOURCE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

add_executable(uct node.cpp uct.cpp uct_main.cpp)
add_dependencies(uct ale)
target_link_libraries(uct -lale ${GFLAGS_LIBRARY} -lpthread)

add_executable(stay_alive node.cpp stay_alive.cpp uct.cpp stay_alive_main.cpp)
add_dependencies(stay_alive ale)
target_link_libraries(stay_alive -lale ${GFLAGS_LIBRARY} -lpthread)
