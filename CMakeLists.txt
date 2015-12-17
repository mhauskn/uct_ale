cmake_minimum_required (VERSION 2.6)
project(uct)

set(ALE_ROOT_DIR "../Arcade-Learning-Environment")
list(APPEND CMAKE_PREFIX_PATH ${ALE_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

add_executable(uct uct.cpp)

find_package(ALE REQUIRED)
include_directories(${ALE_INCLUDE_DIRS})
target_link_libraries(uct ${ALE_LIBRARIES})
