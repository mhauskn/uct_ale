cmake_minimum_required (VERSION 2.6)
project(uct)
include(ExternalProject)

option(USE_SDL "Use SDL" OFF)
if(USE_SDL)
  add_definitions(-D__USE_SDL)
  add_definitions(-DSOUND_SUPPORT)
  find_package(SDL)
  if(SDL_FOUND AND ${SDL_VERSION_STRING} VERSION_LESS 2)
    include_directories(${SDL_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${SDL_LIBRARY} ${SDL_MAIN_LIBRARY})
  else()
    MESSAGE("SDL 1.2 not found: You may need to manually edit CMakeLists.txt or run \"cmake -i\" to specify your SDL path.")
    # Uncomment below to specify the path to your SDL library. Run "locate libSDL" if unsure.
    # link_directories(path_to_your_SDL)
    if(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
      list(APPEND LINK_LIBS sdl sdlmain)
    else()
      list(APPEND LINK_LIBS SDL)
    endif()
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
add_definitions(-std=c++11 -Wno-deprecated)
ExternalProject_Add(ale
  GIT_REPOSITORY "https://github.com/mgbellemare/Arcade-Learning-Environment/"
  GIT_TAG "v0.5.1"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DUSE_SDL="${USE_SDL}"
  UPDATE_COMMAND ""
  INSTALL_COMMAND "")
ExternalProject_Get_Property(ale SOURCE_DIR)
include_directories(${SOURCE_DIR}/src)
link_directories(${SOURCE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

add_executable(uct src/node.cpp src/uct.cpp src/uct_main.cpp)
add_dependencies(uct ale)
target_link_libraries(uct -lale ${GFLAGS_LIBRARY} -lpthread)

add_executable(stay_alive src/node.cpp src/stay_alive.cpp src/uct.cpp src/stay_alive_main.cpp)
add_dependencies(stay_alive ale)
target_link_libraries(stay_alive -lale ${GFLAGS_LIBRARY} -lpthread)
