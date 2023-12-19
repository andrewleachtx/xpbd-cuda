cmake_minimum_required(VERSION 3.18..3.27)

project(pbd-rigidbody-cuda VERSION 0.1
    DESCRIPTION "A CUDA-accelerated Differential XPBD Framework"
    LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # to nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)
endif()

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

add_subdirectory(src)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

