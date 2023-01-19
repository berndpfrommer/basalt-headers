cmake_minimum_required(VERSION 3.10...3.18)

include(CMakePackageConfigHelpers)

project(basalt-headers VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest) # note: this adds a BUILD_TESTING which defaults to ON

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
      message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
      set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
      set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else()
      message(STATUS "Dind't find ccache")
    endif()
  else()
    message(STATUS "Compiler launcher already set. Not configuring ccache.")
    message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
    message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
  endif()

  IF( NOT CMAKE_BUILD_TYPE )
    SET( CMAKE_BUILD_TYPE Release)
  ENDIF()

  IF(NOT CXX_MARCH)
    SET(CXX_MARCH native)
  ENDIF()

  IF(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(BASALT_HEADERS_MARCH_FLAGS "-march=${CXX_MARCH}")
  ELSE()
    message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
  ENDIF()


  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)


  # clang-specific compile flags
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-misleading-indentation")
    endif()
  else()
    set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-error=maybe-uninitialized")
  endif()


  set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
  SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra ${BASALT_HEADERS_MARCH_FLAGS} ${CMAKE_CXX_FLAGS} ${BASALT_HEADERS_CXX_FLAGS}")
  SET(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
  SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
  SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
  set(CMAKE_CXX_FLAGS_SANITIZERDEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined")
  set(CMAKE_CXX_FLAGS_SANITIZERRELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined")

endif()

function(report_dependency_location DEP_NAME DEP_TARGET)
  if(${DEP_NAME}_DIR)
    # this is set if dependency is found via find_package
    set(ACTUAL_DEP_CMAKE_DIR ${${DEP_NAME}_DIR})
  else()
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0")
      get_target_property(ACTUAL_DEP_CMAKE_DIR ${DEP_TARGET} SOURCE_DIR)
    else()
      # SOURCE_DIR works for interface targets only since 3.19
      set(ACTUAL_DEP_CMAKE_DIR "")
    endif()
  endif()
  get_target_property(ACTUAL_DEP_INCLUDE_DIR ${DEP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Using ${DEP_NAME} from: ${ACTUAL_DEP_CMAKE_DIR} (include: ${ACTUAL_DEP_INCLUDE_DIR})")
endfunction()

option(BASALT_BUILTIN_EIGEN "Use builtin Eigen from submodule" ON)
if(NOT TARGET Eigen3::Eigen)
  if(BASALT_BUILTIN_EIGEN)
    message(STATUS "Including internal Eigen from submodule")
    # Disable Eigen tests
    set(BUILD_TESTING_PREV_VALUE ${BUILD_TESTING})  # Note: BUILD_TESTING is always defined because we include CTest above
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/eigen EXCLUDE_FROM_ALL)
    # Restore previous value of BUILD_TESTING
    set(BUILD_TESTING ${BUILD_TESTING_PREV_VALUE} CACHE BOOL "Build the testing tree." FORCE)
  else()
    message(STATUS "Finding external Eigen")
    find_package(Eigen3 3.4 REQUIRED)
  endif()
else()
  message(STATUS "Eigen already available")
endif()
report_dependency_location(Eigen3 Eigen3::Eigen)

option(BASALT_BUILTIN_SOPHUS "Use builtin Sophus from submodule" ON)
if(NOT TARGET Sophus::Sophus)
  if(BASALT_BUILTIN_SOPHUS)
    message(STATUS "Including internal Sophus from submodule")
    set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "disable BUILD_SOPHUS_TESTS")
    set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "disable BUILD_SOPHUS_EXAMPLES")
    add_subdirectory(thirdparty/Sophus)
  else()
    message(STATUS "Finding external Sophus")
    find_package(Sophus REQUIRED)
  endif()
else()
  message(STATUS "Sophus already available")
endif()
report_dependency_location(Sophus Sophus::Sophus)

option(BASALT_BUILTIN_CEREAL "Use builtin Cereal from submodule" ON)
if(NOT TARGET cereal::cereal)
  if(BASALT_BUILTIN_CEREAL)
    message(STATUS "Including internal Cereal from submodule")
    add_subdirectory(thirdparty/cereal)
  else()
    message(STATUS "Finding external Cereal")
    find_package(cereal REQUIRED)
  endif()
else()
  message(STATUS "Cereal already available")
endif()
report_dependency_location(cereal cereal::cereal)

add_library(basalt-headers INTERFACE)
add_library (basalt::basalt-headers ALIAS basalt-headers)
target_link_libraries(basalt-headers INTERFACE Eigen3::Eigen Sophus::Sophus cereal::cereal)

# Associate target with include directory
target_include_directories(basalt-headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# List all header files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
set(ACTUAL_INCLUDE_DIR "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>$<INSTALL_INTERFACE:include>")

target_sources(basalt-headers
  INTERFACE
    ${ACTUAL_INCLUDE_DIR}/basalt/calibration/calib_bias.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/calibration/calibration.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/bal_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/camera_static_assert.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/double_sphere_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/extended_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/fov_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/generic_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/kannala_brandt_camera4.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/pinhole_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/stereographic_param.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/camera/unified_camera.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/image/image.h
    ${ACTUAL_INCLUDE_DIR}/basalt/image/image_pyr.h
    ${ACTUAL_INCLUDE_DIR}/basalt/imu/imu_types.h
    ${ACTUAL_INCLUDE_DIR}/basalt/imu/preintegration.h
    ${ACTUAL_INCLUDE_DIR}/basalt/serialization/eigen_io.h
    ${ACTUAL_INCLUDE_DIR}/basalt/serialization/headers_serialization.h
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/ceres_local_param.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/ceres_spline_helper.h
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/rd_spline.h
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/se3_spline.h
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/so3_spline.h
    ${ACTUAL_INCLUDE_DIR}/basalt/spline/spline_common.h
    ${ACTUAL_INCLUDE_DIR}/basalt/utils/assert.h
    ${ACTUAL_INCLUDE_DIR}/basalt/utils/eigen_utils.hpp
    ${ACTUAL_INCLUDE_DIR}/basalt/utils/hash.h
    ${ACTUAL_INCLUDE_DIR}/basalt/utils/sophus_utils.hpp
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  INCLUDES DESTINATION include)
 
# generate the ConfigVersion.cmake file that will be included by Config.cmake
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${EXPORT_LIBRARY_VERSION}
  COMPATIBILITY AnyNewerVersion)

# generate the *Config.cmake file
configure_file(cmake/${PROJECT_NAME}Config.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)

# generate Targets.cmake file from exports learned during the installation
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# install the generated version and config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BASALT_HEADERS_BUILD_TESTING) AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
