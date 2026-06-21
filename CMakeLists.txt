cmake_minimum_required(VERSION 3.22)

project(
  quant_engine
  VERSION 0.1.0
  DESCRIPTION "Production Bermudan swaption pricing engine"
  LANGUAGES CXX
)

include(GNUInstallDirs)

set(QUANT_ENGINE_BINDINGS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/src/bindings.cpp")
set(QUANT_ENGINE_DEFAULT_BUILD_PYTHON OFF)
if(EXISTS "${QUANT_ENGINE_BINDINGS_SOURCE}")
  set(QUANT_ENGINE_DEFAULT_BUILD_PYTHON ON)
endif()

option(QUANT_ENGINE_BUILD_PYTHON "Build the pybind11 Python extension" ${QUANT_ENGINE_DEFAULT_BUILD_PYTHON})
option(QUANT_ENGINE_BUILD_TESTS "Build native C++ tests when present" OFF)
option(QUANT_ENGINE_ENABLE_IPO "Enable interprocedural optimization in release builds" ON)
option(QUANT_ENGINE_ENABLE_NATIVE "Enable host-specific CPU tuning for local builds" OFF)
option(QUANT_ENGINE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QUANT_ENGINE_REQUIRE_COMPLETE_SOURCES "Fail if expected production sources are missing" ${QUANT_ENGINE_BUILD_PYTHON})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(QUANT_ENGINE_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(QUANT_ENGINE_CXX_COMPILER_NAME MATCHES "^zig(\\.exe)?$")
  set(CMAKE_AR "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Zig archiver driver" FORCE)
  set(CMAKE_CXX_COMPILER_AR "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Zig C++ archiver driver" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Zig C++ ranlib driver" FORCE)
  file(TO_NATIVE_PATH "${CMAKE_CXX_COMPILER}" QUANT_ENGINE_ZIG_COMPILER_NATIVE)
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY
      "\"${QUANT_ENGINE_ZIG_COMPILER_NATIVE}\" ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

set(QUANT_ENGINE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include")

set(QUANT_ENGINE_EXPECTED_PUBLIC_HEADERS
  "${QUANT_ENGINE_INCLUDE_DIR}/YieldCurve.hpp"
  "${QUANT_ENGINE_INCLUDE_DIR}/StochasticProcess.hpp"
  "${QUANT_ENGINE_INCLUDE_DIR}/QuasiMonteCarlo.hpp"
  "${QUANT_ENGINE_INCLUDE_DIR}/HullWhiteProcess.hpp"
  "${QUANT_ENGINE_INCLUDE_DIR}/LsmcEngine.hpp"
)

set(QUANT_ENGINE_EXPECTED_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/src/YieldCurve.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/src/HullWhiteProcess.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/src/QuasiMonteCarlo.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/src/LsmcEngine.cpp"
)

set(QUANT_ENGINE_PUBLIC_HEADERS)
set(QUANT_ENGINE_MISSING_PUBLIC_HEADERS)
foreach(header IN LISTS QUANT_ENGINE_EXPECTED_PUBLIC_HEADERS)
  if(EXISTS "${header}")
    list(APPEND QUANT_ENGINE_PUBLIC_HEADERS "${header}")
  else()
    list(APPEND QUANT_ENGINE_MISSING_PUBLIC_HEADERS "${header}")
  endif()
endforeach()

set(QUANT_ENGINE_CORE_SOURCES)
set(QUANT_ENGINE_MISSING_CORE_SOURCES)
foreach(source IN LISTS QUANT_ENGINE_EXPECTED_CORE_SOURCES)
  if(EXISTS "${source}")
    list(APPEND QUANT_ENGINE_CORE_SOURCES "${source}")
  else()
    list(APPEND QUANT_ENGINE_MISSING_CORE_SOURCES "${source}")
  endif()
endforeach()

if(QUANT_ENGINE_REQUIRE_COMPLETE_SOURCES)
  if(QUANT_ENGINE_MISSING_PUBLIC_HEADERS OR QUANT_ENGINE_MISSING_CORE_SOURCES)
    message(FATAL_ERROR
      "Expected production headers or sources are missing. "
      "Disable QUANT_ENGINE_REQUIRE_COMPLETE_SOURCES only for phased scaffolding."
    )
  endif()
endif()

if(QUANT_ENGINE_CORE_SOURCES)
  add_library(quant_engine_core STATIC
    ${QUANT_ENGINE_PUBLIC_HEADERS}
    ${QUANT_ENGINE_CORE_SOURCES}
  )
  set(QUANT_ENGINE_CORE_IS_INTERFACE OFF)
else()
  add_library(quant_engine_core INTERFACE)
  set(QUANT_ENGINE_CORE_IS_INTERFACE ON)
endif()

add_library(quant_engine::core ALIAS quant_engine_core)

if(QUANT_ENGINE_CORE_IS_INTERFACE)
  target_include_directories(quant_engine_core
    INTERFACE
      $<BUILD_INTERFACE:${QUANT_ENGINE_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/quant_engine>
  )
  target_compile_features(quant_engine_core INTERFACE cxx_std_17)
  target_compile_definitions(quant_engine_core
    INTERFACE
      QUANT_ENGINE_VERSION="${PROJECT_VERSION}"
  )
else()
  target_include_directories(quant_engine_core
    PUBLIC
    $<BUILD_INTERFACE:${QUANT_ENGINE_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/quant_engine>
  )

  target_compile_features(quant_engine_core PUBLIC cxx_std_17)

  target_compile_definitions(quant_engine_core
    PUBLIC
      QUANT_ENGINE_VERSION="${PROJECT_VERSION}"
  )
endif()

if(NOT QUANT_ENGINE_CORE_IS_INTERFACE)
  if(MSVC)
    target_compile_options(quant_engine_core PRIVATE /W4 /permissive- /EHsc /bigobj)
  else()
    target_compile_options(quant_engine_core PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      -Wnon-virtual-dtor
    )
  endif()

  if(QUANT_ENGINE_WARNINGS_AS_ERRORS)
    if(MSVC)
      target_compile_options(quant_engine_core PRIVATE /WX)
    else()
      target_compile_options(quant_engine_core PRIVATE -Werror)
    endif()
  endif()

  if(QUANT_ENGINE_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(quant_engine_core PRIVATE -march=native)
  endif()
endif()

if(QUANT_ENGINE_ENABLE_IPO AND NOT QUANT_ENGINE_CORE_IS_INTERFACE)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT QUANT_ENGINE_IPO_SUPPORTED OUTPUT QUANT_ENGINE_IPO_ERROR)
  if(QUANT_ENGINE_IPO_SUPPORTED)
    set_property(TARGET quant_engine_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endif()

if(QUANT_ENGINE_BUILD_PYTHON)
  if(NOT EXISTS "${QUANT_ENGINE_BINDINGS_SOURCE}")
    message(FATAL_ERROR "QUANT_ENGINE_BUILD_PYTHON requires cpp_core/src/bindings.cpp.")
  endif()
  if(QUANT_ENGINE_CORE_IS_INTERFACE)
    message(FATAL_ERROR "QUANT_ENGINE_BUILD_PYTHON requires native implementation sources.")
  endif()

  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE QUANT_ENGINE_PYBIND11_CMAKE_DIR
      RESULT_VARIABLE QUANT_ENGINE_PYBIND11_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(QUANT_ENGINE_PYBIND11_RESULT EQUAL 0 AND EXISTS "${QUANT_ENGINE_PYBIND11_CMAKE_DIR}")
      list(PREPEND CMAKE_PREFIX_PATH "${QUANT_ENGINE_PYBIND11_CMAKE_DIR}")
      find_package(pybind11 CONFIG REQUIRED)
    else()
      message(FATAL_ERROR "pybind11 was not found. Install pybind11 or provide pybind11_DIR.")
    endif()
  endif()

  pybind11_add_module(quant_engine_cpp MODULE
    "${QUANT_ENGINE_BINDINGS_SOURCE}"
  )

  target_link_libraries(quant_engine_cpp PRIVATE quant_engine::core)
  target_compile_features(quant_engine_cpp PRIVATE cxx_std_17)
  target_compile_definitions(quant_engine_cpp PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

  set_target_properties(quant_engine_cpp PROPERTIES
    OUTPUT_NAME "quant_engine_cpp"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )

  if(QUANT_ENGINE_ENABLE_IPO AND QUANT_ENGINE_IPO_SUPPORTED)
    set_property(TARGET quant_engine_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()

  install(TARGETS quant_engine_cpp
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
  )
endif()

if(QUANT_ENGINE_BUILD_TESTS)
  enable_testing()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    add_subdirectory(tests/cpp)
  endif()
endif()

if(QUANT_ENGINE_CORE_IS_INTERFACE)
  install(TARGETS quant_engine_core
    EXPORT quant_engineTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quant_engine
  )
else()
  install(TARGETS quant_engine_core
    EXPORT quant_engineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quant_engine
  )
endif()

install(DIRECTORY "${QUANT_ENGINE_INCLUDE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/quant_engine"
  FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT quant_engineTargets
  NAMESPACE quant_engine::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/quant_engine"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
  DESTINATION quant_engine
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_layer"
  DESTINATION quant_engine
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)
