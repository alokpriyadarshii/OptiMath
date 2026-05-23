cmake_minimum_required(VERSION 3.20)

project(OptiMathCPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OPTIMATH_BUILD_TESTS "Build unit tests" ON)
option(OPTIMATH_BUILD_APPS  "Build CLI app(s)" ON)
option(OPTIMATH_ENABLE_ASAN "Enable AddressSanitizer (Clang/GCC)" OFF)
option(OPTIMATH_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (Clang/GCC)" OFF)
option(OPTIMATH_ENABLE_WERROR "Treat warnings as errors" OFF)

add_library(optimath
    src/core/status.cpp
    src/core/timer.cpp
    src/linalg/vector.cpp
    src/linalg/matrix.cpp
    src/linalg/solve.cpp
    src/lp/linear_program.cpp
    src/lp/simplex.cpp
    src/lp/mip_branch_and_bound.cpp
    src/nlp/objective.cpp
    src/nlp/bfgs.cpp
    src/nlp/penalty.cpp
)

target_include_directories(optimath
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(optimath PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(optimath PRIVATE /W4)
    if(OPTIMATH_ENABLE_WERROR)
        target_compile_options(optimath PRIVATE /WX)
    endif()
else()
    target_compile_options(optimath PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
    if(OPTIMATH_ENABLE_WERROR)
        target_compile_options(optimath PRIVATE -Werror)
    endif()
    if(OPTIMATH_ENABLE_ASAN)
        target_compile_options(optimath PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(optimath PRIVATE -fsanitize=address)
    endif()
    if(OPTIMATH_ENABLE_UBSAN)
        target_compile_options(optimath PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(optimath PRIVATE -fsanitize=undefined)
    endif()
endif()

if(OPTIMATH_BUILD_APPS)
    add_executable(optimath_cli apps/optimath_cli.cpp)
    target_link_libraries(optimath_cli PRIVATE optimath)
endif()

if(OPTIMATH_BUILD_TESTS)
    enable_testing()
    add_executable(optimath_tests
        tests/test_main.cpp
        tests/test_lp_simplex.cpp
        tests/test_nlp_bfgs.cpp
    )
    target_link_libraries(optimath_tests PRIVATE optimath)
    add_test(NAME optimath_tests COMMAND optimath_tests)
endif()
