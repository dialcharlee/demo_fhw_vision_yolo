cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE "D:/deeplearning/ultralytics-8.4.53/Assess_Vision_27-main/Assess_Vision_27-main/detect/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(armor_detect VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# 启动！
# ============================================================
option(ENABLE_JUDGE "Enable judge logging module" ON)
option(ENABLE_EVALUATE "Enable evaluate module (requires ENABLE_JUDGE)" OFF)

# ============================================================
# 这里是包含了 OpenCV 的依赖
# ============================================================
find_package(OpenCV 4.5 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# 这里是包含了你项目里面需要链接的源文件
# ============================================================
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/target_selector.cpp
    src/visualizer.cpp
)

if(ENABLE_JUDGE)
    list(APPEND SOURCES src/judge.cpp)
    add_definitions(-DENABLE_JUDGE)
    message(STATUS "Judge module: ENABLED")
else()
    message(STATUS "Judge module: DISABLED")
endif()

if(ENABLE_EVALUATE)
    list(APPEND SOURCES src/evaluate.cpp)
    add_definitions(-DENABLE_EVALUATE)
    message(STATUS "Evaluate module: ENABLED")
else()
    message(STATUS "Evaluate module: DISABLED")
endif()

# ============================================================
# 加入构建文件 然后你在 build 里面就能找到他
# ============================================================
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>: /Zi /RTC1 /W4>
    $<$<CONFIG:Release>: /O2 /W4>
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
