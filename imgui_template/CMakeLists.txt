# CMakeList.txt : CMake project for imgui_template, include source and define
# project specific logic here.
#

# glfw3
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

# opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

# openmp
find_package(OpenMP REQUIRED)

# pthreads?
find_package(PThreads REQUIRED)

# cuda
find_package(CUDA REQUIRED)

message(STATUS "CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_PATH ${CUDA_PATH}")

# cudnn
find_package(CUDNN REQUIRED)

# yolo
find_package(Darknet CONFIG REQUIRED)

# fmt
find_package(fmt CONFIG REQUIRED)

#find_library(ZLIBWAPI zlibwapi)
## not finding these although they're part of cuda?
## curand
#find_library(CURAND curand)
#
## cublas
#find_library(CUBLAS cublas)
set(BACKUP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
add_subdirectory("nativefiledialog-extended")
set(BUILD_SHARED_LIBS ${BACKUP_BUILD_SHARED_LIBS})
#find_package(nativefiledialog-extended REQUIRED)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 20)
endif()

# Add source to this project's executable.
add_executable (${PROJECT_NAME} "imgui_template.cpp" "imgui_template.h" "imgui/imconfig.h" "imgui/imgui.cpp" "imgui/imgui.h" "imgui/imgui_demo.cpp" "imgui/imgui_draw.cpp" "imgui/imgui_impl_glfw.cpp" "imgui/imgui_impl_glfw.h" "imgui/imgui_impl_opengl3.cpp" "imgui/imgui_impl_opengl3.h" "imgui/imgui_impl_opengl3_loader.h" "imgui/imgui_internal.h" "imgui/imgui_stdlib.cpp" "imgui/imgui_stdlib.h" "imgui/imgui_tables.cpp" "imgui/imgui_widgets.cpp" "imgui/imstb_rectpack.h" "imgui/imstb_textedit.h" "imgui/imstb_truetype.h" "zpp_bits/zpp_bits.h" "ankerl/unordered_dense.h" "file-cpp/file.h" "nothings/stb_image.h" "yolo_preconfig/yolo7_template.h" "utf8-cpp/source/utf8.h" "utf8-cpp/source/utf8/checked.h" "utf8-cpp/source/utf8/core.h" "utf8-cpp/source/utf8/unchecked.h" "stack_vector/stack_vector.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()


# TODO: Add tests and install targets if needed.
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/nativefiledialog-extended")
target_link_directories(${PROJECT_NAME} PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" "${CUDA_TOOLKIT_ROOT_DIR}/bin")

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to ${PROJECT_NAME}
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
#        "${CUDA_TOOLKIT_ROOT_DIR}/bin/test.dll"      # <--this is in-file
#        $<TARGET_FILE_DIR:${PROJECT_NAME}>)        # <--this is out-file path

#copy cuda related dlls
#if (BUILD_SHARED_LIBS) 
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CUDA_TOOLKIT_ROOT_DIR}/bin"
		"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>") #$<TARGET_FILE_DIR:${PROJECT_NAME}>
#install(
#  DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}/bin"
#  TYPE BIN
#  FILES_MATCHING REGEX "[^\\\\/.]\\.[dD][lL][lL]$"
#)
#endif()
##copy cuda related libs
#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
#        $<TARGET_FILE_DIR:${PROJECT_NAME}>)

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD        # Adds a post-build event to ${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
        "${CMAKE_CURRENT_SOURCE_DIR}/zlib123dllx64/dll_x64/zlibwapi.dll"      # <--this is in-file
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/zlibwapi.dll")        # <--this is out-file path
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD        # Adds a post-build event to ${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
        "${CMAKE_CURRENT_SOURCE_DIR}/zlib123dllx64/dll_x64/zlibwapi.exp"      # <--this is in-file
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/zlibwapi.exp")        # <--this is out-file path
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD        # Adds a post-build event to ${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
        "${CMAKE_CURRENT_SOURCE_DIR}/zlib123dllx64/dll_x64/zlibwapi.lib"      # <--this is in-file
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/zlibwapi.lib")        # <--this is out-file path

target_link_libraries(${PROJECT_NAME} PRIVATE
	glfw
	fmt::fmt-header-only
	nfd
	OpenMP::OpenMP_CXX
	#zlibwapi
	#curand
	#cublas
	#cuda
	${CUDNN_LIBRARIES}
	Darknet::dark
	${OPENGL_LIBRARIES}
)