set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O3 -Wall")
cmake_minimum_required(VERSION 3.0)

project(timing)

set(OpenGL_GL_PREFERENCE "GLVND")
find_package( OpenGL REQUIRED )

include_directories( ${OPENGL_INCLUDE_DIRS} )

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "external/GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "external/GLFW lib only" )

add_subdirectory( external/glfw )
add_library(glew STATIC external/GL/gl3w.h external/GL/gl3w.c)
add_library( "imgui" "external/imgui/imgui.cpp"
                     "external/imgui/imgui_demo.cpp"
                     "external/imgui/imgui_draw.cpp"
                     "external/imgui/imgui_tables.cpp"
                     "external/imgui/imgui_widgets.cpp"
                     "external/imgui/imgui_impl_glfw.cpp"
                     "external/imgui/imgui_impl_opengl3.cpp" )

set( GLFW-CMAKE-STARTER-SRC
  timing.cpp
)

add_executable( timing WIN32 ${GLFW-CMAKE-STARTER-SRC} )
target_link_libraries( timing ${OPENGL_LIBRARIES} glfw glew imgui)

install(
  TARGETS
    timing
)

install(
  FILES
    timing_marker0.png
    timing_marker1.png
    timing_marker2.png
    timing_marker3.png
  DESTINATION
    include
)
