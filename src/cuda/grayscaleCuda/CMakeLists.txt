cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project( GrayCuda CXX CUDA )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( GrayCuda GrayCuda.cu )
target_link_libraries( GrayCuda ${OpenCV_LIBS} )
