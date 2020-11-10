#mod1/CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
include_directories(../json)
add_library(json STATIC
    json_reader.cpp
    json_value.cpp
    json_writer.cpp
    

)