cmake_minimum_required(VERSION 2.8) 

add_definitions("-Wall -std=c++11") # <= 新增的编译选项 

add_subdirectory(json) #添加子目录
add_subdirectory(lib_json)  #添加子目录

add_executable(Main main.cpp)
target_link_libraries(Main json)#目标执行文件的lib依赖
