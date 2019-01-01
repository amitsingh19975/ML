project (ML)
find_package (Eigen3 3.3.5 REQUIRED NO_MODULE)

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "../bin"
) 
add_executable(ML main.cpp) 
target_link_libraries(ML ${Boost_LIBRARIES} Eigen3::Eigen)