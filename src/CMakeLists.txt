project (ML)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.68.0) 
find_package (Eigen3 3.3.5 REQUIRED NO_MODULE)

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "../bin"
)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(ML main.cpp) 
    target_link_libraries(ML ${Boost_LIBRARIES} Eigen3::Eigen)
endif()