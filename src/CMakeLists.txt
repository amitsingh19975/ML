project (ML)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.68.0) 

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "../bin"
)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(ML main.cpp) 
    target_link_libraries(ML ${Boost_LIBRARIES})
endif()