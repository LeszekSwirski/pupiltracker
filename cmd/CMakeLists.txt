cmake_minimum_required(VERSION 2.8.4)

PROJECT(examples)

include(../3rdParty/dlib/cmake)



find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED)
find_package (TBB REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${pupiltracker_SOURCE_DIR}/lib)
    
 link_directories(${OpenCV_LIB_DIR})


   ADD_EXECUTABLE(PupilTrackerCmd PupilTrackerCmd.cpp)
   TARGET_LINK_LIBRARIES(PupilTrackerCmd pupiltracker dlib ${OpenCV_LIBS} ${TBB_LIBRARIES} ${Boost_LIBRARIES})



