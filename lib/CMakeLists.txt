find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED)
find_package (TBB REQUIRED)

add_library (
    pupiltracker

    PupilTracker.cpp
    cvx.cpp
    utils.cpp
)

include_directories(
    SYSTEM
	${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}    
)
