find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED)
find_package (TBB REQUIRED)

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)
link_directories(
    ${OpenCV_LIB_DIR}
)
add_executable (
    pupiltrackercmd

    PupilTrackerCmd.cpp
)

target_link_libraries(
    pupiltrackercmd

    pupiltracker
    ${OpenCV_LIBS}
    ${TBB_LIBRARIES}
)
