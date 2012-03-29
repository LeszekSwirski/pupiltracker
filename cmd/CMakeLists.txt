find_package (OpenCV REQUIRED)

add_executable (
    pupiltrackercmd

    PupilTrackerCmd.cpp
)

target_link_libraries(
    pupiltrackercmd

    pupiltracker
    ${OpenCV_LIBS}
    tbb
)
