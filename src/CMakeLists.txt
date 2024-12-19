add_library(ndt_lib SHARED
    VoxelGrid.cpp
    Octree.cpp
    ndt_lib.cpp
)
target_link_libraries(ndt_lib PUBLIC
    Eigen3::Eigen
    ${PCL_LIBRARIES}
)