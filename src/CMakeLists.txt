file(GLOB SE3_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/se3/*.h")

add_library(se3-lib se3/lib.cu ${SE3_HEADER_LIST})

target_include_directories(se3-lib PUBLIC ../include)

target_link_libraries(se3-lib PUBLIC Eigen3::Eigen)

file(GLOB APBD_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/apbd/*.h")

add_library(apbd-lib apbd/Model.cu apbd/Body.cu apbd/Collider.cu apbd/Shape.cu ${APBD_HEADER_LIST})

target_include_directories(apbd-lib PUBLIC ../include)

target_link_libraries(apbd-lib PUBLIC se3-lib)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${SE3_HEADER_LIST} ${APBD_HEADER_LIST})
