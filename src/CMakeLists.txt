# collideBoxBox
file(GLOB CBB_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/collideBoxBox/*.h")

add_library(cbb-lib collideBoxBox/odeBoxBox.cu ${CBB_HEADER_LIST})

target_include_directories(cbb-lib PUBLIC ../include)

target_link_libraries(cbb-lib PUBLIC Eigen3::Eigen)

# se3
file(GLOB SE3_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/se3/*.h")

add_library(se3-lib se3/lib.cu ${SE3_HEADER_LIST})

target_include_directories(se3-lib PUBLIC ../include)

target_link_libraries(se3-lib PUBLIC Eigen3::Eigen)

# jgt-float
file(GLOB JGT_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/JGT-float/*.h")

add_library(jgt-lib JGT-float/aabox.cu JGT-float/ray.cu JGT-float/smits_mul.cu ${JGT_HEADER_LIST})

target_include_directories(jgt-lib PUBLIC ../include)

# apbd
file(GLOB APBD_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/apbd/*.h")

add_library(apbd-lib apbd/Model.cu apbd/Body.cu apbd/Collider.cu apbd/Constraint.cu apbd/Shape.cu ${APBD_HEADER_LIST})

target_include_directories(apbd-lib PUBLIC ../include)

target_link_libraries(apbd-lib PUBLIC se3-lib svd3_cuda jgt-lib cbb-lib)

# utils
add_library(utils-lib util.cu ../include/util.h)

target_include_directories(utils-lib PUBLIC ../include)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ../include/util.h ${CBB_HEADER_LIST} ${SE3_HEADER_LIST} ${APBD_HEADER_LIST} ${JGT_HEADER_LIST})
