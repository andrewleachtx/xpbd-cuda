# collideBoxBox
file(GLOB CBB_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/collideBoxBox/*.h")

add_library(cbb-lib STATIC collideBoxBox/odeBoxBox.cu ${CBB_HEADER_LIST})
set_target_properties(cbb-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cbb-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(cbb-lib PUBLIC ../include)

target_link_libraries(cbb-lib PUBLIC Eigen3::Eigen)

# se3
file(GLOB SE3_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/se3/*.h")

add_library(se3-lib STATIC se3/lib.cu ${SE3_HEADER_LIST})
set_target_properties(se3-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(se3-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(se3-lib PUBLIC ../include)

target_link_libraries(se3-lib PUBLIC Eigen3::Eigen)

# jgt-float
file(GLOB JGT_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/JGT-float/*.h")

add_library(jgt-lib STATIC JGT-float/aabox.cu JGT-float/ray.cu JGT-float/smits_mul.cu ${JGT_HEADER_LIST})
set_target_properties(jgt-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(jgt-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(jgt-lib PUBLIC ../include)

# utils
add_library(utils-lib STATIC util.cu ../include/util.h ../include/config.h)
set_target_properties(utils-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(utils-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(utils-lib PUBLIC ../include)

# data
file(GLOB DATA_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/data/*.h")
add_library(data-lib STATIC data/soa.cu ${DATA_HEADER_LIST})
set_target_properties(data-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(data-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(data-lib PUBLIC ../include)
target_link_libraries(data-lib PUBLIC Eigen3::Eigen)

# apbd
file(GLOB APBD_HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/apbd/*.h")

add_library(apbd-lib STATIC apbd/Model.cu apbd/Body.cu apbd/BodyReference.cu apbd/Collider.cu apbd/Constraint.cu apbd/Shape.cu ${APBD_HEADER_LIST} ../include/config.h)
set_target_properties(apbd-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(apbd-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_include_directories(apbd-lib PUBLIC ../include)

target_link_libraries(apbd-lib PUBLIC se3-lib svd3_cuda jgt-lib cbb-lib utils-lib data-lib)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ../include/util.h ${CBB_HEADER_LIST} ${SE3_HEADER_LIST} ${APBD_HEADER_LIST} ${JGT_HEADER_LIST})

if (USE_CUDA)
  target_compile_definitions(utils-lib PUBLIC USE_CUDA)
endif()
if (WRITE)
  target_compile_definitions(utils-lib PUBLIC WRITE)
endif()

