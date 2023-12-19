add_executable(integration integration.cu)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

target_link_libraries(integration PRIVATE apbd-lib)

add_test(NAME integration-test COMMAND integration)

