set(GENERIC_FIELD_SOURCES
    GenericField/Block_Map.cpp
    GenericField/GhostManager.cpp
    GenericField/DofManager.cpp
)

add_library(HiperlifePrototype_GenericField STATIC ${GENERIC_FIELD_SOURCES})

target_include_directories(HiperlifePrototype_GenericField PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/GenericField
)

target_link_libraries(HiperlifePrototype_GenericField PUBLIC ${hiperlife_LIBRARIES})

add_executable(test_Fields tests/test_Fields.cpp)
target_link_libraries(test_Fields HiperlifePrototype_GenericField)

add_executable(test_Ghosts tests/test_Ghosts.cpp)
target_link_libraries(test_Ghosts HiperlifePrototype_GenericField)

add_executable(test_Mesh tests/test_Mesh.cpp)
target_link_libraries(test_Mesh HiperlifePrototype_GenericField)

add_executable(test_hlComparison tests/test_hlComparison.cpp)
target_link_libraries(test_hlComparison HiperlifePrototype_GenericField)

add_executable(test_DofManager tests/test_DofManager.cpp)
target_link_libraries(test_DofManager HiperlifePrototype_GenericField)

install(TARGETS test_Fields test_Ghosts test_hlComparison test_DofManager DESTINATION apps)