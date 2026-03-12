set(GENERIC_FIELD_SOURCES
    GenericField/Block_Map.cpp
    GenericField/GhostManager.cpp
)

add_library(EmptyApp_GenericField STATIC ${GENERIC_FIELD_SOURCES})

target_include_directories(EmptyApp_GenericField PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/GenericField
)

target_link_libraries(EmptyApp_GenericField PUBLIC ${hiperlife_LIBRARIES})

add_executable(test_Fields tests/test_Fields.cpp)
target_link_libraries(test_Fields EmptyApp_GenericField)

add_executable(test_Ghosts tests/test_Ghosts.cpp)
target_link_libraries(test_Ghosts EmptyApp_GenericField)

install(TARGETS test_Fields test_Ghosts DESTINATION apps)
