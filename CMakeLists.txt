#============================================================
#======================== Initialize ========================
#============================================================
cmake_minimum_required(VERSION 3.17.0)

INCLUDE(${CMAKE_SOURCE_DIR}/userConfig.cmake)
project(${PROJECT_NAME} VERSION 5.0)

#============================================================
#===================== Options and flags ====================
#============================================================
# Options
option(BUILD_SHARED_LIBS "Build project with shared libraries." ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/apps" CACHE PATH "Install directory" FORCE)
  message(STATUS "Setting default install path at: ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "To change the install path, add the option -DCMAKE_INSTALL_PREFIX=/path/")
else()
  message(STATUS "Install path set at: ${CMAKE_INSTALL_PREFIX}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  message(STATUS "Project build type not set, default build type: ${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "Project build type set to: ${CMAKE_BUILD_TYPE}")
endif()

#============================================================
#============ HIPERLIFE AND DEPENDENCIES ====================
#============================================================
message(STATUS "Third party libraries:")

if(NOT HL_BASE_PATH)
    # Try to get HL_BASE_PATH from the environment
    if(DEFINED ENV{HL_BASE_PATH})
        set(HL_BASE_PATH $ENV{HL_BASE_PATH})
        message(STATUS "Using HL_BASE_PATH: ${HL_BASE_PATH}")
    else()
        message(FATAL_ERROR "HL_BASE_PATH is not set and no environment variable HL_BASE_PATH found.")
    endif()
endif()

find_package(hiperlife QUIET HINTS ${HL_BASE_PATH})
if(hiperlife_FOUND)
  message(STATUS "  hiperlife:\t Found version ${hiperlife_VERSION} at ${hiperlife_DIR}")
  message(STATUS "  hiperlife include dirs: ${hiperlife_INCLUDE_DIR}")
  message(STATUS "  hiperlife libs: ${hiperlife_LIBRARIES}")
else()
  message(FATAL_ERROR "hiperlife not found at '${HL_BASE_PATH}'. Please add a valid hiperlife installation with the cmake option -DHL_BASE_PATH=/path/to/hiperlife")
endif(hiperlife_FOUND)

if(hiperlife_VERSION VERSION_GREATER ${PROJECT_NAME}_VERSION)
  message(FATAL_ERROR "The version of hiperlife (${hiperlife_VERSION}) is greater than this project version (${hl_BaseProject_VERSION}). Please update")
elseif(NOT hiperlife_VERSION VERSION_EQUAL ${PROJECT_NAME}_VERSION)
  message(WARNING "hiperlife version (${hiperlife_VERSION}) and this project version (${${PROJECT_NAME}_VERSION}) do not match. This may cause issues. Please consider updating")
endif()

#============================================================
#=================== PROJECT SETTINGS =======================
#============================================================
set(PROJECT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
include_directories(${hiperlife_INCLUDE_DIR})

#Add (or modify) applications that do not depend on vtk and gmsh here:
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/EmptyApp)

