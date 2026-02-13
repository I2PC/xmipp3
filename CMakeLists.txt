#***************************************************************************
# Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307  USA
#
#  All comments concerning this program package may be sent to the
#  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************

cmake_minimum_required(VERSION 3.17)

# Import functions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cifpp.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_ctpl.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cuFFTAdvisor.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_googletest.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_libsvm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_bashrc.cmake)

# Define the project
project(
  xmipp
	VERSION 3.24.06
	LANGUAGES C CXX
)

include(CheckLanguage)
include(CTest)

# Clear version file
set(XMIPP_VERSIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/versions.txt)
file(WRITE ${XMIPP_VERSIONS_FILE} "CMake=${CMAKE_VERSION}\n")
file(APPEND ${XMIPP_VERSIONS_FILE} "CC=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION}\n")
file(APPEND ${XMIPP_VERSIONS_FILE} "CXX=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\n")

# Set compile options
option(XMIPP_USE_CUDA "Compile CUDA programs" ON)
option(XMIPP_USE_MPI "Compile MPI programs" ON)
option(XMIPP_USE_MATLAB "Compile MATLAB binding" ON)
option(XMIPP_LINK_TO_SCIPION "Link to scipion3" ON)
option(XMIPP_SAVE_VERSIONS "Save versions.txt with dependency versions" ON)

# Avoid installing to lib64 directory
set(CMAKE_INSTALL_LIBDIR lib)

# Do not check for updates in installation
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Set C++ 17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(FATAL_ERROR "GCC version must be at least 9.0. Found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
else()
    message(WARNING "You are not using GCC. Currenlty, we only officially support GCC.")
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
file(APPEND ${XMIPP_VERSIONS_FILE} "Python3=${Python3_VERSION}\n")

# Find CUDA
if(${XMIPP_USE_CUDA})
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 10.2)
    if(CUDAToolkit_FOUND)
      file(APPEND ${XMIPP_VERSIONS_FILE} "CUDA=${CUDAToolkit_VERSION}\n")

      set(CMAKE_CUDA_STANDARD 17)

      string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")

      # 1. Define base architectures supported by almost everyone (Turing, Ampere)
      set(XMIPP_CUDA_ARCHS 75 86)

      # 2. Conditionally add Pascal (compute_60) ONLY if CUDA is older than 12.0
      if(${CUDAToolkit_VERSION} VERSION_LESS "12.0")
        list(APPEND XMIPP_CUDA_ARCHS 60)
      endif()

      # 3. Add newer architectures for CUDA 11.8+ (Ada, Hopper)
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
        list(APPEND XMIPP_CUDA_ARCHS 89 90)
      endif()

      if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        # Modern CMake approach: Just pass the list
        set(CMAKE_CUDA_ARCHITECTURES ${XMIPP_CUDA_ARCHS})
      else()
        # Fallback for older CMake: Manual flags construction
        foreach(arch ${XMIPP_CUDA_ARCHS})
          string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=sm_${arch}")
        endforeach()
      endif()

    else()
     set(XMIPP_USE_CUDA OFF)
     message("CUDA was requested but CUDA Toolkit was not found")
    endif()
    
  else()
    set(XMIPP_USE_CUDA OFF)
    message("CUDA was requested but nvcc was not found. Compiling without CUDA")
  endif()
endif()

# Find MPI
if(${XMIPP_USE_MPI})
	find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    set(XMIPP_USE_MPI OFF)
    message("MPI was requested but MPI was not found. Compiling without MPI")
  else()
	  file(APPEND ${XMIPP_VERSIONS_FILE} "MPI=${MPI_CXX_VERSION}\n")
  endif()
endif()

# Find MATLAB
if(${XMIPP_USE_MATLAB})
	find_package(Matlab)

  if (NOT Matlab_FOUND)
    set(XMIPP_USE_MATLAB OFF)
    message("Matlab was requested but Matlab was not found. Compiling without Matlab")
  endif()
endif()

# Fetch dependencies
fetch_cifpp()
fetch_ctpl()
fetch_googletest()
fetch_libsvm()
if(XMIPP_USE_CUDA)
  fetch_cuFFTAdvisor()
endif()

# Import subdirectories
add_subdirectory(src/xmippCore)
add_subdirectory(src/xmipp)
add_subdirectory(src/xmippViz)

# Configure bashrc file for installation
write_bashrc(${CMAKE_CURRENT_BINARY_DIR}/xmipp.bashrc ${CMAKE_INSTALL_PREFIX})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/xmipp.bashrc
  DESTINATION ./
)

# Link to scipion. Keep last so that all other install jobs are complete
if(XMIPP_LINK_TO_SCIPION)
  if (NOT SCIPION_SOFTWARE)
    set(SCIPION_SOFTWARE "$ENV{SCIPION_SOFTWARE}")
  endif()

  if (SCIPION_SOFTWARE)
    message("Scipion software directory found at ${SCIPION_SOFTWARE}")
    set(
      SCIPION_XMIPP_LIBRARIES 
        libcifpp${CMAKE_SHARED_LIBRARY_SUFFIX}
        libcifpp${CMAKE_SHARED_LIBRARY_SUFFIX}.3
        libcifpp${CMAKE_SHARED_LIBRARY_SUFFIX}.5.0.9
        libsvm.so
        libXmippCore${CMAKE_SHARED_LIBRARY_SUFFIX}
        libXmipp${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    install(
      CODE
        "
        include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/link_to_scipion.cmake\")
        link_to_scipion(\"${CMAKE_INSTALL_PREFIX}\" \"${SCIPION_SOFTWARE}\" \"${SCIPION_XMIPP_LIBRARIES}\")
        "
    )
  else()
    message("Linking to scipion was requested, but SCIPION_SOFTWARE is not set")
  endif()
endif()
