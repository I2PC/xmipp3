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

cmake_minimum_required(VERSION 3.16)

include(GNUInstallDirs)

# Find source files
file(
	GLOB_RECURSE 
	SOURCES 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/classification/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/data/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/dimred/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/interface/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/tomo/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/external/condor/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/external/delaunay/*.cpp 
)
file(
	GLOB_RECURSE 
	HEADERS 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/classification/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/data/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/dimred/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/interface/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/tomo/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/external/condor/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/external/delaunay/*.h
)	
file(
	GLOB_RECURSE 
	FRM_SOURCES 
		${CMAKE_CURRENT_SOURCE_DIR}/external/sh_alignment/*.cpp 
)
file(
	GLOB_RECURSE 
	PYTHON_BINDING_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/*.cpp 
)
file(
	GLOB_RECURSE 
	MATLAB_BINDING_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/bindings/matlab/*.cpp 
)
file(
	GLOB_RECURSE 
	MATLAB_BINDING_SCRIPTS
		${CMAKE_CURRENT_SOURCE_DIR}/bindings/matlab/*.m
)
file(
	GLOB_RECURSE 
	PARALLEL_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/parallel/*.cpp
)
file(
	GLOB_RECURSE 
	CUDA_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction_cuda/*.cpp
)
file(
	GLOB_RECURSE 
	CUDA11_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction_cuda11/*.cpp
)
file(
	GLOB_RECURSE 
	CUDA_INTERFACE_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction_adapt_cuda/*.cpp
)
file(
	GLOB_RECURSE 
	CUDA11_INTERFACE_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/reconstruction_adapt_cuda11/*.cpp
)
file(
	GLOB_RECURSE 
	PARALLEL_CUDA_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/parallel_adapt_cuda/*.cpp
)
file(
	GLOB_RECURSE 
	PARALLEL_CUDA11_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/libraries/parallel_adapt_cuda11/*.cpp
)

# Compile FRM
add_library(swig_frm MODULE ${FRM_SOURCES})
set_target_properties(swig_frm PROPERTIES PREFIX "_")
target_link_libraries(
	swig_frm
	PUBLIC
		Python3::Python
		Python3::NumPy
		XmippCore
)
install(
	TARGETS swig_frm
	LIBRARY DESTINATION bindings/python
)
install(
	DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/sh_alignment/python/
	DESTINATION bindings/python/sh_alignment
)
install(
	FILES ${CMAKE_CURRENT_SOURCE_DIR}/external/sh_alignment/swig_frm.py
	DESTINATION bindings/python/sh_alignment
)

# Create the shared library
add_library(Xmipp SHARED ${SOURCES})
target_include_directories(
	Xmipp
	PUBLIC 
		${CMAKE_CURRENT_SOURCE_DIR}/libraries
		${CMAKE_CURRENT_SOURCE_DIR}/external
)
#target_precompile_headers(
#	xmippLib
#	PRIVATE
#		${HEADERS}
#)
target_link_libraries(
	Xmipp
	PUBLIC
		XmippCore
		Python3::Python
		Python3::NumPy
		cifpp::cifpp
		ctpl
		libsvm
)
install(
	TARGETS Xmipp
 	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Create the Python binding

set(PYTHON_BINDING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python)
set(PYTHON_BINDING_INSTALL_DIR bindings/python)

install(
	FILES
		${PYTHON_BINDING_DIR}/xmipp_base.py
		${PYTHON_BINDING_DIR}/xmipp_conda_envs.py
		${PYTHON_BINDING_DIR}/xmipp.py
	DESTINATION
		${PYTHON_BINDING_INSTALL_DIR}
)
install(
	DIRECTORY
		${PYTHON_BINDING_DIR}/envs_DLTK
	DESTINATION
		${PYTHON_BINDING_INSTALL_DIR}
)


# Register MPI sources
if(${XMIPP_USE_MPI})
	add_library(XmippParallel SHARED ${PARALLEL_SOURCES})
	target_link_libraries(
		XmippParallel
		PUBLIC
			Xmipp
			MPI::MPI_CXX
	)
	install(
		TARGETS XmippParallel
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

# Register CUDA sources
if(${XMIPP_USE_CUDA})
	set_source_files_properties(
 		${CUDA_SOURCES}
		PROPERTIES LANGUAGE CUDA
	)
	add_library(XmippCuda STATIC ${CUDA_SOURCES})
	if (${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
		set_source_files_properties(
			${CUDA11_SOURCES}
			PROPERTIES LANGUAGE CUDA
		)
		target_sources(XmippCuda PRIVATE ${CUDA11_SOURCES})
	endif()

	set_property(
		TARGET 
			XmippCuda 
		PROPERTY 
			POSITION_INDEPENDENT_CODE ON
	)

	# HACK: Manually add cuFFTAdvisor. Linking the target leads to wrong RPATH
	target_include_directories(
		XmippCuda
		PUBLIC
			${CUFFTADVISOR_INCLUDE_DIR}
	)	
	target_link_directories(
		XmippCuda
		PUBLIC
			${CUFFTADVISOR_LIB_DIR}
	)
	target_link_libraries(
		XmippCuda
		PUBLIC
			${CUFFTADVISOR_LIB_NAME}
	)
	add_dependencies(XmippCuda cuFFTAdvisor)

	target_link_libraries(
		XmippCuda
		PUBLIC
			Xmipp
			CUDA::cudart
			CUDA::cufft
			CUDA::nvml
	)

	add_library(XmippInterfaceCuda SHARED ${CUDA_INTERFACE_SOURCES})
	if (${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
		target_sources(XmippInterfaceCuda PRIVATE ${CUDA11_INTERFACE_SOURCES})
	endif()

	target_link_libraries(
		XmippInterfaceCuda
		PUBLIC
			Xmipp
		INTERFACE
			XmippCuda
	)
	install(
		TARGETS XmippInterfaceCuda
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

# Register combined MPI and CUDA sources
if(${XMIPP_USE_CUDA} AND ${XMIPP_USE_MPI})
	add_library(XmippParallelCuda SHARED ${PARALLEL_CUDA_SOURCES})
	if (${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
		target_sources(XmippParallelCuda PRIVATE ${PARALLEL_CUDA11_SOURCES})
	endif()

	target_link_libraries(
		XmippParallelCuda
		PUBLIC
			XmippParallel
		INTERFACE
			XmippInterfaceCuda
	)
	install(
		TARGETS XmippParallelCuda
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

# Add MATLAB binding
if(${XMIPP_USE_MATLAB})
	foreach(MATLAB_BINDING_SOURCE ${MATLAB_BINDING_SOURCES})
		get_filename_component(MEX_NAME ${MATLAB_BINDING_SOURCE} NAME_WE)
		set(TARGET_NAME "XmippMatlab-${MEX_NAME}")
		matlab_add_mex(
			NAME ${TARGET_NAME}
			SRC ${MATLAB_BINDING_SOURCE}
			OUTPUT_NAME ${MEX_NAME}
			LINK_TO Xmipp
		)
		install(
			TARGETS ${TARGET_NAME}
			DESTINATION bindings/matlab
		)
	endforeach()
	
	install(
		FILES ${MATLAB_BINDING_SCRIPTS}
		DESTINATION bindings/matlab
	)
endif()

# Install python library
install(
	DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libraries/py_xmipp/
	DESTINATION pylib/xmippPyModules/
)

# Add applications
add_subdirectory(applications)
