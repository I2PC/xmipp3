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

include(ExternalProject)

function(fetch_cuFFTAdvisor)
	set(SOURCE_DIR ${CMAKE_BINARY_DIR}/cuFFTAdvisor)

	# Find make
	find_program(MAKE_EXECUTABLE make)

	ExternalProject_Add(
		build_cuFFTAdvisor
		GIT_REPOSITORY https://github.com/I2PC/cuFFTAdvisor.git
		GIT_TAG master
		SOURCE_DIR ${SOURCE_DIR}
		UPDATE_COMMAND ""
		CONFIGURE_COMMAND ""
		BUILD_COMMAND 
			${CMAKE_COMMAND} -E env "PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}" 
			${MAKE_EXECUTABLE} -C <SOURCE_DIR> libcuFFTAdvisor.so
		INSTALL_COMMAND ""
		BUILD_IN_SOURCE TRUE
	)


	# Set output variables
	set(CUFFTADVISOR_INCLUDE_DIR ${SOURCE_DIR})
	set(CUFFTADVISOR_LIB_DIR ${SOURCE_DIR}/build)
	set(CUFFTADVISOR_LIB_NAME libcuFFTAdvisor.so)
	set(CUFFTADVISOR_LIB ${CUFFTADVISOR_LIB_DIR}/${CUFFTADVISOR_LIB_NAME})

	add_library(cuFFTAdvisor SHARED IMPORTED)
	set_target_properties(cuFFTAdvisor PROPERTIES IMPORTED_LOCATION ${CUFFTADVISOR_LIB})
	target_include_directories(cuFFTAdvisor INTERFACE ${CUFFTADVISOR_INCLUDE_DIR})
	add_dependencies(cuFFTAdvisor build_cuFFTAdvisor)
	install(
		FILES ${CUFFTADVISOR_LIB}
 		DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)

	# Propagate variables to parent scope
	set(CUFFTADVISOR_INCLUDE_DIR ${CUFFTADVISOR_INCLUDE_DIR} PARENT_SCOPE)
	set(CUFFTADVISOR_LIB_DIR ${CUFFTADVISOR_LIB_DIR} PARENT_SCOPE)
	set(CUFFTADVISOR_LIB_NAME ${CUFFTADVISOR_LIB_NAME} PARENT_SCOPE)
	set(CUFFTADVISOR_LIB ${CUFFTADVISOR_LIB} PARENT_SCOPE)
endfunction()
