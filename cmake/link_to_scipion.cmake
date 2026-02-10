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


function(link_to_scipion INSTALL_DIRECTORY SCIPION_SOFTWARE SCIPION_XMIPP_LIBRARIES)
	set(SCIPION_SOFTWARE_XMIPP ${SCIPION_SOFTWARE}/em/xmipp)
	set(XMIPP_BINDINGS_DIRECTORY ${SCIPION_SOFTWARE_XMIPP}/bindings/python)
	set(SCIPION_BINDINGS_DIRECTORY ${SCIPION_SOFTWARE}/bindings)
	set(XMIPP_LIB_DIRECTORY ${SCIPION_SOFTWARE_XMIPP}/lib)
	set(SCIPION_LIB_DIRECTORY ${SCIPION_SOFTWARE}/lib)
	set(PYCACHE_DIR_NAME "__pycache__")

	# Link installation
	message("Linking Xmipp installation to Scipion (${INSTALL_DIRECTORY} -> ${SCIPION_SOFTWARE_XMIPP})")
	file(
		CREATE_LINK
			${INSTALL_DIRECTORY}/
			${SCIPION_SOFTWARE_XMIPP}
		COPY_ON_ERROR
		SYMBOLIC
	)
#
#
#	# Link python binding
#	message("Linking Xmipp Python bindings to Scipion (${XMIPP_BINDINGS_DIRECTORY}/* -> ${SCIPION_BINDINGS_DIRECTORY})")
#	file(GLOB PYTHON_DIR_CONTENT ${XMIPP_BINDINGS_DIRECTORY}/*)
#	foreach(x IN LISTS PYTHON_DIR_CONTENT)
#		get_filename_component(y ${x} NAME)
#		if(NOT ${y} MATCHES ${PYCACHE_DIR_NAME}) # Ignore pycache
#			file(
#				CREATE_LINK
#					${x}
#					${SCIPION_BINDINGS_DIRECTORY}/${y}
#				COPY_ON_ERROR
#				SYMBOLIC
#			)
#		endif()
#	endforeach()
#
#	# Link shared libraries
#	message("Linking Xmipp C++ libraries to Scipion (${XMIPP_LIB_DIRECTORY}/* -> ${SCIPION_LIB_DIRECTORY})")
#	foreach(x IN LISTS SCIPION_XMIPP_LIBRARIES)
#		file(
#			CREATE_LINK
#				${XMIPP_LIB_DIRECTORY}/${x}
#				${SCIPION_LIB_DIRECTORY}/${x}
#			COPY_ON_ERROR
#			SYMBOLIC
#		)
#	endforeach()
#endfunction()
#