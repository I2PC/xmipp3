# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307 USA
# *
# * All comments concerning this program package may be sent to the
# * e-mail address 'scipion@cnb.csic.es'

"""
Submodule containing all version info required for Xmipp's installation process.
"""

from .main import XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN

# Xmipp's current versions
LATEST_RELEASE_NUMBER = '3.25.06.0'
LATEST_RELEASE_NAME = 'v3.25.06.0-Rhea'
RELEASE_DATE = '24/06/2025'
#####################################
DEVEL_BRANCHNAME = 'devel'					
MASTER_BRANCHNAME = 'master'				
																		
VERSION_KEY = 'version'							
VERNAME_KEY = 'vername'							
XMIPP_VERSIONS = {									
	XMIPP: {													
		VERSION_KEY: LATEST_RELEASE_NUMBER,
		VERNAME_KEY: LATEST_RELEASE_NAME
	},																
	XMIPP_CORE: {											
		VERSION_KEY: LATEST_RELEASE_NUMBER,
		VERNAME_KEY: LATEST_RELEASE_NAME
	},																
	XMIPP_VIZ: {											
		VERSION_KEY: LATEST_RELEASE_NUMBER,
		VERNAME_KEY: LATEST_RELEASE_NAME
	},																
	XMIPP_PLUGIN: {										
		VERSION_KEY: LATEST_RELEASE_NUMBER,
		VERNAME_KEY: LATEST_RELEASE_NAME
	}																	
}																		
#####################################

# Other variables
UNKNOWN_VALUE = 'Unknown'
