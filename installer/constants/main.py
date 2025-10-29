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
# ***************************************************************************/

"""
Submodule containing all the general constants needed for Xmipp's installation.
"""

import os

XMIPP = 'xmipp'
XMIPP_CORE = 'xmippCore'
XMIPP_VIZ = 'xmippViz'
XMIPP_PLUGIN = 'scipion-em-xmipp'

# Formatting characters
UP = "\x1B[1A\r"
REMOVE_LINE = '\033[K'
BOLD = "\033[1m"
BLUE = "\033[34m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
END_FORMAT = "\033[0m"
FORMATTING_CHARACTERS = [UP, REMOVE_LINE, BOLD, BLUE, RED, GREEN, YELLOW, END_FORMAT]

__BASE_GITHUB_URL = 'https://github.com'
__ORGANIZATION_NAME = 'I2PC'
REPOSITORIES = {
	XMIPP: [os.path.join(__BASE_GITHUB_URL, __ORGANIZATION_NAME, XMIPP), None],
	XMIPP_CORE: [os.path.join(__BASE_GITHUB_URL, __ORGANIZATION_NAME, XMIPP_CORE), None],
	XMIPP_VIZ: [os.path.join(__BASE_GITHUB_URL, __ORGANIZATION_NAME, XMIPP_VIZ), None],
	XMIPP_PLUGIN: [os.path.join(__BASE_GITHUB_URL, __ORGANIZATION_NAME, XMIPP_PLUGIN), None],
}
XMIPP_SOURCES = [XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]
SOURCES_PATH = "src"
BUILD_PATH = "build"
INSTALL_PATH = "dist"
BUILD_TYPE = "Release"
CMAKE_CACHE_PATH = os.path.join(BUILD_PATH, 'CMakeCache.txt')

# Internal URLs
API_URL = 'https://xmipp.i2pc.es/api/attempts/'
DOCUMENTATION_URL = 'https://i2pc.github.io/docs/'

# External URLs
CMAKE_INSTALL_DOCS_URL = 'https://i2pc.github.io/docs/Installation/Requirements/index.html#requirements'
MODELS_URL = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
SCIPION_FILES_REMOTE_PATH = "scipionfiles/downloads/scipion/software/em"
SCIPION_TESTS_URLS = "http://scipion.cnb.csic.es/downloads/scipion/data/tests"
SCIPION_SOFTWARE_EM = "scipionfiles/downloads/scipion/software/em"

# Other variables
TAB_SIZE = 4
SECTION_MESSAGE_LEN = 60
TAG_BRANCH_NAME = 'v3.25.06.0-Rhea'

# File names
CONFIG_FILE = 'xmipp.conf'
LOG_FILE = 'compilation.log'
VERSION_FILE = 'build/versions.txt'
TAIL_LOG_NCHARS = 300
