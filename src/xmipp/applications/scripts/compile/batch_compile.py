#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:     Carlos Oscar Sorzano
 *              J. M. de la Rosa Trevin
 *
 * Universidad Autonoma de Madrid
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""

import os
import sys
from xmipp_base import XmippScript


class ScriptCompile(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Compile a C++ program using Xmipp libraries')
        ## params
        self.addParamsLine(' -i <cpp_file>          : C++ file to compile')
        self.addParamsLine('   alias --input;')
        self.addParamsLine(' [--debug]              : Compile with debugging flags')
        ## examples
        self.addExampleLine('Compile myprogram.cpp', False)
        self.addExampleLine('xmipp_compile myprogram.cpp')
        self.addExampleLine('where myprogram.cpp contains')
        self.addExampleLine('')
        self.addExampleLine('#include <core/xmipp_image.h>')
        self.addExampleLine('int main()')
        self.addExampleLine('{')
        self.addExampleLine('   Image<double> V("volume.mrc:mrc");')
        self.addExampleLine('   V().printShape();')
        self.addExampleLine('   return 0')
        self.addExampleLine('}')
                                                
    def getFlags(self):
        try:
            from ConfigParser import ConfigParser, ParsingError
        except ImportError:
            from configparser import ConfigParser, ParsingError  # Python 3
        if not 'XMIPP_SRC' in os.environ:
            print("Cannot find the environment variable XMIPP_SRC. Make sure you have sourced the xmipp.bashrc or equivalent")
            sys.exit(1)
        xmippSrc=os.environ['XMIPP_SRC']
        xmippSrcHome=os.path.split(xmippSrc)[0]
        cf = ConfigParser()
        cf.optionxform = str  # keep case (stackoverflow.com/questions/1611799)
        try:
            cf.read(os.path.join(xmippSrcHome,"xmipp.conf"))
        except ParsingError:
            sys.exit("%s\nPlease fix the configuration file install/xmipp.conf." % sys.exc_info()[1])
        flagDict=dict(cf.items('BUILD'))

        flags="-I"+xmippSrc+"/xmippCore -I"+xmippSrc+"/xmipp -lXmipp -lXmippCore "+flagDict["INCDIRFLAGS"]+" "+\
              flagDict["CXXFLAGS"]+" "+flagDict["LIBDIRFLAGS"]
        return flags

    def run(self):
        # type: () -> object
        fn = self.getParam('-i')
        from os.path import splitext, join
        [fnBase,ext]=splitext(fn)
        if ext!=".cpp" and ext!=".cc":
            raise Exception(fn+" is not a .cpp or .cc file")
        command='g++ ';
        if self.checkParam("--debug"):
            command +="-g -pg";
        xmippHome=os.environ['XMIPP_HOME']
        command+=" -o "+fnBase+" "+fn+" -O -D_LINUX -L"+xmippHome+"/lib "+self.getFlags()
        print(command)
        os.system(command)

if __name__ == '__main__':
    ScriptCompile().tryRun()
