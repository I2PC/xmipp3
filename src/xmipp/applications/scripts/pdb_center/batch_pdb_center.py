#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:     Carlos Oscar Sorzano
 *
 * CSIC
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
from xmipp_base import *

class ScriptPDBCenter(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Center a PDB with the center of mass')
        ## params
        self.addParamsLine(' -i <pdb>          : PDB file to process')
        self.addParamsLine(' -o <pdb>          : Output PDB')
        ## examples
        self.addExampleLine('   xmipp_pdb_center -i myfile.pdb -o myfileCenter.pdb')
            

    def readPDB(self,fnIn):
        with open(fnIn) as f:
            self.lines = f.readlines()

    def centerPDB(self):
        xsum = 0.0
        ysum = 0.0
        zsum = 0.0
        N = 0.0
        for line in self.lines:
            if line.startswith("ATOM "):
                try:
                    xsum += float(line[30:38])
                    ysum += float(line[38:46])
                    zsum += float(line[46:54])
                    N += 1
                except:
                    pass

        if N>0:
            xsum /= N
            ysum /= N
            zsum /= N
            # print("Center of mass (x,y,z)=(%f,%f,%f)"%(xsum,ysum,zsum))

        newLines = []
        for line in self.lines:
            if line.startswith("ATOM "):
                try:
                    x = float(line[30:38])-xsum
                    y = float(line[38:46])-ysum
                    z = float(line[46:54])-zsum
                    newLine=line[0:30]+"%8.3f%8.3f%8.3f"%(x,y,z)+line[54:]
                except:
                    pass
            else:
                newLine = line
            newLines.append(newLine)
        self.lines = newLines


    def run(self):
        fnIn = self.getParam('-i')
        fnOut = self.getParam('-o')
        self.readPDB(fnIn)
        self.centerPDB()
        with open(fnOut,"w") as f:
            for line in self.lines:
                f.write("%s"%line)

if __name__ == '__main__':
    ScriptPDBCenter().tryRun()
