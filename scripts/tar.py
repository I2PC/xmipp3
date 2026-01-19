#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/

import subprocess

import sys
import os
import shutil
from os.path import join, isfile, exists


def usage(error=''):
    errorStr = 'error\n' if error else ''
    print("\n"
          "    %s"
          "\n"
          "    Usage: python tar.py <mode> [v=version] [br=branch]\n"
          "\n"
          "             mode: BinXXXX: Just the binaries. (XXXX is a OS label like 'Debian', 'Centos'...) \n"
          "                   Sources: Just the source code.\n"
          "\n"
          "             branch if for the sources (default: master)\n"
          "\n"
          "             version: X.YY.MM  (version, year and month)\n"
          "    " % errorStr)
    sys.exit(1)


def run(label, version, branch, debug):
    MODES = {'Binaries': 'build', 'Sources': 'src'}

    def getAndWriteCommitInfo(repo):
        """ We write the last commit info in the commit.info file
        """
        cwd = os.getcwd()
        os.chdir('src/%s' % repo)
        hash = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                stdout=subprocess.PIPE).stdout.read().decode("utf-8")
        with open('commit.info', 'w') as file:
            file.write("%s (%s)" % (branch, hash.strip()))
        os.chdir(cwd)

    def makeTarget(target, label):
        if exists(target):
            print("'%s' already exists. Removing it..." % target)
            os.system("rm -rf %s" % target)
        print("...preparing the bundle...")
        sys.stdout.flush()
        cwd = os.getcwd()
        os.system('git clone https://github.com/I2PC/xmipp %s -b %s'
                   % (target, branch))
        if debug:  # in debug mode, the main script and this one is packed
            os.system('cp xmipp %s/xmipp' % target)
            os.system('cp scripts/tar.py %s/scripts/tar.py' % target)
        os.chdir(target)
        os.system('./xmipp get_devel_sources %s' % branch)
        getAndWriteCommitInfo('xmipp')
        getAndWriteCommitInfo('xmippCore')
        getAndWriteCommitInfo('xmippViz')
        getAndWriteCommitInfo('scipion-em-xmipp')
        os.environ['CUDA'] = 'True'  # To include cuFFTAdvisor
        os.system('./xmipp config noAsk')  # just to write the config file
        os.system('./xmipp get_dependencies')
        os.chdir(cwd)

    excludeTgz = ''
    if label.startswith('Bin'):
        print("Recompiling to make sure that last version is there...")
        sublabel = label.split('Bin')[1]
        target = 'xmippBin_%s-%s' % (sublabel, version)
        makeTarget(target, label)
        try:
            # doing compilation and install separately to skip overwriting config
            cwd = os.getcwd()
            os.chdir(target)
            os.system("./xmipp compile 8")
            os.system("./xmipp install %s" % target)
            os.chdir(cwd)
        except:
            raise Exception("  ...some error occurred during the compilation!!!\n")
        checkFile = isfile(join(target, 'bin', 'xmipp_cuda_movie_alignment_correlation'))
        if not checkFile:
            print("\n"
                  "     ERROR: %s not found. \n"
                  "            Xmipp should be compiled using CUDA to make the binaries.tgz."
                  % checkFile)
            sys.exit(1)
        os.system("rm %s/v%s" % (target, version))
        os.system("touch %s/v%s_%s" % (target, version, sublabel))
        excludeTgz = "--exclude='*.tgz' --exclude='*.h' --exclude='*.cpp' " \
                     "--exclude='*.java' --exclude='resources/test' " \
                     "--exclude='*xmipp_test*main'"
    elif label == 'Sources':
        target = 'xmippSrc-v'+version
        makeTarget(target, label)
        excludeTgz = (" --exclude='xmipp.conf' --exclude='xmippEnv.json'"
                      " --exclude='src/scipion-em-xmipp' ")
    else:
        usage("Incorrect <mode>")

    # FIXME: This is breaking the Sources bundle. Please, use a clean dir and skip this
    # excludeTgz += " --exclude='*.o' --exclude='*.os' --exclude='*pyc'"
    # excludeTgz += " --exclude='*.gz' --exclude='*.bashrc' --exclude='*.fish'"
    # excludeTgz += " --exclude=tests/data --exclude='*.scons*' --exclude=.git"
    excludeTgz += ("--exclude=.* --exclude=sonar-project.properties "
                   "--exclude='xmipp.bashrc' --exclude='xmipp.fish' "
                   "--exclude=src/scipion-em-xmipp")

    cmdStr = "tar czf %(target)s.tgz %(excludeTgz)s %(target)s"

    args = {'excludeTgz': excludeTgz, 'target': target}

    cmd = cmdStr % args

    if exists(target+'.tgz'):
        print("%s.tgz already exists. Removing it..." % target)
        os.system("rm -rf %s.tgz" % target)

    print(cmd)
    os.system(cmd)
    os.system("rm -rf %s" % target)


if __name__ == '__main__':

    if not len(sys.argv) > 3:
        usage("Incorrect number of input parameters")

    label = sys.argv[1]
    version = sys.argv[2]
    branch = sys.argv[3] if len(sys.argv)>3 else 'master'
    debug = any([x.lower() == 'debug' for x in sys.argv])

    run(label, version, branch, debug)
