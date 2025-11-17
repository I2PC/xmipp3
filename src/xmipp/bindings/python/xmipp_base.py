import re
from collections import OrderedDict
from distutils.spawn import find_executable

from xmippLib import *
import os
import sys
import subprocess


def xmippExists(path):
    return FileName(path).exists()

def getXmippPath(*paths):
    """ Return the path of the Xmipp installation folder
        if a subfolder is provided, will be concatenated to the path
    """

    candidates = []  # First candidate from XMIPP_HOME, second from this file path
    envHome = os.environ.get('XMIPP_HOME', '')  # the join do nothing if second is absolute
    candidates.append(os.path.join(os.environ.get('SCIPION_HOME', ''), envHome))
    # xmipp    =          build      <   bindings    <     python    <     xmipp_base.py
    candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

    for xmippHome in candidates:
        if (os.path.isfile(os.path.join(xmippHome, 'lib', 'libXmipp.so')) and
            os.path.isfile(os.path.join(xmippHome, 'bin', 'xmipp_mpi_reconstruct_significant'))):
            return os.path.join(xmippHome, *paths)

    raise Exception("Error: Xmipp build directory not found. Searched at:\n - %s"
                    % '\n - '.join(candidates))


# def getMatlabEnviron(*toolPaths):
# TODO: check that this is not needed before deleting
# TODO:    getMatlabEnviron() function can be found in the xmipp3.Plugin class.
# TODO:    The function here below uses undefined objs: Environ and getEnviron()
#     """ Return an Environment prepared for launching Matlab
#     scripts using the Xmipp binding.
#     """
#     env = getEnviron()
#     env.set('PATH', os.environ['MATLAB_BINDIR'], Environ.BEGIN)
#     env.set('LD_LIBRARY_PATH', os.environ['MATLAB_LIBDIR'], Environ.BEGIN)
#     for toolpath in toolPaths:
#         env.set('MATLABPATH', toolpath, Environ.BEGIN)
#     env.set('MATLABPATH', os.path.join(os.environ['XMIPP_HOME'], 'libraries', 'bindings', 'matlab'),
#             Environ.BEGIN)
#
#     return env


class XmippScript:
    """ This class will serve as wrapper around the XmippProgram class
        to have same facilities from Python scripts
    """
    def __init__(self, runWithoutArgs=False):
        self._prog = Program(runWithoutArgs)

    def defineParams(self):
        """ This function should be overwrited by subclasses for
        define its own parameters """
        pass

    def readParams(self):
        """ This function should be overwrited by subclasses for
        and take desired params from command line """
        pass

    def checkParam(self, param):
        return self._prog.checkParam(param)

    def getParam(self, param, index=0):
        return self._prog.getParam(param, index)

    def getIntParam(self, param, index=0):
        return int(self._prog.getParam(param, index))

    def getDoubleParam(self, param, index=0):
        return float(self._prog.getParam(param, index))

    def getListParam(self, param):
        return self._prog.getListParam(param)

    def addUsageLine(self, line, verbatim=False):
        self._prog.addUsageLine(line, verbatim)

    def addExampleLine(self, line, verbatim=True):
        self._prog.addExampleLine(line, verbatim)

    def addParamsLine(self, line):
        self._prog.addParamsLine(line)

    def run(self):  # type: () -> object
        """ This function should be overwrited by subclasses and
        it the main body of the script """
        pass

    def tryRun(self):
        """ This function should be overwrited by subclasses and
        it the main body of the script """
        try:
            print("WARNING: This is xmipp_base implementation for script")
            self.defineParams()
            doRun = self._prog.read(sys.argv)
            if doRun:
                self.readParams()
                self.run()
            return 0
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

    @staticmethod
    def getModel(*modelPath, **kwargs):
        """ Returns the path to the models folder followed by
            the given relative path.
        .../xmipp/models/myModel/myFile.h5 <= getModel('myModel', 'myFile.h5')

            NOTE: it raise and exception when model not found, set doRaise=False
                  in the arguments to skip that raise, especially in validation
                  asserions!
        """
        return getModel(*modelPath, **kwargs)

    @classmethod
    def runCondaCmd(cls, program, arguments, **kwargs):
        """ This class method is used to run programs that are independent of xmipp
            but employ conda. The class should possess a _conda_env attribute to be used.
            Otherwise  CONDA_DEFAULT_ENVIRON is used. To use xmipp dependent
            programs, runCondaJob within a XmippProtocol is preferred.
                :param program: str. A program/pythonScript to execute
                :param arguments: str. The arguments for the program
                :param kwargs: options
                :return: None
        """
        condaEnvName = CondaEnvManager.getCondaName(cls)

        kwargs['env'] = CondaEnvManager.getCondaEnv(kwargs.get('env', os.environ),
                                                    condaEnvName)
        cmd_args = program + " " + arguments
        print(cmd_args)
        try:
            subprocess.check_call(cmd_args, shell=True, **kwargs)
        except subprocess.CalledProcessError as e:
            subprocess.check_call(cmd_args, shell=True, **kwargs)


class CondaEnvManager(object):
    CONDA_DEFAULT_ENVIRON = "xmipp_DLTK_v0.3"
    from xmipp_conda_envs import XMIPP_CONDA_ENVS

    @staticmethod
    def getCondaName(xmippCls, **kwargs):
        """ Returns the conda environ name associated to the xmippCls.
                XmippCls can be:
                  - XmippProtocol (Scipion's plugin)
                  - XmippScript (defined above)
            > _conda_env preference: kwargs > protocol default > general default
        """
        name = kwargs.get('_conda_env', None)
        if name is None:
            if hasattr(xmippCls, '_conda_env'):
                name = xmippCls._conda_env
            else:
                name = CondaEnvManager.CONDA_DEFAULT_ENVIRON
                print("Warning: using default Xmipp conda environment '%s'. "
                    "CondaJobs should be run under a specific environment to "
                    "avoid problems. Please, fix it or contact to the developer."
                    % name)
        return name

    @staticmethod
    def getCondaExe(env=None):
        """ Tries to find the conda executable in an environment
            :param: env. An environ, using os.environ by default
            :return: None if CondaExe not found via CONDA_EXE,
                     CONDA_ACTIVATION_CMD, CONDA_HOME nor 'which conda'.
        """
        env = env if env else os.environ

        condaExe = find_executable(env.get("CONDA_EXE", "conda"), env['PATH'])
        condaHome = ''
        condaActCmd = env.get("CONDA_ACTIVATION_CMD", None)
        if not condaExe and condaActCmd:
            # CONDA_ACTIVATION_CMD = "eval '$(condaExe shell.bash hook)'"  ('<=>")
            condaRe = re.match("[\'\"]?eval [\'\"]?\$\((.*) shell.bash hook\)[\'\"]?",
                               condaActCmd)
            if condaRe:
                condaExe = condaRe.group(1)
            else:
                # CONDA_ACTIVATION_CMD = ". '/path/to/condaHome/etc/profile.d/conda.sh)'"  ('<=>")
                condaRe = re.match("[\'\"]?\. [\'\"]?(.*)/etc/profile.d/conda.sh[\'\"]?[\'\"]?",
                                   condaActCmd)
                if condaRe:
                    condaHome = condaRe.group(1)
                    p = subprocess.Popen(condaActCmd+"&& echo $_CONDA_EXE",
                                         shell=True,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                    condaExe = p.stdout.readline().decode('utf8').strip()
            # print("getting conda from CONDA_ACTIVATION_CMD = %s" % condaActCmd)

        condaHome = env.get("CONDA_HOME", condaHome)
        if not condaExe and condaHome:
            condaExe = os.path.join(condaHome, 'bin', 'conda')

        condaExe = os.path.expanduser(condaExe if condaExe else '')
        if os.path.isfile(condaExe):
            return condaExe
        else:
            print("No conda found...")
            if condaActCmd:
                print("please, check the CONDA_ACTIVATION_CMD = %s "
                      "in the config file. If it seems fine, try to add "
                      "CONDA_EXE = /path/to/conda/executable" % condaActCmd)

    @staticmethod
    def getEnvironDir(condaEnv):
        """ This function returns the condaEnv directory
            using the 'conda info --env' command.
        """
        cmd = "%s info --env" % CondaEnvManager.getCondaExe()
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.stdout.readlines()
        for line in output:
            regex = re.match(condaEnv+"[ ]+(\*)?[ ]+(.*)", line.decode("utf-8"))
            if regex:
                # isActived = regex.group(1) is not None
                return regex.group(2)
        print("\n$ "+cmd)
        print("".join([l.decode('utf8') for l in output]))
        print(" >>> '%s' conda environment not found... (check list above)" % condaEnv)

    @staticmethod
    def getCondaEnv(env, condaEnv):
        """ Setting the environ (based on 'env')
            according to the 'condaEnv' name passed.
            'environDir/bin' is prepended in PATH
            PYTHONPATH is set/prepend (depending on the 'xmippEnviron' flag)
               with 'environDir/lib/python*/site-packages'
            TODO: consider to also prepend the LD_LIBRARY_PATH
        """
        environDir = CondaEnvManager.getEnvironDir(condaEnv)
        envBin = os.path.join(environDir, "bin")
        env.update({"PATH": envBin+':'+env["PATH"]})  # <- PATH

        sitePackages = os.path.join("lib", "python*", "site-packages")
        newPythonPath= os.path.join(environDir, sitePackages)
        if CondaEnvManager.XMIPP_CONDA_ENVS[condaEnv]["xmippEnviron"]:
            newPythonPath += ":"+env["PYTHONPATH"]
        env.update({"PYTHONPATH": newPythonPath})  # <- PYTHONPATH
        # print(env["PYTHONPATH"])
        env['PYTHONWARNINGS'] = 'ignore::FutureWarning'  # to skip warnings
        return env

    def getCondaEnvTargetFilename(self, name):
        env = CondaEnvManager.XMIPP_CONDA_ENVS[name]

        if "versionId" in env:
            target = f'{name}-{env["versionId"]}.yml'
        else:
            target = f'{name}.yml'

        return target

    @staticmethod
    def getCondaActivationCmd():
        """ This method takes the command to activate conda
            the CONDA_ACTIVATION_CMD present in the environ.
            If not there or it fails, we construct one from
            the 'conda' executable if found.
        """
        condaActCmd = os.environ.get('CONDA_ACTIVATION_CMD', "")
        if (condaActCmd.startswith("'") and condaActCmd.endswith("'") or
            condaActCmd.startswith('"') and condaActCmd.endswith('"')):
            condaActCmd = condaActCmd[1:-1]  # remove extra quotes

        if ((not condaActCmd or os.system(condaActCmd))
            and CondaEnvManager.getCondaExe()):
            condaActCmd = ('eval "$(%s shell.bash hook)"'
                           % CondaEnvManager.getCondaExe())

        if condaActCmd.endswith("&&"):
            condaActCmd = condaActCmd[:-2]
        elif condaActCmd.endswith(";") or condaActCmd.endswith("&"):
            condaActCmd = condaActCmd[:-1]

        if not condaActCmd:
            msg = ("\n\nConda activation command not found. "
                   "Please, add CONDA_ACTIVATION_CMD to the config file.\n\n\n")
            condaActCmd = "echo %s exit " % msg
        return condaActCmd

    @staticmethod
    def yieldInstallAllCmds(useGpu):
        for name, env in CondaEnvManager.XMIPP_CONDA_ENVS.items():
            yield CondaEnvManager.installEnvironCmd(
                name=name, 
                versionId=env.get('versionId', None), 
                requirementsFn=env['requirements'], 
                gpu=useGpu
            )

    @staticmethod
    def getCurInstalledDep(dependency, defaultVersion=None, environ=None):
        """ Returns the current version of a certain dependency
            installed in pip. Returns just the dependency if not found.
            i.e.: 'numpy==1.18.1'=getCurInstalledDep('numpy') <- 1.18.1 ver. found
                  'numpy==1.18.0'=getCurInstalledDep('numpy', 1.18.0) <- no ver. found
                  'numpy'=getCurInstalledDep('numpy') <- no ver. found
        """
        env = environ if environ else os.environ
        p = subprocess.Popen("pip list --no-cache-dir | grep "+dependency,
                             shell=True, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            # expected string: "dep    1.2.34a3"
            reMatch = re.match("%s +([0-9a-zA-Z\.]+)" % dependency,
                               line.decode('utf8').strip())
            if reMatch:
                defaultVersion = reMatch.group(1)
        return dependency+'=='+defaultVersion if defaultVersion else dependency

    @staticmethod
    def installEnvironCmd(name: str,requirementsFn: str,  versionId: int = None, gpu=False):
        # Consider the gpu version if requested
        if gpu:
            root, ext = os.path.splitext(requirementsFn)
            gpuRequirementsFn = root + '-gpu' + ext
            
            if os.path.exists(gpuRequirementsFn):
                requirementsFn = gpuRequirementsFn
        
        if versionId is not None:
            target = f'{name}-{versionId}.yml'
        else:
            target = f'{name}.yml'

        commands = [] 
        commands.append('conda env create -f %s || conda env update -f %s' % (requirementsFn, requirementsFn))
        commands.append('touch %s' % target)
        return ' && '.join(commands), target

def getModel(*modelPath, **kwargs):
    """ Returns the path to the models folder followed by
        the given relative path.
    .../xmipp/models/myModel/myFile.h5 <= getModel('myModel', 'myFile.h5')

        NOTE: it raise and exception when model not found, set doRaise=False
              in the arguments to skip that raise, especially in validation
              asserions!
    """
    model = getXmippPath('models', *modelPath)
    # Raising an error to prevent posterior errors and to print a hint
    if kwargs.get('doRaise', True) and not os.path.exists(model):
        raise Exception("'%s' model not found. Please, run: \n"
                        " > scipion installb deepLearningToolkit" % modelPath[0])
    return model


# FIXME: XmippMdRow is almost the same than pwem.emlib.metadata.utils.Row
# FIXME:  It's here because deep_denoising needs it and pwem can't be imported
class XmippMdRow:
    """ Support Xmipp class to store label and value pairs corresponding to a
        Metadata row.  - Code duplication alert!
         - Use this only outside of a Scipion plugin (i.e. in Xmipp programs).
         - For Scipion plugins (including Xmipp), use pwem.emlib.metadata.Row()
    """
    def __init__(self):
        self._labelDict = OrderedDict()  # Dictionary containing labels and values
        self._objId = None  # Set this id when reading from a metadata

    def getObjId(self):
        return self._objId

    def hasLabel(self, label):
        return self.containsLabel(label)

    def containsLabel(self, label):
        # Allow getValue using the label string
        if isinstance(label, str):
            label = str2Label(label)
        return label in self._labelDict

    def removeLabel(self, label):
        if self.hasLabel(label):
            del self._labelDict[label]

    def setValue(self, label, value):
        """args: this list should contains tuples with
        MetaData Label and the desired value"""
        # Allow setValue using the label string
        if isinstance(label, str):
            label = str2Label(label)
        self._labelDict[label] = value

    def getValue(self, label, default=None):
        """ Return the value of the row for a given label. """
        # Allow getValue using the label string
        if isinstance(label, str):
            label = str2Label(label)
        return self._labelDict.get(label, default)

    def readFromMd(self, md, objId):
        """ Get all row values from a given id of a metadata. """
        self._labelDict.clear()
        self._objId = objId

        for label in md.getActiveLabels():
            self._labelDict[label] = md.getValue(label, objId)

    def addToMd(self, md):
        self.writeToMd(md, md.addObject())

    def writeToMd(self, md, objId):
        """ Set back row values to a metadata row. """
        for label, value in self._labelDict.items():
            # TODO: Check how to handle correctly unicode type
            # in Xmipp and Scipion
            t = type(value)

            if t is str:
                value = str(value)

            if t is int and labelType(label) == LABEL_SIZET:
                value = int(value)

            try:
                md.setValue(label, value, objId)
            except Exception as ex:
                print("XmippMdRow.writeToMd: Error writing value to metadata.",
                      file=sys.stderr)
                print("                     label: %s, value: %s, type(value): %s"
                      % (label2Str(label), value, type(value)), file=sys.stderr)
                raise ex

    def readFromFile(self, fn):
        md = MetaData(fn)
        self.readFromMd(md, md.firstObject())

    def copyFromRow(self, other):
        for label, value in other._labelDict.items():
            self.setValue(label, value)

    def __str__(self):
        s = '{'
        for k, v in self._labelDict.items():
            s += '  %s = %s\n' % (emlib.label2Str(k), v)
        return s + '}'

    def __iter__(self):
        return self._labelDict.items()

    def printDict(self):
        """ Fancy printing of the row, mainly for debugging. """
        print(str(self))


def createMetaDataFromPattern(pattern, isStack=False, label="image"):
    ''' Create a metadata from files matching pattern'''
    import glob
    if isinstance(pattern, list):
        files = []
        for pat in pattern:
            files += glob.glob(pat)
    else:
        files = glob.glob(pattern)
    files.sort()

    label = str2Label(label)  # Check for label value
    
    mD = MetaData()
    inFile = FileName()
    
    nSize = 1
    for file in files:
        fileAux=file
        if isStack:
            if file.endswith(".mrc"):
                fileAux=file+":mrcs"
            x, x, x, nSize = getImageSize(fileAux)
        if nSize != 1:
            counter = 1
            for jj in range(nSize):
                inFile.compose(counter, fileAux)
                objId = mD.addObject()
                mD.setValue(label, inFile, objId)
                mD.setValue(MDL_ENABLED, 1, objId)
                counter += 1
        else:
            objId = mD.addObject()
            mD.setValue(label, fileAux, objId)
            mD.setValue(MDL_ENABLED, 1, objId)
    return mD


# TODO: All this MD functions can be implemented in core.metadata
#        but they are not, so far. Thus, they are here directly in python.
def getMdSize(filename):
    """ Return the metadata size without parsing entirely. """
    md = MetaData()
    md.read(filename, 1)
    return md.getParsedLines()


def isMdEmpty(filename):
    """ Use getMdSize to check if metadata is empty. """
    return getMdSize(filename) == 0


def readInfoField(fnDir, block, label, xmdFile="iterInfo.xmd"):
    mdInfo = MetaData("%s@%s" % (block, os.path.join(fnDir, xmdFile)))
    return mdInfo.getValue(label, mdInfo.firstObject())


def writeInfoField(fnDir, block, label, value, xmdFile="iterInfo.xmd"):
    mdInfo = MetaData()
    objId = mdInfo.addObject()
    mdInfo.setValue(label, value, objId)
    mdInfo.write("%s@%s" % (block, os.path.join(fnDir, xmdFile)), MD_APPEND)
