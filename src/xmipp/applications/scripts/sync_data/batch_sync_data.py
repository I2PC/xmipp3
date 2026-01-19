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

import hashlib
import os
import sys

from subprocess import call
from os.path import join
from urllib.request import urlopen

import time

def blue(text):
    return "\033[34m"+text+"\033[0m"

def download(destination=None, url=None, dataset=None):
    """ Download all the data files mentioned in url/dataset/MANIFEST
    """
    isDLmodel = dataset=="DLmodels"
    if not isDLmodel:
        # First make sure that we ask for a known dataset.
        if dataset not in [x.decode("utf8").strip('./\n') for x in urlopen('%s/MANIFEST'%url)]:
            print(blue("Unknown dataset/model: %s)" % dataset))
            return
        remoteManifest = '%s/%s/MANIFEST' % (url, dataset)
        inFolder = "/%s" % dataset
    else:
        remoteManifest = '%s/xmipp_models_MANIFEST' % url
        inFolder = ''

    # Retrieve the dataset's MANIFEST file.
    # It contains a list of "file md5sum" of all files included in the dataset.
    if not os.path.isdir(destination):
        os.makedirs(destination)
    manifest = join(destination, 'MANIFEST')
    try:
        print(blue("Retrieving MANIFEST file"))
        open(manifest, 'wb').writelines(
            urlopen(remoteManifest))
    except Exception as e:
        sys.exit("ERROR reading %s (%s)" % (remoteManifest, e))

    # Now retrieve all of the files mentioned in MANIFEST, and check their md5.
    print(blue('Fetching files...'))
    md5sRemote = readManifest(remoteManifest, isDLmodel)
    done = 0.0  # fraction already done
    inc = 1.0 / len(md5sRemote)  # increment, how much each iteration represents
    oldPartial = 100
    for fname, md5Remote in md5sRemote.items():
        fpath = join(destination, fname)
        try:
            # Download content and create file with it.
            if not os.path.isdir(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
            open(fpath, 'wb').writelines(
                urlopen('%s%s/%s' % (url, inFolder, fname)))

            md5 = md5sum(fpath)
            assert md5 == md5Remote, \
                "Bad md5. Expected: %s Computed: %s" % (md5Remote, md5)

            done += inc
            partial = int(done*10)
            if int((done-inc)*100%10) == 0 and partial != oldPartial:
                print(blue("%3d%%..." % (100 * done)))
                sys.stdout.flush()
                oldPartial = partial
        except Exception as e:
            print(blue("\nError in %s (%s)" % (fname, e)))
            print(blue("URL: %s/%s/%s" % (url, dataset, fname)))
            print(blue("Destination: %s" % fpath))
            if input("Continue downloading? (y/[n]): ").lower() != 'y':
                sys.exit()
    if isDLmodel:
        unTarModels(destination)

def update(destination=None, url=None, dataset=None):
    """ Update local dataset with the contents of the remote one.
    It compares the md5 of remote files in url/dataset/MANIFEST with the
    ones in workingCopy/dataset/MANIFEST, and downloads only when necessary.
    """
    isDLmodel = dataset=="DLmodels"
    prefix = "xmipp_models_" if isDLmodel else ''
    inFolder = "" if isDLmodel else "/%s" % dataset

    # Read contents of *remote* MANIFEST file, and create a dict {fname: md5}
    remoteManifest = '%s/%sMANIFEST' % (url, prefix) if isDLmodel \
                         else '%s/%s/MANIFEST' % (url, dataset)

    md5sRemote = readManifest(remoteManifest, isDLmodel)

    # just in case
    if not os.path.isdir(destination):
        os.makedirs(destination)

    # Update and read contents of *local* MANIFEST file, and create a dict
    try:
        last = max(os.stat(join(destination, x)).st_mtime for x in md5sRemote)
        t_manifest = os.stat(join(destination, 'MANIFEST')).st_mtime
        assert t_manifest > last and time.time() - t_manifest < 60*60*24*7
    except (OSError, IOError, AssertionError, FileNotFoundError) as e:
        print(blue("Regenerating local MANIFEST..."))
        if isDLmodel:
            if any(x.startswith('xmipp_model_') and x.endswith('.tgz')
                   for x in os.listdir(destination)):
                os.system('(cd %s ; md5sum xmipp_model_*.tgz '
                          '> MANIFEST)' % destination)
            else:
                os.system('touch %s/MANIFEST' % destination)
        else:
            createMANIFEST(destination)

    md5sLocal = dict(x.split() for x in open(join(destination, 'MANIFEST')))
    if isDLmodel:  # DLmodels has hashs before fileNames
        md5sLocal = {v: k for k, v in md5sLocal.items()}
    # Check that all the files mentioned in MANIFEST are up-to-date
    print(blue("Verifying MD5s..."))

    filesUpdated = []  # number of files that have been updated
    taintedMANIFEST = False  # can MANIFEST be out of sync?

    done = 0.0  # fraction already done
    inc = 1.0 / len(md5sRemote)  # increment, how much each iteration represents
    oldPartial = 100
    for fname in md5sRemote:
        fpath = join(destination, fname)
        try:
            if os.path.exists(fpath) and md5sLocal.get(fname, 'None') == md5sRemote.get(fname, ''):
                pass  # just to emphasize that we do nothing in this case
            else:
                if not os.path.isdir(os.path.dirname(fpath)):
                    os.makedirs(os.path.dirname(fpath))
                open(fpath, 'wb').writelines(
                    urlopen('%s%s/%s' % (url, inFolder, fname)))
                filesUpdated.append(fname)
        except Exception as e:
            print(blue("\nError while updating %s: %s" % (fname, e)))
            taintedMANIFEST = True  # if we don't update, it can be wrong
        done += inc
        partial = int(done*10)
        if int((done-inc)*100%10) == 0 and partial != oldPartial:
            print(blue("%3d%%..." % (100 * done)))
            sys.stdout.flush()
            oldPartial = partial


    print(blue("...done. Updated files: %d" % len(filesUpdated)))
    sys.stdout.flush()

    # Save the new MANIFEST file in the folder of the downloaded dataset
    if len(filesUpdated) > 0:
        open(join(destination, 'MANIFEST'), 'wb').writelines(urlopen(remoteManifest).readlines())

    if taintedMANIFEST:
        print(blue("Some files could not be updated. Regenerating local MANIFEST ..."))
        createMANIFEST(destination)

    if isDLmodel:
        unTarModels(destination)

def upload(login, tgzName, remoteFolder, update):
    """ Upload a dataset to our repository
    """
    localFn = os.path.join("models", tgzName)
    if not os.path.exists(localFn):
        sys.exit("ERROR: local folder/file %s does not exist." % localFn)

    modelName = os.path.basename(localFn)
    isDLmodel = modelName.startswith("xmipp_model_")
    url = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
    remoteModels = readManifest(url+'/xmipp_models_MANIFEST', True).keys()

    if update != "--update":
        for model in remoteModels:
            if modelName == model:
                print("\nError: The '%s' name already exists." % modelName)
                print("       Add '--update' to OVERRIDE it.\n")
                sys.exit(1)

    # Upload the dataset files (with rsync)
    print("Uploading files...")
    callResult = call(['rsync', '-rlv', '--chmod=a+r', localFn,
                       '%s:%s' % (login, remoteFolder)])
    if callResult != 0:
        sys.exit("\n > Upload failed, you may have no permissions.\n\n"
                 "   Please check the login introduced or contact to "
                 "'scipion@cnb.csic.es' \n"
                 "   for uploading the model placed at '%s'.\n" % localFn)

    # Regenerate remote MANIFEST (which contains a list of datasets)
    if isDLmodel:
        # This is a file that just contains the name of the xmipp_models
        # in remoteFolder. Nothing to do with the MANIFEST files in
        # the datasets, which contain file names and md5s.
        print("Regenerating remote MANIFEST file...")
        call(['ssh', login,
              'cd %s && md5sum xmipp_model_*.tgz > xmipp_models_MANIFEST'
              % remoteFolder])

def md5sum(fname):
    """ Return the md5 hash of file fname
    """
    mhash = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * mhash.block_size), b""):
            mhash.update(chunk)
    return mhash.hexdigest()

def createMANIFEST(path):
    """ Create a MANIFEST file in path with the md5 of all files below
    """
    with open(join(path, 'MANIFEST'), 'wb') as manifest:
        for root, dirs, files in os.walk(path):
            for filename in set(files) - {'MANIFEST'}:  # all but ourselves
                fn = join(root, filename)  # file to check
                manifest.write(('%s %s\n' % (os.path.relpath(fn, path), md5sum(fn))).encode())

def readManifest(remoteManifest, isDLmodel):
    manifest = urlopen(remoteManifest).readlines()
    md5sRemote = dict(x.decode("utf8").strip().split() for x in manifest)
    if isDLmodel:  # DLmodels has hashs before fileNames
        md5sRemote = {v: k for k, v in md5sRemote.items()}
    return md5sRemote

def unTarModels(dirname):
    cmd = ("cat %s/xmipp_model_*.tgz | tar xzf - -i --directory=%s"
           % (dirname, dirname))
    print(blue("Uncompressing models: %s" % cmd))
    sys.stdout.flush()
    os.system(cmd)

if __name__ == '__main__':

    mode = sys.argv[1]

    if mode == 'download':
        download(*sys.argv[2:])
    elif mode == 'update':
        update(*sys.argv[2:])
    elif mode == 'upload':
        upload(*sys.argv[2:])
