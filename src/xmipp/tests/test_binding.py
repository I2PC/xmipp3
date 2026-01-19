#!/usr/bin/env python3
# **************************************************************************
# *
# * Authors:      David Maluenda (dmaluenda@cnb.csic.es)
# *      based on J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *               David Strelak (davidstrelak@gmail.com)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *
# **************************************************************************

import unittest
from tempfile import NamedTemporaryFile
import time
from  xmippLib import *
import os


class PythonBindingTest(unittest.TestCase):
    def createTmpFile(self, filename):
        return os.path.join(os.environ.get("XMIPP_TEST_DATA"), 'input', filename)

    def getTmpName(self, suffix='.xmd'):
        """ Get temporary filenames in /tmp' that will be removed
        after tests execution.
        """
        ntf = NamedTemporaryFile(dir="/tmp/", suffix=suffix)
        self.tempNames.append(ntf)
        return ntf.name

    def setUp(self):
        """This function performs all the setup stuff.
        """
        # os.chdir(self.testsPath)
        self.tempNames = []

    def tearDown(self):
        for ntf in self.tempNames:
            ntf.close()

    def test_xmipp_Euler_angles2matrix(self):
        from numpy  import array

        a = array([[ 0.70710678, 0.70710678, -0.        ],
                   [-0.70710678, 0.70710678, 0.        ],
                   [ 0., 0., 1.        ]])
        rot = 45.
        tilt = 0.
        psi = 0.
        b = Euler_angles2matrix(rot, tilt, psi)
        self.assertAlmostEqual(a.all(), b.all(), 2)


    def test_xmipp_Euler_matrix2angles(self):
        from numpy import array, arange

        a = array([[ 0.70710678, 0.70710678, -0.        ],
                   [-0.70710678, 0.70710678, 0.        ],
                   [ 0., 0., 1.        ]])
        rot = 45.
        tilt = 0.
        psi = 0.
        rot1,tilt1,psi1 = Euler_matrix2angles(a)
        if psi1 > 1:
            rot1 = psi1
            psi1=0.
        self.assertAlmostEqual(rot, rot1, 4)
        self.assertAlmostEqual(tilt, tilt1, 4)
        self.assertAlmostEqual(psi, psi1, 4)

        # test that slicing the array won't affect the result
        rot = tilt = psi = 0.0
        A = arange(16.0).reshape(4, -1)
        B = A[:3,:3]
        rot1,tilt1,psi1 = Euler_matrix2angles(B)
        self.assertAlmostEqual(48.36646, rot1, 4)
        self.assertAlmostEqual(32.31153, tilt1, 4)
        self.assertAlmostEqual(108.43494, psi1, 4)

        # test that integer values won't affect it
        rot = tilt = psi = 0.0
        D = [[0,1,2],[4,5,6],[8,9,10]]
        rot1,tilt1,psi1 = Euler_matrix2angles(D)
        self.assertAlmostEqual(48.36646, rot1, 4)
        self.assertAlmostEqual(32.31153, tilt1, 4)
        self.assertAlmostEqual(108.43494, psi1, 4)

        # test that other types fail
        self.assertRaises(TypeError, Euler_matrix2angles, B.astype(complex))

        # test that bad dimension fails
        self.assertRaises(IndexError, Euler_matrix2angles, A)
        self.assertRaises(IndexError, Euler_matrix2angles, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_xmipp_activateMathExtensions(self):
        md1 = MetaData()
        activateMathExtensions()
        id = md1.addObject()
        md1.setValue(MDL_ANGLE_ROT, 4., id)
        md1.setValue(MDL_ANGLE_TILT, 3., id)
        id = md1.addObject()
        md1.setValue(MDL_ANGLE_ROT, 9., id)
        md1.setValue(MDL_ANGLE_TILT, 5., id)
        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 2., id)
        md2.setValue(MDL_ANGLE_TILT, 3., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 3., id)
        md2.setValue(MDL_ANGLE_TILT, 5., id)
        angleRotLabel = label2Str(MDL_ANGLE_ROT)
        operateString = angleRotLabel + "= sqrt(" + angleRotLabel+")"
        md1.operate(operateString)
        self.assertEqual(md1, md2)

    def test_xmipp_errorBetween2CTFs(self):
        md1 = MetaData()
        id = md1.addObject()
        md1.setValue(MDL_CTF_SAMPLING_RATE, 1., id)
        md1.setValue(MDL_CTF_VOLTAGE, 200., id);
        md1.setValue(MDL_CTF_DEFOCUSU, 18306.250000, id);
        md1.setValue(MDL_CTF_DEFOCUSV, 16786.470000, id);
        md1.setValue(MDL_CTF_DEFOCUS_ANGLE, 30.100000, id);
        md1.setValue(MDL_CTF_CS, 2., id);
        md1.setValue(MDL_CTF_Q0, 0.07, id);

        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_CTF_SAMPLING_RATE, 1., id)
        md2.setValue(MDL_CTF_VOLTAGE, 200., id);
        md2.setValue(MDL_CTF_DEFOCUSU, 17932.700000, id);
        md2.setValue(MDL_CTF_DEFOCUSV, 16930.300000, id);
        md2.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., id);
        md2.setValue(MDL_CTF_CS, 2., id);
        md2.setValue(MDL_CTF_Q0, 0.07, id);

        error = errorBetween2CTFs(md1,md2, 256, 0.05,0.25)

        self.assertAlmostEqual(error, 5045.79,0)

    def test_xmipp_errorMaxFreqCTFs(self):
        from math import pi
        md1 = MetaData()
        id = md1.addObject()
        md1.setValue(MDL_CTF_SAMPLING_RATE, 2., id)
        md1.setValue(MDL_CTF_VOLTAGE, 300., id);
        md1.setValue(MDL_CTF_DEFOCUSU, 6000., id);
        md1.setValue(MDL_CTF_DEFOCUSV, 7500., id);
        md1.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., id);
        md1.setValue(MDL_CTF_CS, 2., id);
        md1.setValue(MDL_CTF_Q0, 0.1, id);

        resolution = errorMaxFreqCTFs(md1,pi/2.)
        self.assertAlmostEqual(resolution, 7.6852355,2)


    def test_xmipp_errorMaxFreqCTFs2D(self):
        md1 = MetaData()
        id = md1.addObject()
        md1.setValue(MDL_CTF_SAMPLING_RATE, 2., id)
        md1.setValue(MDL_CTF_VOLTAGE, 300., id);
        md1.setValue(MDL_CTF_DEFOCUSU, 10000., id);
        md1.setValue(MDL_CTF_DEFOCUSV, 5400., id);
        md1.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., id);
        md1.setValue(MDL_CTF_CS, 2., id);
        md1.setValue(MDL_CTF_Q0, 0.1, id);

        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_CTF_SAMPLING_RATE, 2., id)
        md2.setValue(MDL_CTF_VOLTAGE, 300., id);
        md2.setValue(MDL_CTF_DEFOCUSU, 5000., id);
        md2.setValue(MDL_CTF_DEFOCUSV, 5000., id);
        md2.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., id);
        md2.setValue(MDL_CTF_CS, 2., id);
        md2.setValue(MDL_CTF_Q0, 0.1, id);
        resolution = errorMaxFreqCTFs2D(md1,md2)

        self.assertAlmostEqual(resolution, 13.921659080780355,2)

    def test_FileName_compose(self):
        fn1 = FileName("kk000001.xmp")
        fn2 = FileName("")
        fn2.compose("kk", 1, "xmp")
        self.assertEqual(str(fn1), str(fn2))
        self.assertNotEqual(str(fn1) + 'kk', str(fn2))

        fn1 = FileName("000001@kk.xmp")
        fn2 = FileName("")
        fn2.compose(1, "kk.xmp")
        self.assertEqual(str(fn1), str(fn2))

        fn1 = FileName("jj@kk.xmp")
        fn2 = FileName("")
        fn2.compose("jj", "kk.xmp")
        self.assertEqual(str(fn1), str(fn2))

    def test_FileName_isInStack(self):
        fn1 = FileName("1@.xmp")
        fn2 = FileName("1.xmp")
        self.assertTrue (fn1.isInStack())
        self.assertFalse(fn2.isInStack())

    def test_FileName_isMetaData(self):
        imgPath = self.createTmpFile("smallStack.stk")
        fn1 = FileName(imgPath)
        result = fn1.isMetaData()
        self.assertFalse(result)

        imgPath = self.createTmpFile("test.xmd")
        fn2 = FileName(imgPath)
        self.assertTrue (fn2.isMetaData())

    @unittest.skip("reference data not available")
    def test_Image_add(self):
        stackPath = self.createTmpFile("smallStack.stk")
        img1 = Image("1@" + stackPath)
        img2 = Image("2@" + stackPath)
        sum = img1 + img2
        sum2 = Image("1@" + stackPath)
        sum2.inplaceAdd(img2)

        sumRef = Image(self.createTmpFile("sum.spi"))
        self.assertEqual(sum, sumRef)
        self.assertEqual(sum2, sumRef)

        img1 += img2
        self.assertEqual(sum, img1)
        img1 += img2
        self.assertNotEqual(sum, img1)

    def test_Image_multiplyDivide(self):
        imgPath = self.createTmpFile("singleImage.spi")
        img1 = Image(imgPath)
        img2 = Image(imgPath)
        imgSum = Image(imgPath)
        imgSum2 = Image(imgPath)

        for i in range(2):
            imgSum += img1
            imgSum2.inplaceAdd(img1)

        imgMult = img1 * 3
        imgMult2 = Image(imgPath)
        imgMult2.inplaceMultiply(3)

        self.assertEqual(imgSum, imgMult)
        self.assertEqual(imgSum, imgSum2)
        self.assertEqual(imgSum, imgMult2)

        img1 *= 3
        self.assertEqual(imgSum, img1)

        imgSum2.inplaceDivide(3.)
        # TODO: division does not work
        # imgSum /= 3.
        imgSum *= 0.3333333

        self.assertEqual(imgSum, imgSum2)
        self.assertNotEqual(imgSum, img1)
        self.assertEqual(imgSum, img2)

    def test_Image_multiplyInplace(self):
        from numpy  import array

        def _createImage(data):
            img = Image()
            img.setDataType(DT_FLOAT)
            img.resize(3, 3)
            img.setData(data)
            return img

        img1 = _createImage(array([[2., 0., 0.],
                                   [2., 0., 0.],
                                   [2., 0., 0.]]))
        img2 = _createImage(array([[3., 0., 0.],
                                   [3., 0., 0.],
                                   [3., 0., 0.]]))
        img3 = _createImage(array([[6., 0., 0.],
                                   [6., 0., 0.],
                                   [6., 0., 0.]]))
        img1.inplaceMultiply(3.)
        self.assertEqual(img1, img3)

        img1 = _createImage(array([[2., 0., 0.],
                                   [2., 0., 0.],
                                   [2., 0., 0.]]))
        img1.inplaceMultiply(img2)
        self.assertEqual(img1, img3)

    def test_Image_compare(self):
        imgPath = self.createTmpFile("singleImage.spi")
        img1 = Image()
        img1.read(imgPath)
        img2 = Image(imgPath)
        # Test that image is equal to itself
        self.assertEqual(img1, img2)
        # Test different images
        imgPath = "1@" + self.createTmpFile("smallStack.stk")
        img2.read(imgPath)
        self.assertNotEqual(img1, img2)

    def test_Image_computeStatistics(self):
        stackPath = self.createTmpFile("smallStack.stk")
        img1 = Image("1@" + stackPath)
        mean, dev, min, max = img1.computeStats()
        self.assertAlmostEqual(mean, -0.000360, 5)
        self.assertAlmostEqual(dev, 0.105687, 5)
        self.assertAlmostEqual(min, -0.415921, 5)
        self.assertAlmostEqual(max, 0.637052, 5)

    def test_Image_equal(self):
        img = Image()
        img2 = Image()
        img.setDataType(DT_FLOAT)
        img2.setDataType(DT_FLOAT)
        img.resize(3, 3)
        img2.resize(3, 3)
        img.setPixel(0, 0, 1, 1, 1.)
        img2.setPixel(0, 0, 1, 1, 1.01)

        self.assertFalse(img.equal(img2, 0.005))
        self.assertTrue(img.equal(img2, 0.1))

    def test_Image_getData(self):
        from numpy import array
        img1 = Image(self.createTmpFile("singleImage.spi"))
        Z = img1.getData()
        Zref = array([[-0.27623099,-0.13268562, 0.32305956],
                      [ 1.07257104, 0.73135602, 0.49813408],
                      [ 0.90717429, 0.6812411, -0.09380955]])
        self.assertEqual(Z.all(), Zref.all())

    def test_Image_initConstant(self):
        imgPath = self.createTmpFile("singleImage.spi")
        img = Image(imgPath)
        img.initConstant(1.)
        for i in range(0, 3):
            for j in range (0, 3):
                p = img.getPixel(0, 0, i, j)
                self.assertAlmostEqual(p, 1.0)

    def test_Image_initRandom(self):
        imgPath = self.createTmpFile("singleImage.spi")
        img = Image(imgPath)
        img.resize(1024, 1024)
        img.initRandom(0., 1., XMIPP_RND_GAUSSIAN)
        mean, dev, min, max = img.computeStats()
        self.assertAlmostEqual(mean, 0., 2)
        self.assertAlmostEqual(dev, 1., 2)

    @unittest.skip("reference data not available")
    def test_Image_minus(self):
        pathSum = self.createTmpFile("sum.spi")
        imgAdd = Image(pathSum)
        path1 = "1@" + self.createTmpFile("smallStack.stk")
        img1 = Image(path1)
        path2 = "2@" + self.createTmpFile("smallStack.stk")
        img2 = Image(path2)

        imgAdd -= img2
        self.assertEqual(img1, imgAdd)

    @unittest.skip("reference data not available")
    def test_Image_project(self):
        vol=Image(self.createTmpFile('progVol.vol'))
        vol.convert2DataType(DT_DOUBLE)
        proj=projectVolumeDouble(vol,0.,0.,0.)
        self.assertEqual(1,1)

    @unittest.skip("reference data not available")
    def test_Image_projectFourier(self):
        vol = Image(self.createTmpFile('progVol.vol'))
        vol.convert2DataType(DT_DOUBLE)
        proj = Image()
        proj.setDataType(DT_DOUBLE)

        padding = 2
        maxFreq = 0.5
        splineDegree = 2
        vol.write('/tmp/vol1.vol')
        fp = FourierProjector(vol, padding, maxFreq, splineDegree)
        fp.projectVolume(proj, 0, 0, 0)
        proj.write('/tmp/kk.spi')

    @unittest.skip("reference data not available")
    def test_Image_read(self):
        imgPath = self.createTmpFile("tinyImage.spi")
        img = Image(imgPath)
        count = 0.
        for i in range(0, 3):
            for j in range (0, 3):
                p = img.getPixel(0, 0, i, j)
                self.assertAlmostEqual(p, count)
                count += 1.

    def test_Image_read_header(self):
        imgPath = self.createTmpFile("singleImage.spi")
        img = Image()
        img.read(imgPath, HEADER)

        (x, y, z, n) = img.getDimensions()

        self.assertEqual(x, 64)
        self.assertEqual(y, 64)
        self.assertEqual(z, 1)
        self.assertEqual(n, 1)

    @unittest.skip("reference data not available")
    def test_Image_readApplyGeo(self):
        imgPath = self.createTmpFile("tinyRotated.spi")
        img = Image(imgPath)
        imgPath = self.createTmpFile("tinyImage.spi")
        md = MetaData()
        id = md.addObject()
        md.setValue(MDL_IMAGE, imgPath, id)
        md.setValue(MDL_ANGLE_PSI, 90., id)
        img2 = Image()
        img2.readApplyGeo(md, id)
        self.assertEqual(img, img2)

    def test_Image_setDataType(self):
        img = Image()
        img.setDataType(DT_FLOAT)
        img.resize(3, 3)
        img.setPixel(0, 0, 1, 1, 1.)
        img.computeStats()
        mean, dev, min, max = img.computeStats()
        self.assertAlmostEqual(mean, 0.111111, 5)
        self.assertAlmostEqual(dev, 0.314270, 5)
        self.assertAlmostEqual(min, 0., 5)
        self.assertAlmostEqual(max, 1., 5)

    def test_Image_mirrorY(self):
        img = Image()
        img.setDataType(DT_FLOAT)
        imgY = Image()
        imgY.setDataType(DT_FLOAT)
        dim=3
        img.resize(dim, dim)
        imgY.resize(dim, dim)
        for i in range(0,dim):
            for j in range(0,dim):
                img.setPixel(0, 0, i, j, 1.*dim*i+j)
        img.mirrorY();
        for i in range(0,dim):
            for j in range(0,dim):
                imgY.setPixel(0, 0, dim -1 -i, j, 1.*dim*i+j)
        self.assertEqual(img, imgY)

    def test_Image_applyTransforMatScipion(self):
        img = Image()
        img2 = Image()
        img.setDataType(DT_FLOAT)
        img2.setDataType(DT_FLOAT)
        dim=3
        img.resize(dim, dim)
        img2.resize(dim, dim)
        for i in range(0,dim):
            for j in range(0,dim):
                img.setPixel(0, 0, dim -1 -i, j, 1.*dim*i+j)
        A=[0.,-1.,0.,0.,
           1.,0.,0.,0.,
           0.,0.,1.,1.]
        img2.setPixel(0, 0, 0,0,0)
        img2.setPixel(0, 0, 0,1,3.)
        img2.setPixel(0, 0, 0,2,6.)

        img2.setPixel(0, 0, 1,0,1.)
        img2.setPixel(0, 0, 1,1,4.)
        img2.setPixel(0, 0, 1,2,7.)

        img2.setPixel(0, 0, 2,0,2.)
        img2.setPixel(0, 0, 2,1,5.)
        img2.setPixel(0, 0, 2,2,8.)

        img.applyTransforMatScipion(A)
        for i in range(0, dim):
            for j in range (0, dim):
                p1 = img.getPixel(0, 0, i, j)
                p2 = img2.getPixel(0, 0, i, j)
                self.assertAlmostEqual(p1, p2)

    @unittest.skip("reference data not available")
    def test_Metadata_getValue(self):
        mdPath = self.createTmpFile("test.xmd")
        mdPath2 = self.createTmpFile("proj_ctf_1.stk")
        mD = MetaData(mdPath)
        img = mD.getValue(MDL_IMAGE, 1)
        self.assertEqual(img, '000001@' + mdPath2)
        defocus = mD.getValue(MDL_CTF_DEFOCUSU, 2)
        self.assertAlmostEqual(defocus, 200)
        count = mD.getValue(MDL_COUNT, 3)
        self.assertEqual(count, 30)
        list = mD.getValue(MDL_CLASSIFICATION_DATA, 1)
        self.assertEqual(list, [1.0, 2.0, 3.0])
        ref = mD.getValue(MDL_REF3D, 2)
        self.assertEqual(ref, 2)

    @unittest.skip("reference data not available")
    def test_Metadata_importObjects(self):
        mdPath = self.createTmpFile("test.xmd")
        mD = MetaData(mdPath)
        mDout = MetaData()
        mDout.importObjects(mD, MDValueEQ(MDL_REF3D, -1))
        mdPath = self.createTmpFile("importObject.xmd")
        mD = MetaData(mdPath)
        self.assertEqual(mD, mDout)

    @unittest.skip("reference data not available")
    def test_Metadata_iter(self):
        mdPath = self.createTmpFile("test.xmd")
        mD = MetaData(mdPath)
        i = 1
        for id in mD:
            img = mD.getValue(MDL_IMAGE, id)
            expImg = self.createTmpFile("test.xmd")
            expImg = ("00000%d@" % i) + self.createTmpFile("proj_ctf_1.stk")
            self.assertEqual(img, expImg)
            i += 1

    @unittest.skip("reference data not available")
    def test_Metadata_existsBlockInMetaDataFile(self):
        mdPath = "b2@"+self.createTmpFile("testBlock.xmd")
        self.assertTrue(existsBlockInMetaDataFile(mdPath))
        mdPath = "nonexisting@"+self.createTmpFile("testBlock.xmd")
        self.assertFalse(existsBlockInMetaDataFile(mdPath))
        mdPath = "gggg@"+self.createTmpFile("testBlock.xmd")
        self.assertRaises(Exception,existsBlockInMetaDataFile(mdPath))

    def test_Metadata_operate(self):
        md = MetaData()
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 1., id)
        md.setValue(MDL_ANGLE_TILT, 2., id)
        md.setValue(MDL_ANGLE_PSI, 3., id)
        md.setValue(MDL_ANGLE_ROT2, 6., id)
        md.setValue(MDL_ANGLE_TILT2, 5., id)
        md.setValue(MDL_ANGLE_PSI2, 4., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 11., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        md.setValue(MDL_ANGLE_PSI, 13., id)
        md.setValue(MDL_ANGLE_ROT2, 16., id)
        md.setValue(MDL_ANGLE_TILT2, 15., id)
        md.setValue(MDL_ANGLE_PSI2, 14., id)

        md2 = MetaData(md)
        anglePsiLabel = label2Str(MDL_ANGLE_PSI)
        angleRotLabel = label2Str(MDL_ANGLE_ROT)
        operateString = angleRotLabel + "=3*" + angleRotLabel
        operateString += "," + anglePsiLabel + "=2*" + anglePsiLabel
        md.operate(operateString)
        for id in md2:
            md2.setValue(MDL_ANGLE_ROT, md2.getValue(MDL_ANGLE_ROT, id) * 3., id);
            md2.setValue(MDL_ANGLE_PSI, md2.getValue(MDL_ANGLE_PSI, id) * 2., id);
        self.assertEqual(md, md2)

    def test_xmipp_addLabelAlias(self):
        tmpFileName = '/tmp/test_pythoninterface_readPlain1.xmd'
        line1="""# XMIPP_STAR_1 *
#
data_images

loop_
_rlnVoltage #1
_rlnDefocusU #2
  200.0 10000.0
  201.0 10000.0"""
        f = open(tmpFileName,'w')
        f.write(line1)
        f.flush()
        f.close()
        addLabelAlias(MDL_CTF_VOLTAGE,"rlnVoltage")
        addLabelAlias(MDL_CTF_DEFOCUSU,"rlnDefocusU")
        md=MetaData(tmpFileName)
        md2=MetaData()
        id = md2.addObject()
        md2.setValue(MDL_CTF_VOLTAGE, 200.0, id)
        md2.setValue(MDL_CTF_DEFOCUSU, 10000.0, id)
        id = md2.addObject()
        md2.setValue(MDL_CTF_VOLTAGE, 201.0, id)
        md2.setValue(MDL_CTF_DEFOCUSU, 10000.0, id)
        self.assertEqual(md, md2)
        os.remove(tmpFileName)

    def test_Metadata_renameColumn(self):
        md = MetaData()
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 1., id)
        md.setValue(MDL_ANGLE_TILT, 2., id)
        md.setValue(MDL_ANGLE_PSI, 3., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 11., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        md.setValue(MDL_ANGLE_PSI, 13., id)

        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT2, 1., id)
        md2.setValue(MDL_ANGLE_TILT2, 2., id)
        md2.setValue(MDL_ANGLE_PSI2, 3., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT2, 11., id)
        md2.setValue(MDL_ANGLE_TILT2, 12., id)
        md2.setValue(MDL_ANGLE_PSI2, 13., id)

        md.renameColumn(MDL_ANGLE_ROT,MDL_ANGLE_ROT2)
        md.renameColumn(MDL_ANGLE_TILT,MDL_ANGLE_TILT2)
        md.renameColumn(MDL_ANGLE_PSI,MDL_ANGLE_PSI2)
        self.assertEqual(md, md2)

        md.clear()
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 1., id)
        md.setValue(MDL_ANGLE_TILT, 2., id)
        md.setValue(MDL_ANGLE_PSI, 3., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 11., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        md.setValue(MDL_ANGLE_PSI, 13., id)
        oldLabel=[MDL_ANGLE_ROT,MDL_ANGLE_TILT,MDL_ANGLE_PSI]
        newLabel=[MDL_ANGLE_ROT2,MDL_ANGLE_TILT2,MDL_ANGLE_PSI2]
        md.renameColumn(oldLabel,newLabel)
        self.assertEqual(md, md2)

    def test_Metadata_sort(self):
        md = MetaData()
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 1., id)
        md.setValue(MDL_ANGLE_TILT, 2., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 11., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 5., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 6., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)

        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 11., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 6., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 5., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 1., id)
        md2.setValue(MDL_ANGLE_TILT, 2., id)

        mdAux=MetaData(md)
        mdAux.sort(MDL_ANGLE_ROT,False)
        self.assertEqual(mdAux, md2)

        mdAux=MetaData(md)
        mdAux.sort(MDL_ANGLE_ROT,False,2)#limit
        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 11., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 6., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        self.assertEqual(mdAux, md2)

        mdAux=MetaData(md)
        mdAux.sort(MDL_ANGLE_ROT,False,2,1)#limit && offset
        md2 = MetaData()
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 6., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        id = md2.addObject()
        md2.setValue(MDL_ANGLE_ROT, 5., id)
        md2.setValue(MDL_ANGLE_TILT, 12., id)
        self.assertEqual(mdAux, md2)

    def test_Metadata_joinNatural(self):
        md = MetaData()
        md2 = MetaData()
        mdout = MetaData()
        listOrig = [1.0, 2.0, 3.0]
        for i in range(1, 4):
            id = md.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_COUNT, (i * 10), id)
        for i in range(1, 3):
            id = md2.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            md2.setValue(MDL_IMAGE, img, id)
            md2.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md2.setValue(MDL_ANGLE_PSI, 1., id)
        mdout.joinNatural(md, md2)

        md.clear()
        for i in range(1, 3):
            id = md.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_COUNT, (i * 10), id)
            md.setValue(MDL_ANGLE_PSI, 1., id)
        self.assertEqual(mdout, md)

    def test_Metadata_intersect(self):
        md = MetaData()
        md2 = MetaData()
        mdout = MetaData()
        listOrig = [1.0, 2.0, 3.0]
        for i in range(1, 4):
            id = md.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_COUNT, (i * 10), id)
            md.setValue(MDL_ANGLE_PSI, 1., id)
        for i in range(1, 3):
            id = md2.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            md2.setValue(MDL_IMAGE, img, id)
        md.intersection (md2, MDL_IMAGE)

        for i in range(1, 3):
            id = mdout.addObject()
            img = '%06d@pythoninterface/proj_ctf_1.stk' % i
            mdout.setValue(MDL_IMAGE, img, id)
            mdout.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            mdout.setValue(MDL_COUNT, (i * 10), id)
            mdout.setValue(MDL_ANGLE_PSI, 1., id)

        self.assertEqual(mdout, md)

    @unittest.skip("reference data not available")
    def test_Metadata_read(self):
        '''This test should produce the following metadata, which is the same of 'test.xmd'
        data_
        loop_
         _image
         _CTFModel
         _CTF_Defocus_U
         _count
         _classificationData
         _ref3d
         000001@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -100.000000         10 ' 1.000000     2.000000     3.000000 '         -1
         000002@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam   200.000000         20 ' 1.000000     4.000000     9.000000 '          2
         000003@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -300.000000         30 ' 1.000000     8.000000    27.000000 '         -3
        '''
        mdPath = self.createTmpFile("test.xmd")
        mdRef = MetaData(mdPath)
        md = MetaData()
        ii = -1
        listOrig = [1.0, 2.0, 3.0]
        for i in range(1, 4):
            id = md.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_CTF_DEFOCUSU, (i * ii * 100.0), id)
            md.setValue(MDL_COUNT, (i * 10), id)
            list = [x ** i for x in listOrig]
            md.setValue(MDL_CLASSIFICATION_DATA, list, id)
            md.setValue(MDL_REF3D, (i * ii), id)
            ii *= -1

        tmpFileName = createTmpFile('test_pythoninterface_read_tmp.xmd')
        md.write(tmpFileName)

        md2 = MetaData()
        md2.read(tmpFileName)
        os.remove(tmpFileName)
        self.assertEqual(md, md2)

    @unittest.skip("reference data not available")
    def test_Metadata_read_with_labels(self):
        '''This test should produce the following metadata, which is the same of 'test.xmd'
        data_
        loop_
         _image
         _CTFModel
         _CTF_Defocus_U
         _count
         _angleComparison
         _ref3d
         000001@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -100.000000         10 ' 1.000000     2.000000     3.000000 '         -1
         000002@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam   200.000000         20 ' 1.000000     4.000000     9.000000 '          2
         000003@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -300.000000         30 ' 1.000000     8.000000    27.000000 '         -3
        '''
        mdPath = self.createTmpFile("test.xmd")
        mdRef = MetaData(mdPath)
        md = MetaData()
        ii = -1
        listOrig = [1.0, 2.0, 3.0]
        for i in range(1, 4):
            id = md.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_CTF_DEFOCUSU, (i * ii * 100.0), id)
            md.setValue(MDL_COUNT, (i * 10), id)
            list = [x ** i for x in listOrig]
            md.setValue(MDL_CLASSIFICATION_DATA, list, id)
            md.setValue(MDL_REF3D, (i * ii), id)
            ii *= -1

        tmpFileName = self.createTmpFile('test_pythoninterface_read_tmp.xmd')
        md.write(tmpFileName)

        mdList = [MDL_IMAGE, MDL_COUNT]
        md.read(tmpFileName, mdList)
        os.remove(tmpFileName)

        md2 = MetaData()
        for i in range(1, 4):
            id = md2.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % i
            md2.setValue(MDL_IMAGE, img, id)
            md2.setValue(MDL_COUNT, (i * 10), id)
        self.assertEqual(md, md2)

    @unittest.skip("reference data not available")
    def test_Metadata_setColumnFormat(self):
        '''MetaData_setValues'''
        '''This test should produce the following metadata, which is the same of 'test_row.xmd'
        data_
         _image 000001@Images/proj_ctf_1.stk
         _CTFModel CTFs/10.ctfparam
        '''
        mdPath = self.createTmpFile("test_row.xmd")
        mdRef = MetaData(mdPath)
        md = MetaData()
        ii = -1
        listOrig = [1.0, 2.0, 3.0]
        id = md.addObject()
        img = '000001@Images/proj_ctf_1.stk'
        md.setValue(MDL_IMAGE, img, id)
        md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)

        md.setColumnFormat(False)
        rowFileName = '/tmp/test_row_tmp.xmd'
        md.write(rowFileName)

        equalBool = binaryFileComparison(mdPath, rowFileName)
        self.assertEqual(equalBool, True)
        os.remove(rowFileName)

    @unittest.skip("reference data not available")
    def test_Metadata_setValue(self):
        '''MetaData_setValues'''
        '''This test should produce the following metadata, which is the same of 'test.xmd'
        data_
        loop_
         _image
         _CTFModel
         _CTF_Defocus_U
         _count
         _angleComparison
         _ref3d
         000001@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -100.000000         10 [     1.000000     2.000000     3.000000 ]         -1
         000002@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam   200.000000         20 [     1.000000     4.000000     9.000000 ]          2
         000003@pythoninterface/proj_ctf_1.stk  CTFs/10.ctfparam  -300.000000         30 [     1.000000     8.000000    27.000000 ]         -3
        '''
        mdPath = self.createTmpFile("test.xmd")
        mdRef = MetaData(mdPath)
        md = MetaData()
        ii = -1
        listOrig = [1.0, 2.0, 3.0]
        for i in range(1, 4):
            id = md.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % i
            md.setValue(MDL_IMAGE, img, id)
            md.setValue(MDL_CTF_MODEL, 'CTFs/10.ctfparam', id)
            md.setValue(MDL_CTF_DEFOCUSU, (i * ii * 100.0), id)
            md.setValue(MDL_COUNT, (i * 10), id)
            list = [x ** i for x in listOrig]
            md.setValue(MDL_CLASSIFICATION_DATA, list, id)
            md.setValue(MDL_REF3D, (i * ii), id)
            ii *= -1


        self.assertEqual(mdRef, md)
        print("THIS IS NOT AN ERROR: we are testing exceptions: an error message should appear regarding count and DOUBLE")
        self.assertRaises(XmippError, md.setValue, MDL_COUNT, 5.5, 1)


    def test_Metadata_compareTwoMetadataFiles(self):
        fn = self.getTmpName()
        fn2 = self.getTmpName()
        fn3 = self.getTmpName()

        auxMd = MetaData()
        auxMd2 = MetaData()

        auxMd.setValue(MDL_IMAGE, "image_1.xmp", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_2.xmp", auxMd.addObject())

        auxMd.write("block_000000@" + fn, MD_OVERWRITE)
        auxMd.clear()
        auxMd.setValue(MDL_IMAGE, "image_data_1_1.xmp", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_data_1_2.xmp", auxMd.addObject())
        auxMd.write("block_000001@" + fn, MD_APPEND)
        auxMd.clear()
        auxMd.setValue(MDL_IMAGE, "image_data_2_1.xmp", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_data_2_2.xmp", auxMd.addObject())
        auxMd.write("block_000000@" + fn2, MD_OVERWRITE)
        auxMd.clear()
        auxMd.setValue(MDL_IMAGE, "image_data_A_1.xmp", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_data_A_2.xmp", auxMd.addObject())
        auxMd.write("block_000001@" + fn2, MD_APPEND)
        auxMd.clear()

        self.assertFalse(compareTwoMetadataFiles(fn, fn2))
        self.assertTrue(compareTwoMetadataFiles(fn, fn))

        sfnFN = FileName(fn)
        sfn2FN = FileName(fn2)
        self.assertFalse(compareTwoMetadataFiles(sfnFN, sfn2FN))
        self.assertTrue(compareTwoMetadataFiles(sfnFN, sfnFN))

        auxMd.setValue(MDL_IMAGE, "image_1.xmpSPACE", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_2.xmp", auxMd.addObject())
        auxMd.write("block_000000@" + fn2, MD_OVERWRITE)
        auxMd.clear()
        auxMd.setValue(MDL_IMAGE, "image_data_1_1.xmp", auxMd.addObject())
        auxMd.setValue(MDL_IMAGE, "image_data_1_2.xmp", auxMd.addObject())
        auxMd.write("block_000001@" + fn2, MD_APPEND)

        command = "sed 's/SPACE/ /g' " + fn2 + ">" + fn3
        os.system(command)

        self.assertTrue(compareTwoMetadataFiles(fn, fn3))

    def test_Metadata_agregate(self):
        mdOut = MetaData()
        md    = MetaData()
        mdResult = MetaData()
        for linea in range(1, 101):
            id = md.addObject()
            md.setValue(MDL_IMAGE, '%06d@proj.stk' % linea, id)
            md.setValue(MDL_XCOOR, int(linea%3), id)
        mdOut.aggregate(md, AGGR_COUNT, MDL_XCOOR, MDL_XCOOR, MDL_COUNT)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 0, id)
        mdResult.setValue(MDL_COUNT, 33, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 1, id)
        mdResult.setValue(MDL_COUNT, 34, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 2, id)
        mdResult.setValue(MDL_COUNT, 33, id)

        self.assertEqual(mdOut, mdResult)

    def test_Metadata_aggregateMdGroupBy(self):
        mdOut = MetaData()
        md    = MetaData()
        mdResult = MetaData()
        for linea in range(1, 101):
            id = md.addObject()
            md.setValue(MDL_IMAGE, '%06d@proj.stk' % linea, id)
            md.setValue(MDL_XCOOR, int(linea%3), id)
            md.setValue(MDL_YCOOR, int(linea%2), id)
        mdOut.aggregateMdGroupBy(md, AGGR_COUNT, [MDL_XCOOR,MDL_YCOOR], MDL_XCOOR, MDL_COUNT)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 0, id)
        mdResult.setValue(MDL_YCOOR, 0, id)
        mdResult.setValue(MDL_COUNT, 16, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 0, id)
        mdResult.setValue(MDL_YCOOR, 1, id)
        mdResult.setValue(MDL_COUNT, 17, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 1, id)
        mdResult.setValue(MDL_YCOOR, 0, id)
        mdResult.setValue(MDL_COUNT, 17, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 1, id)
        mdResult.setValue(MDL_YCOOR, 1, id)
        mdResult.setValue(MDL_COUNT, 17, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 2, id)
        mdResult.setValue(MDL_YCOOR, 0, id)
        mdResult.setValue(MDL_COUNT, 17, id)
        id = mdResult.addObject()
        mdResult.setValue(MDL_XCOOR, 2, id)
        mdResult.setValue(MDL_YCOOR, 1, id)
        mdResult.setValue(MDL_COUNT, 16, id)

        self.assertEqual(mdOut, mdResult)

    def test_Metadata_fillExpand(self):
        '''MetaData_fillExpand'''
        ctf = MetaData()
        objId = ctf.addObject()
        ctf.setColumnFormat(False)
        # Create ctf param metadata 1
        u1, v1 = 10000.0, 10001.0
        u2, v2 = 20000.0, 20001.0

        ctf.setValue(MDL_CTF_DEFOCUSU, u1, objId)
        ctf.setValue(MDL_CTF_DEFOCUSV, v1, objId)
        ctfFn1 = self.getTmpName(suffix='_ctf1.param')
        ctf.write(ctfFn1)

        # Create ctf param metadata 2
        ctf.setValue(MDL_CTF_DEFOCUSU, u2, objId)
        ctf.setValue(MDL_CTF_DEFOCUSV, v2, objId)
        ctfFn2 = self.getTmpName(suffix='_ctf2.param')
        ctf.write(ctfFn2)

        # True values
        ctfs = [ctfFn1, ctfFn1, ctfFn2]
        defocusU = [u1, u1, u2]
        defocusV = [v1, v1, v2]
        # Create image metadata
        md1 = MetaData()
        for i in range(3):
            objId = md1.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % (i + 1)
            md1.setValue(MDL_IMAGE, img, objId)
            md1.setValue(MDL_CTF_MODEL, ctfs[i], objId)

        # Create the expected metadata after call fillExpand
        md2 = MetaData()
        for i in range(3):
            objId = md2.addObject()
            img = '00000%i@pythoninterface/proj_ctf_1.stk' % (i + 1)
            md2.setValue(MDL_IMAGE, img, objId)
            md2.setValue(MDL_CTF_MODEL, ctfs[i], objId)
            md2.setValue(MDL_CTF_DEFOCUSU, defocusU[i], objId)
            md2.setValue(MDL_CTF_DEFOCUSV, defocusV[i], objId)
        md1.fillExpand(MDL_CTF_MODEL)
        self.assertEqual(md1, md2)

    def test_metadata_stress(self):
        numberMetadatas=10+1
        numberLines=10+1
        md = MetaData()
        fnStar   = self.getTmpName('.xmd')
        fnSqlite = self.getTmpName('.sqlite')
        outFileName=""
        # fill buffers so comparison is fair
        # if you do not fill buffers first meassurements will be much faster
        tmpFn="/tmp/deleteme.xmd"
        timestamp1 = time.time()
        if os.path.exists(tmpFn):
            os.remove(tmpFn)
        for met in range(1, numberMetadatas):
            outFileName = "b%06d@%s" % (met,tmpFn)
            for lin in range(1, numberLines):
                id = md.addObject()

                md.setValue(MDL_IMAGE, '%06d@proj.stk' % met, id)
                md.setValue(MDL_XCOOR,   lin+met*(numberLines-1), id)
                md.setValue(MDL_YCOOR, 1+lin+met*(numberLines-1), id)
            md.write(outFileName,MD_APPEND)
            md.clear()
        timestamp2 = time.time()

        timestamp1 = time.time()
        for met in range(1, numberMetadatas):
            outFileName = "b%06d@%s" % (met,fnSqlite)
            for lin in range(1, numberLines):
                id = md.addObject()

                md.setValue(MDL_IMAGE, '%06d@proj.stk' % met, id)
                md.setValue(MDL_XCOOR,   lin+met*(numberLines-1), id)
                md.setValue(MDL_YCOOR, 1+lin+met*(numberLines-1), id)
            md.write(outFileName,MD_APPEND)
            md.clear()
        timestamp2 = time.time()
        print("\nwrite(append) sqlite took %.2f seconds" % (timestamp2 - timestamp1))
        timestamp1 = time.time()
        for met in range(1, numberMetadatas):
            outFileName = "b%06d@%s" % (met,fnStar)
            for lin in range(1, numberLines):
                id = md.addObject()

                md.setValue(MDL_IMAGE, '%06d@proj.stk' % met, id)
                md.setValue(MDL_XCOOR,   lin+met*(numberLines-1), id)
                md.setValue(MDL_YCOOR, 1+lin+met*(numberLines-1), id)
            md.write(outFileName,MD_APPEND)
            md.clear()
        timestamp2 = time.time()
        print("\nwrite(append) star took %.2f seconds" % (timestamp2 - timestamp1))
        self.assertEqual(1, 1)

        # read sequential
        timestamp1 = time.time()
        for met in range(1, numberMetadatas):
            outFileName = "b%06d@%s" % (met,fnSqlite)
            md.read(outFileName)
            md.clear()
        timestamp2 = time.time()
        print("\nread(sequential) sqlite took %.2f seconds" % (timestamp2 - timestamp1))

        timestamp1 = time.time()
        for met in range(1, numberMetadatas):
            outFileName = "b%06d@%s" % (met,fnStar)
            md.read(outFileName)
            md.clear()
        timestamp2 = time.time()
        print("\nread(sequential) star took %.2f seconds" % (timestamp2 - timestamp1))

    def test_SymList_readSymmetryFile(self):
        a = SymList()
        a.readSymmetryFile("i3")
        self.assertEqual(True, True)

    def test_SymList_computeDistance(self):
        SL = SymList()
        SL.readSymmetryFile("i3h")
        md = MetaData()
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 1., id)
        md.setValue(MDL_ANGLE_TILT, 2., id)
        md.setValue(MDL_ANGLE_PSI, 3., id)
        md.setValue(MDL_ANGLE_ROT2, 6., id)
        md.setValue(MDL_ANGLE_TILT2, 5., id)
        md.setValue(MDL_ANGLE_PSI2, 4., id)
        id = md.addObject()
        md.setValue(MDL_ANGLE_ROT, 11., id)
        md.setValue(MDL_ANGLE_TILT, 12., id)
        md.setValue(MDL_ANGLE_PSI, 13., id)
        md.setValue(MDL_ANGLE_ROT2, 16., id)
        md.setValue(MDL_ANGLE_TILT2, 15., id)
        md.setValue(MDL_ANGLE_PSI2, 14., id)
        mdOut = MetaData(md)
        SL.computeDistance(md, False, False, False)
        id = md.firstObject()
        total = md.getValue(MDL_ANGLE_DIFF, id)
        self.assertAlmostEqual(total, 5.23652, 4)

    def test_SymList_getSymmetryMatrices(self):
        try:
            SL = SymList()
            _symList = SL.getSymmetryMatrices("C4")
        except Exception as err:
            print(err)
        m = _symList[1]
        l = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertAlmostEqual(m[1][1],l[1][1])
        self.assertAlmostEqual(m[0][0],l[0][0])
        self.assertAlmostEqual(m[1][1],0.)