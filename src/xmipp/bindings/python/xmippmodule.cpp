/***************************************************************************
 *
 * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
#include "xmippmodule.h"
#include "core/geometry.h"
#include "core/matrix2d.h"
#include "core/metadata_db.h"
#include "core/metadata_extension.h"
#include "core/metadata_sql.h"
#include "core/transformations.h"
#include "core/xmipp_image_generic.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_color.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"
#include "data/projection.h"
#include "python_fourierprojector.h"
#include "python_filename.h"
#include "python_image.h"
#include "python_program.h"
#include "python_metadata.h"
#include "python_symmetry.h"
#include "reconstruction/ctf_estimate_from_micrograph.h"
#include "numpy/arrayobject.h"

PyObject * PyXmippError;

/***************************************************************/
/*                            Global methods                   */
/***************************************************************/
PyObject *
xmipp_str2Label(PyObject *obj, PyObject *args)
{
    char * str;
    if (PyArg_ParseTuple(args, "s", &str))
        return Py_BuildValue("i", (int) MDL::str2Label(str));
    return nullptr;
}

PyObject *
xmipp_label2Str(PyObject *obj, PyObject *args)
{
    int label;
    if (PyArg_ParseTuple(args, "i", &label))
    {
        String labelStr = MDL::label2Str((MDLabel) label);
        return PyUnicode_FromString(labelStr.c_str());
    }
    return nullptr;
}

PyObject *
xmipp_colorStr(PyObject *obj, PyObject *args)
{
    char *str;
    int color;
    int attrib = BRIGHT;
    if (PyArg_ParseTuple(args, "is|i", &color, &str, &attrib))
    {
        String labelStr = colorString(str, color, attrib);
        return PyUnicode_FromString(labelStr.c_str());
    }
    return nullptr;
}

PyObject *
xmipp_labelType(PyObject *obj, PyObject *args)
{
    PyObject * input;
    PyObject* str_exc_type = nullptr;
    PyObject* pyStr = nullptr;
    if (PyArg_ParseTuple(args, "O", &input))
    {
        if (PyUnicode_Check(input))
              return Py_BuildValue("i", (int) MDL::labelType((char*)PyUnicode_AsUTF8(PyObject_Str(input))));

        else if (PyLong_Check(input))
            return Py_BuildValue("i", (int) MDL::labelType((MDLabel) PyLong_AsLong(input)));

        else
            PyErr_SetString(PyExc_TypeError,
                            "labelType: Only int or string are allowed as input");
    }
    return nullptr;
}

PyObject *
xmipp_labelHasTag(PyObject *obj, PyObject *args)
{
    PyObject * input;
    PyObject* str_exc_type = nullptr;
    PyObject* pyStr = nullptr;
    int tag;

    if (PyArg_ParseTuple(args, "Oi", &input, &tag))
    {
        MDLabel label = MDL_UNDEFINED;

        if (PyUnicode_Check(input))
            {
//              str_exc_type = PyObject_Repr(input); //Now a unicode object
//              pyStr = PyUnicode_AsEncodedString(str_exc_type, "utf-8", "Error ~");
//              label = MDL::str2Label(PyBytes_AS_STRING(pyStr));
                label = MDL::str2Label((char*)PyUnicode_AsUTF8(input));
             }
        else if (PyLong_Check(input))
            label = (MDLabel) PyLong_AsLong(input);

        if (label != MDL_UNDEFINED)
        {
            if (MDL::hasTag(label, tag))
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }

        PyErr_SetString(PyExc_TypeError,
                        "labelHasTag: Input label should be int or string");
    }
    return nullptr;
}

PyObject *
xmipp_labelIsImage(PyObject *obj, PyObject *args)
{
    PyObject * input;
    int tag = TAGLABEL_IMAGE;

    if (PyArg_ParseTuple(args, "O", &input))
    {
        MDLabel label = MDL_UNDEFINED;

        if (PyUnicode_Check(input))
              label = MDL::str2Label((char*)PyUnicode_AsUTF8(PyObject_Str(input)));

        else if (PyLong_Check(input))
            label = (MDLabel) PyLong_AsLong(input);

        if (label != MDL_UNDEFINED)
        {
            if (MDL::hasTag(label, tag))
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }

        PyErr_SetString(PyExc_TypeError,
                        "labelIsImage: Input label should be int or string");
    }
    return nullptr;
}

/* isInStack */
PyObject *
xmipp_isValidLabel(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    char *str;
    int label;
    if (PyArg_ParseTuple(args, "s", &str))
        label = MDL::str2Label(str);
    else if (PyArg_ParseTuple(args, "i", &label))
        ;
    else
        return nullptr;
    if (MDL::isValidLabel((MDLabel) label))
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

/* createEmptyFile */
//ROB: argument of this python function do not much arguments of C funcion Ndim does not exists in C
PyObject *
xmipp_createEmptyFile(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    int Xdim;
    int Ydim;
    int Zdim=1;
    size_t Ndim=1;
    DataType dataType = DT_Float;

    PyObject * input;
    if (PyArg_ParseTuple(args, "Oii|iii", &input, &Xdim, &Ydim, &Zdim,
                         &Ndim, &dataType))
    {
    try
        {

        auto inputStr = (std::string)(char*)PyUnicode_AsUTF8(PyObject_Str(input));
        inputStr += "%";
        inputStr += datatype2Str(dataType);
        createEmptyFile(inputStr, Xdim, Ydim, Zdim, Ndim, true, WRITE_REPLACE);
        Py_RETURN_NONE;
        }
     catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}
/* getImageSize */
PyObject *
xmipp_getImageSize(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyValue;

    if (PyArg_ParseTuple(args, "O", &pyValue))
    {
        try
        {

            PyObject * pyStr = PyObject_Str(pyValue);
            auto *str = (char*)PyUnicode_AsUTF8(pyStr);
            size_t xdim;
            size_t ydim;
            size_t zdim;
            size_t ndim;
            getImageSize(str, xdim, ydim, zdim, ndim);
            Py_DECREF(pyStr);
            return Py_BuildValue("kkkk", xdim, ydim, zdim, ndim);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject * xmipp_MetaDataInfo(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyValue; //Only used to skip label and value
    PyObject *pyStr = nullptr;

    if (PyArg_ParseTuple(args, "O", &pyValue))
    {
        try
        {
            MetaData *md = nullptr;
            size_t size; //number of elements in the metadata
            bool destroyMd = true;

            if (PyUnicode_Check(pyValue))
            {
                PyObject* repr = PyObject_Str(pyValue);
                auto * str = (char*)PyUnicode_AsUTF8(pyValue);
                md = new MetaDataDb();
                md->setMaxRows(1);
                md->read(str);
                size = md->getParsedLines();
            }
            else if (FileName_Check(pyValue))
            {
                md = new MetaDataDb();
                md->setMaxRows(1);
                md->read(FileName_Value(pyValue));
                size = md->getParsedLines();
            }
            else if (MetaData_Check(pyValue))
            {
                md = ((MetaDataObject*)pyValue)->metadata.get();
                destroyMd = false;
                size = md->size();
            }
            else
            {
                PyErr_SetString(PyXmippError, "Invalid argument: expected String, FileName or MetaData");
                return nullptr;
            }
            size_t xdim;
            size_t ydim;
            size_t zdim;
            size_t ndim;
            getImageSize(*md, xdim, ydim, zdim, ndim);

            if (destroyMd)
                delete md;
            return Py_BuildValue("iiikk", xdim, ydim, zdim, ndim, size);
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}/* Metadata info (from metadata filename)*/
/* check if block exists in file*/

PyObject *
xmipp_existsBlockInMetaDataFile(PyObject *obj, PyObject *args, PyObject *kwargs)
{

    PyObject *input = nullptr;
    PyObject *pyStr = nullptr;
    char *str = nullptr;
    if (PyArg_ParseTuple(args, "O", &input))
    {
        try
        {
            if ((pyStr = PyObject_Str(input)) != nullptr )
            {
                str = (char*)PyUnicode_AsUTF8(pyStr);
                if (existsBlockInMetaDataFile( (std::string) str))
                    Py_RETURN_TRUE;
                else
                    Py_RETURN_FALSE;
            }
            else
                return nullptr;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
            return nullptr;
        }
    }

    return nullptr;
}

PyObject *
xmipp_CheckImageFileSize(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename;

    if (PyArg_ParseTuple(args, "O", &filename))
    {
        try
        {
            PyObject * pyStr = PyObject_Str(filename);
            auto *str = (char*)PyUnicode_AsUTF8(pyStr);
            bool result = checkImageFileSize(str);
            Py_DECREF(pyStr);
            if (result)
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
xmipp_CheckImageCorners(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename;

    if (PyArg_ParseTuple(args, "O", &filename))
    {
        try
        {
            PyObject * pyStr = PyObject_Str(filename);
            auto * str = (char*)PyUnicode_AsUTF8(pyStr);
            bool result = checkImageCorners(str);
            Py_DECREF(pyStr);
            if (result)
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
xmipp_ImgCompare(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename1;
    PyObject *filename2;

    if (PyArg_ParseTuple(args, "OO", &filename1, &filename2))
    {
        try
        {
            PyObject * pyStr1 = PyObject_Str(filename1);
            PyObject * pyStr2 = PyObject_Str(filename2);

            auto * str1 = (char*)PyUnicode_AsUTF8(pyStr1);
            auto * str2 = (char*)PyUnicode_AsUTF8(pyStr2);
            bool result = compareImage(str1, str2);
            Py_DECREF(pyStr1);
            Py_DECREF(pyStr2);
            if (result)
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
xmipp_compareTwoFiles(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename1;
    PyObject *filename2;
    size_t offset=0;
    if (PyArg_ParseTuple(args, "OO|i", &filename1, &filename2, &offset))
    {
        try
        {
            PyObject * pyStr1 = PyObject_Str(filename1);
            PyObject * pyStr2 = PyObject_Str(filename2);

            auto * str1 = (char*)PyUnicode_AsUTF8(pyStr1);
            auto * str2 = (char*)PyUnicode_AsUTF8(pyStr2);
            bool result = compareTwoFiles(str1, str2, offset);
            Py_DECREF(pyStr1);
            Py_DECREF(pyStr2);
            if (result)
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}


PyObject *
xmipp_bsoftRemoveLoopBlock(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename1;
    PyObject *filename2;
    if (PyArg_ParseTuple(args, "OO", &filename1, &filename2))
    {
        try
        {
            PyObject * pyStr1 = PyObject_Str(filename1);
            PyObject * pyStr2 = PyObject_Str(filename2);

            auto * str1 = (char*)PyUnicode_AsUTF8(pyStr1);
            auto * str2 = (char*)PyUnicode_AsUTF8(pyStr2);
            bsoftRemoveLoopBlock(str1, str2);
            Py_DECREF(pyStr1);
            Py_DECREF(pyStr2);
			Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}


PyObject *
xmipp_bsoftRestoreLoopBlock(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *filename1;
    PyObject *filename2;
    if (PyArg_ParseTuple(args, "OO", &filename1, &filename2))
    {
        try
        {
            PyObject * pyStr1 = PyObject_Str(filename1);
            PyObject * pyStr2 = PyObject_Str(filename2);

            auto * str1 = (char*)PyUnicode_AsUTF8(pyStr1);
            auto * str2 = (char*)PyUnicode_AsUTF8(pyStr2);
            bsoftRestoreLoopBlock(str1, str2);
            Py_DECREF(pyStr1);
            Py_DECREF(pyStr2);
			Py_RETURN_TRUE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

PyObject *
xmipp_compareTwoImageTolerance(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *input1;
    PyObject *input2;
    double tolerance=0.;

    if (PyArg_ParseTuple(args, "OO|d", &input1, &input2, &tolerance))

    {
        try
        {
            size_t index1 = 0;
            size_t index2 = 0;
            char * fn1;
            char * fn2;

            // If the inputs objects are tuples, consider it (index, filename)
            if (PyTuple_Check(input1))
            {
              // Get the index and filename from the Python tuple object
              index1 = PyLong_AsSsize_t(PyTuple_GetItem(input1, 0));
              fn1 = (char*)PyUnicode_AsUTF8(PyObject_Str(PyTuple_GetItem(input1, 1)));
            }
            else
               fn1 = (char*)PyUnicode_AsUTF8(PyObject_Str(input1));

            if (PyTuple_Check(input2))
            {
              // Get the index and filename from the Python tuple object
              index2 = PyLong_AsSsize_t(PyTuple_GetItem(input2, 0));
              fn2 = (char*)PyUnicode_AsUTF8(PyObject_Str(PyTuple_GetItem(input2, 1)));
            }
            else
                fn2 = (char*)PyUnicode_AsUTF8(PyObject_Str(input2));

            bool result = compareTwoImageTolerance(fn1, fn2, tolerance, index1, index2);

            if (result)
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/***************************************************************/
/*                   Some specific utility functions           */
/***************************************************************/
/* readMetaDataWithTwoPossibleImages */
PyObject *
xmipp_readMetaDataWithTwoPossibleImages(PyObject *obj, PyObject *args,
                                        PyObject *kwargs)
{
    PyObject *pyStr;
    PyObject *pyMd; //Only used to skip label and value

    if (PyArg_ParseTuple(args, "OO", &pyStr, &pyMd))
    {
        try
        {
            if (!MetaData_Check(pyMd))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as second argument");
            else
            {
                if (PyUnicode_Check(pyStr))
                {
                    readMetaDataWithTwoPossibleImages((char*)PyUnicode_AsUTF8(pyStr),
                                                      MetaData_Value(pyMd));
                }
                else if (FileName_Check(pyStr))
                    readMetaDataWithTwoPossibleImages(FileName_Value(pyStr),
                                                      MetaData_Value(pyMd));
                else
                    PyErr_SetString(PyExc_TypeError,
                                    "Expected string or FileName as first argument");
                Py_RETURN_NONE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/* substituteOriginalImages */
PyObject *
xmipp_substituteOriginalImages(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyStrFnOrig;
    PyObject *pyStrFnOut;
    int label;
    int skipFirstBlock;

    if (PyArg_ParseTuple(args, "OOOii", &pyStrFn, &pyStrFnOrig, &pyStrFnOut,
                         &label, &skipFirstBlock))
    {
        try
        {
            FileName fn;
            FileName fnOrig;
            FileName fnOut;
            if (PyUnicode_Check(pyStrFn))
                fn = (char*)PyUnicode_AsUTF8(pyStrFn);

            else if (FileName_Check(pyStrFn))
                fn = FileName_Value(pyStrFn);
            else
                PyErr_SetString(PyExc_TypeError,
                                "Expected string or FileName as first argument");

            if (PyUnicode_Check(pyStrFnOrig))
                fnOrig = (char*)PyUnicode_AsUTF8(pyStrFnOrig);

            else if (FileName_Check(pyStrFnOrig))
                fnOrig = FileName_Value(pyStrFnOrig);
            else
                PyErr_SetString(PyExc_TypeError,
                                "Expected string or FileName as second argument");

            if (PyUnicode_Check(pyStrFnOut))
                fnOut = (char*)PyUnicode_AsUTF8(pyStrFnOut);

            else if (FileName_Check(pyStrFnOut))
                fnOut = FileName_Value(pyStrFnOut);
            else
                PyErr_SetString(PyExc_TypeError,
                                "Expected string or FileName as third argument");

            substituteOriginalImages(fn, fnOrig, fnOut, (MDLabel) label,
                                     (bool) skipFirstBlock);
            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

bool validateInputImageString(PyObject * pyImage, PyObject *pyStrFn, FileName &fn)
{
    if (!Image_Check(pyImage))
    {
        PyErr_SetString(PyExc_TypeError,
                        "bad argument: Expected Image as first argument");
        return false;
    }
    if (PyUnicode_Check(pyStrFn))
        fn = (char*)PyUnicode_AsUTF8(pyStrFn);

    else if (FileName_Check(pyStrFn))
        fn = FileName_Value(pyStrFn);
    else
    {
        PyErr_SetString(PyExc_TypeError,
                        "bad argument:Expected string or FileName as second argument");
        return false;
    }
    return true;
}

PyObject *
xmipp_compareTwoMetadataFiles(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn1;
    PyObject *pyStrFn2;
    PyObject *pyStrAux;

    if (PyArg_ParseTuple(args, "OO", &pyStrFn1, &pyStrFn2))
    {
        try
        {
            FileName fn1;
            FileName fn2;

            pyStrAux = PyObject_Str(pyStrFn1);

            if (pyStrAux != nullptr)
                fn1 = (char*)PyUnicode_AsUTF8(pyStrAux);
            else
                PyErr_SetString(PyExc_TypeError, "Expected string or FileName as first argument");

            pyStrAux = PyObject_Str(pyStrFn2);
            if (pyStrAux != nullptr)
                  fn2 = (char*)PyUnicode_AsUTF8(pyStrAux);

            else
                PyErr_SetString(PyExc_TypeError,
                                "Expected string or FileName as first argument");

            if (compareTwoMetadataFiles(fn1, fn2))
                Py_RETURN_TRUE;
            else
                Py_RETURN_FALSE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/** Some helper macros repeated in filter functions*/
#define FILTER_TRY()\
try {\
if (validateInputImageString(pyImage, pyStrFn, fn)) {\
Image<double> img;\
img.read(fn);\
MultidimArray<double> &data = MULTIDIM_ARRAY(img);\
ArrayDim idim;\
data.getDimensions(idim);

#define FILTER_CATCH()\
size_t w = dim, h = dim, &x = idim.xdim, &y = idim.ydim;\
if (x > y) h = y * (dim/x);\
else if (y > x)\
  w = x * (dim/y);\
selfScaleToSize(LINEAR, data, w, h);\
Image_Value(pyImage).setDatatype(DT_Double);\
data.resetOrigin();\
MULTIDIM_ARRAY_GENERIC(Image_Value(pyImage)).setImage(data);\
Py_RETURN_NONE;\
}} catch (XmippError &xe)\
{ PyErr_SetString(PyXmippError, xe.what());}\


/* dump metadatas to database*/
PyObject *
xmipp_dumpToFile(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyStrAux;
    FileName fn;

    if (PyArg_ParseTuple(args, "O", &pyStrFn))
    {
        pyStrAux = PyObject_Str(pyStrFn);
        if (pyStrAux != nullptr)
        {
            fn = (char*)PyUnicode_AsUTF8(pyStrAux);
            MDSql::dumpToFile(fn);
            Py_RETURN_NONE;
        }
    }
    return nullptr;
}

PyObject *
xmipp_Euler_angles2matrix(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    import_array();
    double rot;
    double tilt;
    double psi;
    if (PyArg_ParseTuple(args, "ddd", &rot,&tilt,&psi))
    {
        npy_intp dims[2];
        dims[0] = 3;
        dims[1] = 3; 
        auto * arr = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        void * data = PyArray_DATA(arr);
        Matrix2D<double> euler(3,3);
        Euler_angles2matrix(rot, tilt, psi,euler,false);
        memcpy(data, (euler.mdata), 9 * sizeof(double));
        return (PyObject*)arr;
    }
    return nullptr;
}

PyObject *
xmipp_Euler_matrix2angles(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    import_array()
    PyObject *input;
    if (PyArg_ParseTuple(args, "O", &input))
    {
        // parse python object into numpy array (Numpy/C API)
        auto dType = PyArray_ObjectType(input, NPY_FLOAT);
        auto *arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(input, dType, NPY_ARRAY_IN_ARRAY));
        if (nullptr == arr) {
            return nullptr;
        }
        if (const auto *dims = PyArray_DIMS(arr); 2 != PyArray_NDIM(arr) || (3 != dims[0]) || (3 != dims[1])) {
            PyErr_SetString(PyExc_IndexError, "2D array of size <3,3> expected");
            return nullptr;
        }
        Matrix2D<double> euler(3,3);
        // let's assume that the stride == 1
        if (PyTypeNum_ISFLOAT(dType)) {
            memcpy(euler.mdata, PyArray_DATA(arr), 9 * sizeof(double));
        } else if (PyTypeNum_ISINTEGER(dType)) {
            for (auto i = 0; i < 9; ++i) {
                euler.mdata[i] = static_cast<double>(*reinterpret_cast<int*>(PyArray_GETPTR1(arr, i)));
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Array of type 'double' or 'int' expected");
            return nullptr;
        }
        double rot;
        double tilt;
        double psi;
        Euler_matrix2angles(euler,rot, tilt, psi);
        return Py_BuildValue("fff", rot, tilt, psi);//fff three real
    }
    return nullptr;
}


PyObject *
xmipp_Euler_direction(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    double rot;
    double tilt;
    double psi;
    if (PyArg_ParseTuple(args, "ddd", &rot,&tilt,&psi))
    {
        Matrix1D<double> direction(3);
        Euler_direction(rot, tilt, psi, direction);
        return Py_BuildValue("fff", VEC_ELEM(direction, 0), VEC_ELEM(direction, 1), VEC_ELEM(direction, 2));//fff three real
    }
    return nullptr;
}

PyObject *
xmipp_alignWithZ(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    import_array();
    double x;
    double y;
    double z;
    bool homogeneous;
	auto *pyHomogeneous = Py_False;
    if (PyArg_ParseTuple(args, "ddd|O", &x,&y,&z,&pyHomogeneous))
    {
		if(PyBool_Check(pyHomogeneous))
			homogeneous = pyHomogeneous == Py_True;

		npy_intp dims[2];
		if (!homogeneous)
		{
	        dims[0] = 3;
	        dims[1] = 3; 			
		}
		else
		{
	        dims[0] = 4;
	        dims[1] = 4; 			
		}
        auto * arr = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        void * data = PyArray_DATA(arr);
        Matrix1D<double> u(3);
        Matrix2D<double> R;
        VECTOR_R3(u,x,y,z);
        alignWithZ(u, R, homogeneous);
        memcpy(data, (R.mdata), dims[0] * dims[1] * sizeof(double));
        return (PyObject*)arr;
    }
    return nullptr;
}

/* activateMathExtensions */
PyObject *
xmipp_activateMathExtensions(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        if (MDSql::activateMathExtensions())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* activateRegExtensions */
PyObject *
xmipp_activateRegExtensions(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    try
    {
        if (MDSql::activateRegExtensions())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* calculate enhanced psd and return preview*/
PyObject *
xmipp_fastEstimateEnhancedPSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyImage;
    //ImageObject *pyImage;
    double downsampling;
    int dim;
    int Nthreads;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdii", &pyImage, &pyStrFn, &downsampling, &dim, &Nthreads))
    {
        try
        {
            if (validateInputImageString(pyImage, pyStrFn, fn))
            {
                MultidimArray<double> data;
                fastEstimateEnhancedPSD(fn, downsampling, data, Nthreads);
                selfScaleToSize(xmipp_transformation::LINEAR, data, dim, dim);
                Image_Value(pyImage).setDatatype(DT_Double);
                Image_Value(pyImage).data->setImage(data);
                Py_RETURN_NONE;
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}

/** Some helper macros repeated in filter functions*/
#define FILTER_TRY()\
try {\
if (validateInputImageString(pyImage, pyStrFn, fn)) {\
Image<double> img;\
img.read(fn);\
MultidimArray<double> &data = MULTIDIM_ARRAY(img);\
ArrayDim idim;\
data.getDimensions(idim);

#undef FILTER_CATCH
#define FILTER_CATCH()\
size_t w = dim, h = dim, &x = idim.xdim, &y = idim.ydim;\
if (x > y) h = y * (dim/x);\
else if (y > x)\
  w = x * (dim/y);\
selfScaleToSize(xmipp_transformation::LINEAR, data, w, h);\
Image_Value(pyImage).setDatatype(DT_Double);\
data.resetOrigin();\
MULTIDIM_ARRAY_GENERIC(Image_Value(pyImage)).setImage(data);\
Py_RETURN_NONE;\
}} catch (XmippError &xe)\
{ PyErr_SetString(PyXmippError, xe.what());}\


/* calculate enhanced psd and return preview
* used for protocol preprocess_particles*/
PyObject *
xmipp_bandPassFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyImage;
    double w1;
    double w2;
    double raised_w;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdddi", &pyImage, &pyStrFn, &w1, &w2, &raised_w, &dim))
    {
        FILTER_TRY()
        bandpassFilter(data, w1, w2, raised_w);
        FILTER_CATCH()
    }
    return nullptr;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_gaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyImage;
    double freqSigma;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &freqSigma, &dim))
    {
        FILTER_TRY()
        gaussianFilter(data, freqSigma);
        FILTER_CATCH()
    }
    return nullptr;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_realGaussianFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyImage;
    double realSigma;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &realSigma, &dim))
    {
        FILTER_TRY()
        realGaussianFilter(data, realSigma);
        FILTER_CATCH()
    }
    return nullptr;
}

/* calculate enhanced psd and return preview
 * used for protocol preprocess_particles*/
PyObject *
xmipp_badPixelFilter(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyStrFn;
    PyObject *pyImage;
    double factor;
    int dim;
    FileName fn;

    if (PyArg_ParseTuple(args, "OOdi", &pyImage, &pyStrFn, &factor, &dim))
    {
        FILTER_TRY()
        BadPixelFilter filter;
        filter.type = BadPixelFilter::OUTLIER;
        filter.factor = factor;
        filter.apply(data);
        FILTER_CATCH()
    }
    return nullptr;
}

PyObject *
xmipp_errorBetween2CTFs(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1;
    PyObject *pyMd2;
    double minFreq=0.05;
    double maxFreq=0.25;
    size_t dim=256;

    if (PyArg_ParseTuple(args, "OO|idd"
                         ,&pyMd1, &pyMd2
                         ,&dim,&minFreq,&maxFreq))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else if (!MetaData_Check(pyMd2))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as second argument");
            else
            {
                double error = errorBetween2CTFs(MetaData_Value(pyMd1),
                                                 MetaData_Value(pyMd2),
                                                 dim,
                                                 minFreq,maxFreq);
                return Py_BuildValue("f", error);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;

}

PyObject *
xmipp_errorMaxFreqCTFs(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1;
    double phaseDiffRad;

    if (PyArg_ParseTuple(args, "Od", &pyMd1, &phaseDiffRad))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else
            {
                double resolutionA = errorMaxFreqCTFs(MetaData_Value(pyMd1),
                                                      phaseDiffRad
                                                     );
                return Py_BuildValue("f", resolutionA);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;

}

PyObject *
xmipp_errorMaxFreqCTFs2D(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pyMd1;
    PyObject *pyMd2;

    if (PyArg_ParseTuple(args, "OO", &pyMd1, &pyMd2))
    {
        try
        {
            if (!MetaData_Check(pyMd1))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as first argument");
            else if (!MetaData_Check(pyMd2))
                PyErr_SetString(PyExc_TypeError,
                                "Expected MetaData as second argument");
            else
            {
                double resolutionA = errorMaxFreqCTFs2D(MetaData_Value(pyMd1),
                                                        MetaData_Value(pyMd2)
                                                       );
                return Py_BuildValue("f", resolutionA);
            }
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;

}

/* convert to psd */
PyObject *
Image_convertPSD(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    auto *self = (ImageObject*) obj;

    if (self != nullptr)
    {
        try
        {
            auto &image = self->image;
            image->convert2Datatype(DT_Double);
            MultidimArray<double> *in;
            MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(in);
            xmipp2PSD(*in, *in, true);

            Py_RETURN_NONE;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_convertPSD

/* I2aligned=align(I1,I2) */
PyObject *
Image_align(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pimg1 = nullptr;
    PyObject *pimg2 = nullptr;
    auto * result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
	try
	{
		if (PyArg_ParseTuple(args, "OO", &pimg1, &pimg2))
		{
			auto *img1=(ImageObject *)pimg1;
			auto *img2=(ImageObject *)pimg2;

			result->image = std::make_unique<ImageGeneric>(Image_Value(img2));
			*result->image = *img2->image;

			result->image->convert2Datatype(DT_Double);
			MultidimArray<double> *mimgResult;
			MULTIDIM_ARRAY_GENERIC(*result->image).getMultidimArrayPointer(mimgResult);

			img1->image->convert2Datatype(DT_Double);
			MultidimArray<double> *mimg1;
			MULTIDIM_ARRAY_GENERIC(*img1->image).getMultidimArrayPointer(mimg1);

			//AJ testing
			MULTIDIM_ARRAY_GENERIC(*img1->image).setXmippOrigin();
			MULTIDIM_ARRAY_GENERIC(*result->image).setXmippOrigin();
			//END AJ

			Matrix2D<double> M;
			alignImagesConsideringMirrors(*mimg1, *mimgResult, M, true);
		}
	}
	catch (XmippError &xe)
	{
		PyErr_SetString(PyXmippError, xe.what());
	}
    return (PyObject *)result;
}//function Image_align

/* Apply CTF to this image */
PyObject *
Image_applyCTF(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pimg = nullptr;
    PyObject *input = nullptr;
    double Ts=1.0;
    size_t rowId;
    auto *pyReplace = Py_False;
    bool absPhase = false;

    try
    {
        PyArg_ParseTuple(args, "OOd|kO", &pimg, &input,&Ts,&rowId,&pyReplace);
        if (pimg != nullptr && input != nullptr)
        {
			if(PyBool_Check(pyReplace))
				absPhase = pyReplace == Py_True;

			PyObject *pyStr;
			if (PyUnicode_Check(input) || MetaData_Check(input))
			{
				auto *img = (ImageObject*) pimg;
				auto &image = img->image;
				image->convert2Datatype(DT_Double);
				MultidimArray<double> * mImage=nullptr;
				MULTIDIM_ARRAY_GENERIC(*image).getMultidimArrayPointer(mImage);

				// COSS: This is redundant? image->data->getMultidimArrayPointer(mImage);

				CTFDescription ctf;
				ctf.enable_CTF=true;
				ctf.enable_CTFnoise=false;
				if (MetaData_Check(input))
					ctf.readFromMetadataRow(MetaData_Value(input), rowId );
				else
			   {
				   pyStr = PyObject_Str(input);
				   FileName fnCTF = (char*)PyUnicode_AsUTF8(pyStr);
				   ctf.read(fnCTF);
			   }
				ctf.produceSideInfo();
				ctf.applyCTF(*mImage,Ts,absPhase);
				Py_RETURN_NONE;
			}
		}
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* Apply CTF to this image */
PyObject *
xmipp_getPSF(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    import_array();
    PyObject *inputCTF = nullptr;
    double Ts=0.5;
    size_t rowId=0;

    try
    {
        PyArg_ParseTuple(args, "O|dk", &inputCTF,&Ts,&rowId);
        if (inputCTF != nullptr)
        {
			PyObject *pyStr;
			if (PyUnicode_Check(inputCTF) || MetaData_Check(inputCTF))
			{
				CTFDescription ctf;
				ctf.enable_CTF=true;
				ctf.enable_CTFnoise=false;
				if (MetaData_Check(inputCTF))
					ctf.readFromMetadataRow(MetaData_Value(inputCTF), rowId );
				else
			    {
				   pyStr = PyObject_Str(inputCTF);
				   FileName fnCTF = (char*)PyUnicode_AsUTF8(pyStr);
				   ctf.read(fnCTF);
			    }
			    ctf.Tm=Ts;
				ctf.produceSideInfo();

				MultidimArray<double> psfProfile(512);
				MultidimArray<std::complex<double> > ctfProfile(256);
				double resolutionStep=1/(2*Ts*256);
				for (int k=0; k<256; k++)
				{
                   ctf.precomputeValues(k*resolutionStep,0.0);
                   dAi(ctfProfile, k)=ctf.getValueAt();
				}

                FourierTransformer fft;
                fft.inverseFourierTransform(ctfProfile, psfProfile);
                CenterFFT(psfProfile, true);

                npy_intp dims[1];
                dims[0] = MULTIDIM_SIZE(psfProfile);
                auto * arr = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                void * data = PyArray_DATA(arr);
                void *mymem = psfProfile.getArrayPointer();
                memcpy(data, mymem, MULTIDIM_SIZE(psfProfile)*sizeof(double));
                return (PyObject*)arr;
			}
		}
    }
    catch (XmippError &xe)
    {
        PyErr_SetString(PyXmippError, xe.what());
    }
    return nullptr;
}

/* projectVolumeDouble */
PyObject *
Image_projectVolumeDouble(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject *pvol = nullptr;
    ImageObject * result = nullptr;
    double rot;
    double tilt;
    double psi;

    if (PyArg_ParseTuple(args, "Oddd", &pvol, &rot,&tilt,&psi))
    {
        try
        {
            // We use the following macro to release the Python Interpreter Lock (GIL)
            // while running this C extension code and allows threads to run concurrently.
            // See: https://docs.python.org/2.7/c-api/init.html for details.
            Py_BEGIN_ALLOW_THREADS
            Projection P;
			auto *vol = (ImageObject*) pvol;
            MultidimArray<double> * mVolume;
            vol->image->data->getMultidimArrayPointer(mVolume);
            ArrayDim aDim;
            mVolume->getDimensions(aDim);
            mVolume->setXmippOrigin();
            projectVolume(*mVolume, P, aDim.xdim, aDim.ydim,rot, tilt, psi);
            result = (ImageObject*)PyObject_CallFunction((PyObject*)&ImageType, "");
            Image <double> I;
            result->image = std::make_unique<ImageGeneric>();
            result->image->setDatatype(DT_Double);
            result->image->data->setImage(MULTIDIM_ARRAY(P));
            Py_END_ALLOW_THREADS
            return (PyObject *)result;
        }
        catch (XmippError &xe)
        {
            PyErr_SetString(PyXmippError, xe.what());
        }
    }
    return nullptr;
}//function Image_projectVolumeDouble


static PyMethodDef
xmipp_methods[] =
    {
        { "getBlocksInMetaDataFile",
          xmipp_getBlocksInMetaDataFile, METH_VARARGS,
          "return list with metadata blocks in a file" },
        { "label2Str", xmipp_label2Str, METH_VARARGS,
          "Convert MDLabel to string" },
        { "colorStr", xmipp_colorStr, METH_VARARGS,
          "Create a string with color characters sequence for print in console" },
        { "labelType", xmipp_labelType, METH_VARARGS,
          "Return the type of a label" },
        { "labelHasTag", xmipp_labelHasTag, METH_VARARGS,
          "Return the if the label has a specific tag" },
        { "labelIsImage", xmipp_labelIsImage, METH_VARARGS,
          "Return if the label has the TAGLABEL_IMAGE tag" },
        { "str2Label", xmipp_str2Label, METH_VARARGS,
          "Convert an string to MDLabel" },
        { "isValidLabel", (PyCFunction) xmipp_isValidLabel,
          METH_VARARGS,
          "Check if the label is a valid one" },
        { "MDValueRelational",
          (PyCFunction) xmipp_MDValueRelational,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueEQ", (PyCFunction) xmipp_MDValueEQ,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueNE", (PyCFunction) xmipp_MDValueNE,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueLT", (PyCFunction) xmipp_MDValueLT,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueLE", (PyCFunction) xmipp_MDValueLE,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueGT", (PyCFunction) xmipp_MDValueGT,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueGE", (PyCFunction) xmipp_MDValueGE,
          METH_VARARGS, "Construct a relational query" },
        { "MDValueRange", (PyCFunction) xmipp_MDValueRange,
          METH_VARARGS, "Construct a range query" },
        { "addLabelAlias", (PyCFunction) xmipp_addLabelAlias,
          METH_VARARGS, "Add a label alias dinamically in run time. Use for reading non xmipp star files" },
        { "getNewAlias", (PyCFunction) xmipp_getNewAlias,
          METH_VARARGS, "Add a label dinamically in run time. Use for reading non xmipp star files" },
        { "createEmptyFile", (PyCFunction) xmipp_createEmptyFile,
          METH_VARARGS, "create empty stack (speed up things)" },
        { "getImageSize", (PyCFunction) xmipp_getImageSize,
          METH_VARARGS, "Get image dimensions" },
        { "MetaDataInfo", (PyCFunction) xmipp_MetaDataInfo, METH_VARARGS,
          "Get image dimensions of first metadata entry and the number of entries" },
        { "existsBlockInMetaDataFile", (PyCFunction) xmipp_existsBlockInMetaDataFile, METH_VARARGS,
          "Does block exists in file" },
        { "ImgCompare", (PyCFunction) xmipp_ImgCompare,  METH_VARARGS,
          "return true if both files are identical" },
        { "checkImageFileSize", (PyCFunction) xmipp_CheckImageFileSize,  METH_VARARGS,
          "return true if the file has at least as many bytes as needed to read the image" },
        { "checkImageCorners", (PyCFunction) xmipp_CheckImageCorners,  METH_VARARGS,
          "return false if the image has repeated pixels at some corner" },
        { "compareTwoFiles", (PyCFunction) xmipp_compareTwoFiles, METH_VARARGS,
          "return true if both files are identical" },
        { "bsoftRemoveLoopBlock", (PyCFunction) xmipp_bsoftRemoveLoopBlock, METH_VARARGS,
          "convert bsoft star files to xmipp star files" },
        { "bsoftRestoreLoopBlock", (PyCFunction) xmipp_bsoftRestoreLoopBlock, METH_VARARGS,
          "convert xmipp star files to bsoft star files" },
        { "compareTwoImageTolerance", (PyCFunction) xmipp_compareTwoImageTolerance, METH_VARARGS,
          "return true if both images are very similar" },
        { "readMetaDataWithTwoPossibleImages", (PyCFunction) xmipp_readMetaDataWithTwoPossibleImages, METH_VARARGS,
          "Read a 1 or two column list of micrographs" },
        { "substituteOriginalImages", (PyCFunction) xmipp_substituteOriginalImages, METH_VARARGS,
          "Substitute the original images into a given column of a metadata" },
        { "compareTwoMetadataFiles", (PyCFunction) xmipp_compareTwoMetadataFiles, METH_VARARGS,
          "Compare two metadata files" },
        { "dumpToFile", (PyCFunction) xmipp_dumpToFile, METH_VARARGS,
          "dump metadata to sqlite database" },
        { "Euler_angles2matrix", (PyCFunction) xmipp_Euler_angles2matrix, METH_VARARGS,
          "convert euler angles to transformation matrix" },
        { "Euler_matrix2angles", (PyCFunction) xmipp_Euler_matrix2angles, METH_VARARGS,
          "convert transformation matrix to euler angles" },
        { "Euler_direction", (PyCFunction) xmipp_Euler_direction, METH_VARARGS,
          "converts euler angles to direction" },
		{ "alignWithZ", (PyCFunction) xmipp_alignWithZ, METH_VARARGS,
		  "align vector with Z" },
        { "activateMathExtensions", (PyCFunction) xmipp_activateMathExtensions,
          METH_VARARGS, "activate math function in metadatas" },
        { "activateRegExtensions", (PyCFunction) xmipp_activateRegExtensions,
          METH_VARARGS, "activate regular expressions in metadatas" },
        { "fastEstimateEnhancedPSD", (PyCFunction) xmipp_fastEstimateEnhancedPSD, METH_VARARGS,
          "Utility function to calculate PSD preview" },
        { "bandPassFilter", (PyCFunction) xmipp_bandPassFilter, METH_VARARGS,
          "Utility function to apply bandpass filter" },
        { "gaussianFilter", (PyCFunction) xmipp_gaussianFilter, METH_VARARGS,
          "Utility function to apply gaussian filter in Fourier space" },
        { "realGaussianFilter", (PyCFunction) xmipp_realGaussianFilter, METH_VARARGS,
          "Utility function to apply gaussian filter in Real space" },
        { "badPixelFilter", (PyCFunction) xmipp_badPixelFilter, METH_VARARGS,
          "Bad pixel filter" },
        { "errorBetween2CTFs", (PyCFunction) xmipp_errorBetween2CTFs,
          METH_VARARGS, "difference between two metadatas" },
        { "errorMaxFreqCTFs", (PyCFunction) xmipp_errorMaxFreqCTFs,
          METH_VARARGS, "resolution at which CTFs phase differs more than 90 degrees" },
        { "errorMaxFreqCTFs2D", (PyCFunction) xmipp_errorMaxFreqCTFs2D,
          METH_VARARGS, "resolution at which CTFs phase differs more than 90 degrees, 2D case" },
		{ "convertPSD", (PyCFunction) Image_convertPSD, METH_VARARGS,
		  "Convert to PSD: center FFT and use logarithm" },
		{ "image_align", (PyCFunction) Image_align, METH_VARARGS,
		  "I2aligned=image_align(I1,I2), align I2 to resemble I1." },
		{ "applyCTF", (PyCFunction) Image_applyCTF, METH_VARARGS,
		  "Apply CTF to this image. Ts is the sampling rate of the image." },
		{ "getPSF", (PyCFunction) xmipp_getPSF, METH_VARARGS,
		  "Get PSF from a CTF. default Ts=0.5A" },
		{ "projectVolumeDouble", (PyCFunction) Image_projectVolumeDouble, METH_VARARGS,
		  "project a volume using Euler angles" },
        { nullptr } /* Sentinel */
    };//xmipp_methods

#define INIT_TYPE(type) if (PyType_Ready(&type##Type) < 0) return module; Py_INCREF(&type##Type);\
    PyModule_AddObject(module, #type, (PyObject *) &type##Type);


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "xmippLib",           /* m_name */
        "xmippLib objects",   /* m_doc */
        -1,                   /* m_size */
        xmipp_methods         /* m_methods */
};

PyMODINIT_FUNC
PyInit_xmippLib(void) {
    //Initialize module variable

    PyObject *module = PyModule_Create(&moduledef);

    //Check types and add to module
    INIT_TYPE(FileName);
    INIT_TYPE(Image);
    INIT_TYPE(MDQuery);
    INIT_TYPE(MetaData);
    INIT_TYPE(Program);
    INIT_TYPE(SymList);
    INIT_TYPE(FourierProjector);


    //Add PyXmippError
    char message[32]="xmipp.XmippError";
    PyXmippError = PyErr_NewException(message, nullptr, nullptr);
    Py_INCREF(PyXmippError);
    PyModule_AddObject(module, "XmippError", PyXmippError);

    //Add MDLabel constants
    PyObject * dict = PyModule_GetDict(module);
    addLabels(dict);

    return module;
}
