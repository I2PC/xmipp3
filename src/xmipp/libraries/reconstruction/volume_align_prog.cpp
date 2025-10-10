/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "interface/frm.h"  // must be included first as it defines _POSIX_C_SOURCE
#include <fstream>
#include "core/xmipp_image.h"
#include "data/filters.h"
#include "core/geometry.h"
#include "data/mask.h"
#include "core/xmipp_program.h"
#include "core/transformations.h"

// Alignment parameters needed by fitness ----------------------------------
class AlignParams
{
public:
#define COVARIANCE       1
#define LEAST_SQUARES    2

    int alignment_method;

    Image<double> V1;
    Image<double> V2;
    Image<double> Vaux;
    const MultidimArray<int> *mask_ptr;

    bool wrap;
};

// Global parameters needed by fitness ------------------------------------
static AlignParams params; // create one version for each cpp file

// Apply transformation ---------------------------------------------------
void applyTransformation(const MultidimArray<double> &V2,
                         MultidimArray<double> &Vaux,
                         double *p,
						 bool wrap)
{
    Matrix1D<double> r(3);
    Matrix2D<double> A, Aaux;

    double flip      = p[0];
    double greyScale = p[1];
    double greyShift = p[2];
    double rot       = p[3];
    double tilt      = p[4];
    double psi       = p[5];
    double scale     = p[6];
    ZZ(r)            = p[7];
    YY(r)            = p[8];
    XX(r)            = p[9];

    Euler_angles2matrix(rot, tilt, psi, A, true);
    translation3DMatrix(r,Aaux);
    A = A * Aaux;
    scale3DMatrix(vectorR3(scale, scale, scale),Aaux);
    A = A * Aaux;

    for (int i = 0; i < 4; ++i)
    {
        MAT_ELEM(A,2,i) *= flip;
    }

    applyGeometry(xmipp_transformation::LINEAR, Vaux, V2, A, xmipp_transformation::IS_NOT_INV, wrap);
    if (greyScale!=1 || greyShift!=0)
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vaux)
        DIRECT_MULTIDIM_ELEM(Vaux,n)=DIRECT_MULTIDIM_ELEM(Vaux,n)*greyScale+greyShift;
}


// Fitness between two volumes --------------------------------------------
double fitness(double *p)
{
    applyTransformation(params.V2(),params.Vaux(),p, params.wrap);

    // Correlate
    double fit=0.;
    switch (params.alignment_method)
    {
    case COVARIANCE:
                    fit = -correlationIndex(params.V1(), params.Vaux(), params.mask_ptr);
        break;
    case LEAST_SQUARES:
                    fit = rms(params.V1(), params.Vaux(), params.mask_ptr);
        break;
    }
    return fit;
}


double wrapperFitness(double *p, void *params)
{
    const double flip = *static_cast<double*>(params);
    std::array<double,10> pp;
    pp[0] = flip;
    std::copy_n(p+1, 9, &pp[1]);
    return fitness(pp.data());
}


class ProgAlignVolumes : public XmippProgram
{
public:
    Mask mask;

    FileName fn1, fn2;
    double   rot0, rotF, tilt0, tiltF, psi0, psiF;
    double   step_rot, step_tilt, step_psi;
    double   scale0, scaleF, step_scale;
    double   z0, zF, y0, yF, x0, xF, step_z, step_y, step_x;
    double   grey_scale0, grey_scaleF, step_grey;
    double   grey_shift0, grey_shiftF, step_grey_shift;
    bool     consider_mirror;
    int      tell;
    bool     apply;
    FileName fnOut, fnGeo, fnGray, fnStore;
    bool     mask_enabled;
    bool     usePowell, onlyShift, useFRM, copyGeo, copyGray, store, wrap;
    double   maxFreq;
    int      maxShift;
    bool     dontScale;
    int starting_tilt, ending_tilt; //compensating for a single tilt wedge mask (tomography data)
public:

    void defineParams()
    {

        addUsageLine("Align two volumes varying orientation, position and scale");
        addParamsLine("   --i1 <volume1>        : the first volume to align");
        addParamsLine("   --i2 <volume2>        : the second one");
        addParamsLine("  [--rot   <rot0=0>  <rotF=0>  <step_rot=1>]  : in degrees");
        addParamsLine("  [--tilt  <tilt0=0> <tiltF=0> <step_tilt=1>] : in degrees");
        addParamsLine("  [--psi   <psi0=0>  <psiF=0>  <step_psi=1>]  : in degrees");
        addParamsLine("  [--scale <sc0=1>   <scF=1>   <step_sc=1>]   : size scale margin");
        addParamsLine("  [--grey_scale <sc0=1> <scF=1> <step_sc=1>]  : grey scale margin");
        addParamsLine("    requires --least_squares;");
        addParamsLine("  [--grey_shift <sh0=0> <shF=0> <step_sh=1>]  : grey shift margin");
        addParamsLine("    requires --least_squares;");
        addParamsLine("  [-z <z0=0> <zF=0> <step_z=1>] : Z position in pixels");
        addParamsLine("  [-y <y0=0> <yF=0> <step_y=1>] : Y position in pixels");
        addParamsLine("  [-x <x0=0> <xF=0> <step_x=1>] : X position in pixels");
        addParamsLine("  [--consider_mirror] : Consider the mirror volume");
        addParamsLine("  [--show_fit]      : Show fitness values");
        addParamsLine("  [--apply <file=\"\">] : Apply best movement to --i2 and store results in this file");
        addParamsLine("  [--covariance]    : Covariance fitness criterion");
        addParamsLine("  [--least_squares] : LS fitness criterion");
        addParamsLine("  [--local]         : Use local optimizer instead of exhaustive search");
        addParamsLine("  [--frm <maxFreq=0.25> <maxShift=10> <tilt0=-90> <tiltF=90>] : Use Fast Rotational Matching, if tilt0 and tiltF are left as -90 and 90 they do not effect the results of the alignment");
        addParamsLine("                    : Maximum frequency is in digital frequencies (<0.5)");
        addParamsLine("                    : Maximum shift is in pixels");
        addParamsLine("                    :+ See Y. Chen, et al. Fast and accurate reference-free alignment of subtomograms. JSB, 182: 235-245 (2013)");
        addParamsLine("  [--onlyShift]     : Only shift");
        addParamsLine("  [--dontScale]     : Do not look for scale changes");
        addParamsLine("  [--copyGeo <file=\"\">] : copy transformation matrix in a txt file. ('A' matrix elements)");
        addParamsLine("  [--copyGray <file=\"\">] : copy gray scale and shift txt file.)");
        addParamsLine("  [--store <file=\"\">] : copy angles and shifts to a txt file.");
        addParamsLine("  [--dontWrap] : Do not wrap input2 when aligning to input1");
        addParamsLine(" == Mask Options == ");
        mask.defineParams(this,INT_MASK,nullptr,nullptr,true);
        addExampleLine("Typically you first look for a rough approximation of the alignment using exhaustive search. For instance, for a global rotational alignment use",false);
        addExampleLine("xmipp_volume_align --i1 volume1.vol --i2 volume2.vol --rot 0 360 15 --tilt 0 180 15 --psi 0 360 15");
        addExampleLine("Then, assume the best alignment is obtained for rot=45, tilt=60, psi=90",false);
        addExampleLine("Now you perform a local search to refine the estimation and apply",false);
        addExampleLine("xmipp_volume_align --i1 volume1.vol --i2 volume2.vol --rot 45 --tilt 60 --psi 90 --local --apply volume2aligned.vol");
        addSeeAlsoLine("xmipp_volumeset_align");
    }

    void readParams()
    {
        // Get parameters =======================================================
        fn1 = getParam("--i1");
        fn2 = getParam("--i2");

        rot0 = getDoubleParam("--rot",0);
        rotF = getDoubleParam("--rot",1);
        step_rot = getDoubleParam("--rot",2);

        tilt0 = getDoubleParam("--tilt",0);
        tiltF = getDoubleParam("--tilt",1);
        step_tilt = getDoubleParam("--tilt",2);

        psi0 = getDoubleParam("--psi",0);
        psiF = getDoubleParam("--psi",1);
        step_psi = getDoubleParam("--psi",2);

        scale0 = getDoubleParam("--scale",0);
        scaleF = getDoubleParam("--scale",1);
        step_scale = getDoubleParam("--scale",2);

        grey_scale0 = getDoubleParam("--grey_scale",0);
        grey_scaleF = getDoubleParam("--grey_scale",1);
        step_grey = getDoubleParam("--grey_scale",2);

        grey_shift0 = getDoubleParam("--grey_shift",0);
        grey_shiftF = getDoubleParam("--grey_shift",1);
        step_grey_shift = getDoubleParam("--grey_shift",2);

        z0 = getDoubleParam("-z",0);
        zF = getDoubleParam("-z",1);
        step_z = getDoubleParam("-z",2);

        y0 = getDoubleParam("-y",0);
        yF = getDoubleParam("-y",1);
        step_y = getDoubleParam("-y",2);

        x0 = getDoubleParam("-x",0);
        xF = getDoubleParam("-x",1);
        step_x = getDoubleParam("-x",2);

        consider_mirror = checkParam("--consider_mirror");

        mask_enabled = checkParam("--mask");
        if (mask_enabled)
            mask.read(argc, argv);

        usePowell = checkParam("--local");
        useFRM = checkParam("--frm");
        if (useFRM)
        {
        	maxFreq=getDoubleParam("--frm",0);
        	maxShift=getIntParam("--frm",1);
        	starting_tilt=getIntParam("--frm",2);
        	ending_tilt=getIntParam("--frm",3);
        }
        onlyShift = checkParam("--onlyShift");
        wrap = !checkParam("--dontWrap");

        if (step_rot   == 0)
            step_rot = 1;
        if (step_tilt  == 0)
            step_tilt  = 1;
        if (step_psi   == 0)
            step_psi = 1;
        if (step_scale == 0)
            step_scale = 1;
        if (step_grey  == 0)
            step_grey  = 1;
        if (step_grey_shift  == 0)
            step_grey_shift  = 1;
        if (step_z     == 0)
            step_z = 1;
        if (step_y     == 0)
            step_y = 1;
        if (step_x     == 0)
            step_x = 1;

        tell = checkParam("--show_fit");
        apply = checkParam("--apply");
        fnOut = getParam("--apply");
        copyGeo = checkParam("--copyGeo");
        copyGray = checkParam("--copyGray");
        fnGeo = getParam("--copyGeo");
        fnGray = getParam("--copyGray");
        store = checkParam("--store");
        fnStore = getParam("--store");
        dontScale = checkParam("--dontScale");

        if (checkParam("--covariance"))
        {
            params.alignment_method = COVARIANCE;
        }
        else if (checkParam("--least_squares"))
        {
            params.alignment_method = LEAST_SQUARES;
        }
        else
        {
            params.alignment_method = COVARIANCE;
        }
    }

    void run ()
    {
        mask.allowed_data_types = INT_MASK;

        // Main program =========================================================
        params.V1.read(fn1);
        params.V1().setXmippOrigin();
        params.V2.read(fn2);
        params.V2().setXmippOrigin();
        params.wrap = wrap;

        // Initialize best_fit
        double best_fit = 1e38;
        Matrix1D<double> best_align(10);
        bool first = true;

        // Generate mask
        if (mask_enabled)
        {
            mask.generate_mask(params.V1());
            params.mask_ptr = &(mask.get_binary_mask());
        }
        else
            params.mask_ptr = nullptr;

        // Exhaustive search
        if (!usePowell && !useFRM)
        {
            // Count number of iterations
            int times = 1;
            if (!tell)
            {
                if (consider_mirror)
                    times *= 2;
                if (grey_scale0 != grey_scaleF)
                    times *= FLOOR(1 + (grey_scaleF - grey_scale0) / step_grey);
                if (grey_shift0 != grey_shiftF)
                    times *= FLOOR(1 + (grey_shiftF - grey_shift0) / step_grey_shift);
                if (rot0 != rotF)
                    times *= FLOOR(1 + (rotF - rot0) / step_rot);
                if (tilt0 != tiltF)
                    times *= FLOOR(1 + (tiltF - tilt0) / step_tilt);
                if (psi0 != psiF)
                    times *= FLOOR(1 + (psiF - psi0) / step_psi);
                if (scale0 != scaleF)
                    times *= FLOOR(1 + (scaleF - scale0) / step_scale);
                if (z0 != zF)
                    times *= FLOOR(1 + (zF - z0) / step_z);
                if (y0 != yF)
                    times *= FLOOR(1 + (yF - y0) / step_y);
                if (x0 != xF)
                    times *= FLOOR(1 + (xF - x0) / step_x);
                init_progress_bar(times);
            }
            else
                std::cout << "#grey_factor rot tilt psi scale z y x fitness\n";

            // Iterate
            int itime = 0;
            int step_time = CEIL((double)times / 60.0);
            Matrix1D<double> r(3);
            Matrix1D<double> trial(10);
            for (int mirror = 0; mirror <= static_cast<int>(consider_mirror); mirror++)
                for (double grey_scale = grey_scale0; grey_scale <= grey_scaleF ; grey_scale += step_grey)
                    for (double grey_shift = grey_shift0; grey_shift <= grey_shiftF ; grey_shift += step_grey_shift)
                        for (double rot = rot0; rot <= rotF ; rot += step_rot)
                            for (double tilt = tilt0; tilt <= tiltF ; tilt += step_tilt)
                                for (double psi = psi0; psi <= psiF ; psi += step_psi)
                                    for (double scale = scale0; scale <= scaleF ; scale += step_scale)
                                        for (ZZ(r) = z0; ZZ(r) <= zF ; ZZ(r) += step_z)
                                            for (YY(r) = y0; YY(r) <= yF ; YY(r) += step_y)
                                                for (XX(r) = x0; XX(r) <= xF ; XX(r) += step_x)
                                                {
                                                    // Form trial vector
                                                    trial(0) = mirror ? -1.0 : 1.0;
                                                    trial(1) = grey_scale;
                                                    trial(2) = grey_shift;
                                                    trial(3) = rot;
                                                    trial(4) = tilt;
                                                    trial(5) = psi;
                                                    trial(6) = scale;
                                                    trial(7) = ZZ(r);
                                                    trial(8) = YY(r);
                                                    trial(9) = XX(r);
                                                    // Evaluate
                                                    double fit = fitness(MATRIX1D_ARRAY(trial));
                                                    // The best?
                                                    if (fit < best_fit || first)
                                                    {
                                                        best_fit = fit;
                                                        best_align = trial;
                                                        first = false;
                                                        if (tell)
                                                            std::cout << "Best so far\n";
                                                    }
                                                    // Show fit
                                                    if (tell)
                                                        std::cout << trial << " " << fit << std::endl;
                                                    else
                                                        if (++itime % step_time == 0)
                                                            progress_bar(itime);
                                                }
                if (!tell)
                    progress_bar(times);
            }
        else if (usePowell)
        {
            // Use Powell optimization
            Matrix1D<double> x(9), steps(9);
            double fitness;
            int iter;
            steps.initConstant(1);
            if (onlyShift)
                steps(0)=steps(1)=steps(2)=steps(3)=steps(4)=steps(5)=0;
            if (params.alignment_method == COVARIANCE)
                steps(0)=steps(1)=0;
            if (dontScale)
            	steps(5)=0;

            for (int mirror = 0; mirror <= static_cast<int>(consider_mirror); mirror++)
            {
                x(0)=grey_scale0;
                x(1)=grey_shift0;
                x(2)=rot0;
                x(3)=tilt0;
                x(4)=psi0;
                x(5)=scale0;
                x(6)=z0;
                x(7)=y0;
                x(8)=x0;
                
                double flip = mirror ? -1.0 : 1.0;
                powellOptimizer(x,1,9,&wrapperFitness,&flip,0.01,fitness,iter,steps,true);

                if (fitness < best_fit || first)
                {
                    best_align(0)=flip;
                    for (int i = 0; i < 9; i++)
                        best_align(i+1)=x(i);

                    best_fit=fitness;
                }
                
                first=false;
            }
        }
        else if (useFRM)
        {
    		Python::initPythonAndNumpy();
    		PyObject * pFunc = Python::getFunctionRef("sh_alignment.frm", "frm_align");
    		double rot,tilt,psi,x,y,z,score;
    		Matrix2D<double> A;
            MultidimArray<double> Vaux;
            
            for (int mirror = 0; mirror <= static_cast<int>(consider_mirror); mirror++)
            {
                Vaux = params.V2();
                if (mirror)
                {
                    Vaux.selfReverseZ();
                }

                if(starting_tilt!=-90 || ending_tilt!=90){
                    std::cout<<"you are compensating for the missing wedge, the first volume should be rotated with 90 degrees about the y-axis"<<std::endl;
                    PyObject * pSTMMclass = Python::getClassRef("sh_alignment.tompy.filter", "SingleTiltWedge");
                    PyObject * arglist = Py_BuildValue("(ii)", starting_tilt , ending_tilt);
                    PyObject * SingleTiltWedgeMask = PyObject_CallObject(pSTMMclass, arglist);
                    // The order of volumes has to be flipped in order to compensate for a single tilt missing wedge. For those who are not using this mask, no changes in results will happen.
                    alignVolumesFRM(pFunc, Vaux, params.V1(), SingleTiltWedgeMask, rot,tilt,psi,x,y,z,score,A,maxShift,maxFreq,params.mask_ptr);
                    std::cout<<"If you intend to apply transform using xmipp_transform_geometry, use --inverse flag (if it was not present before), or remove it (if it was present before)"<<std::endl;
                    Py_DECREF(SingleTiltWedgeMask);
                    Py_DECREF(arglist);
                    Py_DECREF(pSTMMclass);
                }
                else{
                    alignVolumesFRM(pFunc, params.V1(), Vaux, Py_None, rot,tilt,psi,x,y,z,score,A,maxShift,maxFreq,params.mask_ptr);
                }

                const auto fit = -score;
                if (fit < best_fit || first)
                {
                    best_align(0)=mirror ? -1.0 : 1.0;
                    best_align(1)=1; // Gray scale
                    best_align(2)=0; // Gray shift
                    best_align(3)=rot;
                    best_align(4)=tilt;
                    best_align(5)=psi;
                    best_align(6)=1; // Scale
                    best_align(7)=z;
                    best_align(8)=y;
                    best_align(9)=x;
                    best_fit=fit;
                }
                
                first=false;
            }
        }

        if (!first)
            std::cout << "The best correlation is for\n"
            << "Mirroring the in X axis: " << (best_align(0) < 0) << std::endl
            << "Scale                  : " << best_align(5) << std::endl
            << "Translation (X,Y,Z)    : " << best_align(8)
            << " " << best_align(7) << " " << best_align(6)
            << std::endl
            << "Rotation (rot,tilt,psi): "
            << best_align(2) << " " << best_align(3) << " "
            << best_align(4) << std::endl
            << "Best grey scale       : " << best_align(0) << std::endl
            << "Best grey shift       : " << best_align(1) << std::endl
            << "Fitness value         : " << best_fit << std::endl;
        Matrix1D<double> r(3);
        XX(r)            = best_align(8);
        YY(r)            = best_align(7);
        ZZ(r)            = best_align(6);
        Matrix2D<double> A,Aaux;

        Euler_angles2matrix(best_align(2), best_align(3), best_align(4),
                            A, true);
        translation3DMatrix(r,Aaux);

        //A = A + Aaux;
        //A(0,0) -= 1;
        //A(1,1) -= 1;
        //A(2,2) -= 1;
        //A(3,3) -= 1;
        A = A * Aaux;

        scale3DMatrix(vectorR3(best_align(5), best_align(5), best_align(5)),Aaux);
        A = A * Aaux;

        for (int i = 0; i < 4; ++i)
        {
            MAT_ELEM(A,2,i) *= best_align(0);
        }

        if (verbose!=0)
			std::cout << "xmipp_transform_geometry will require the following values"
					  << "\n   Angles: " << best_align(2) << " "
					  << best_align(3) << " " << best_align(4)
					  << "\n   Shifts: " << A(0,3) << " " << A(1,3) << " " << A(2,3)
					  << std::endl;
        if (copyGeo)
        {
        	std::ofstream outputGeo (fnGeo.c_str());
        	outputGeo << A(0,0) << "\n" << A(0,1) << "\n" << A(0,2) << "\n" << A(0,3) << "\n"
        			  << A(1,0) << "\n" << A(1,1) << "\n" << A(1,2) << "\n" << A(1,3) << "\n"
					  << A(2,0) << "\n" << A(2,1) << "\n" << A(2,2) << "\n" << A(2,3) << "\n"
					  << A(3,0) << "\n" << A(3,1) << "\n" << A(3,2) << "\n" << A(3,3) << "\n"
					  << std::endl;
        	outputGeo.close();
        }
        if (copyGray)
        {
        	std::ofstream outputGray (fnGray.c_str());
        	outputGray << best_align(0) << "\n"
        			   << best_align(1) << "\n"
					   << std::endl;
        	outputGray.close();
        }
        if (store)
        {
        	std::ofstream outputStore (fnStore.c_str());
        	outputStore << best_align(2)  << ", " << best_align(3) << ", " << best_align(4) << ", " << A(0,3) << ", "
					  << A(1,3) << ", " << A(2,3) << ", " << best_fit << std::endl;
        	outputStore.close();
        }
        if (apply)
        {
            applyTransformation(params.V2(),params.Vaux(),MATRIX1D_ARRAY(best_align), wrap);
            params.V2()=params.Vaux();
            params.V2.write(fnOut);
        }
    }
};
