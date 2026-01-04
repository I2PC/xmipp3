/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             James Krieger                jamesmkrieger@gmail.com
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

#include "continuous_create_residuals.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/mask.h"

// Empty constructor =======================================================
ProgContinuousCreateResiduals::ProgContinuousCreateResiduals()
{
    produces_a_metadata = true;
    each_image_produces_an_output = true;
    projector = nullptr;
    ctfImage = nullptr;
    rank = 0;
}

// Empty destructor =======================================================
ProgContinuousCreateResiduals::~ProgContinuousCreateResiduals()
{
	delete projector;
	delete ctfImage;
}

// Read arguments ==========================================================
void ProgContinuousCreateResiduals::readParams()
{
	XmippMetadataProgram::readParams();
    fnVol = getParam("--ref");
    maxShift = getDoubleParam("--max_shift");
    maxScale = getDoubleParam("--max_scale");
    maxDefocusChange = getDoubleParam("--max_defocus_change");
    maxAngularChange = getDoubleParam("--max_angular_change");
    maxA = getDoubleParam("--max_gray_scale");
    maxB = getDoubleParam("--max_gray_shift");
    Ts = getDoubleParam("--sampling");
    Rmax = getIntParam("--Rmax");
    pad = getIntParam("--padding");
    optimizeGrayValues = checkParam("--optimizeGray");
    optimizeShift = checkParam("--optimizeShift");
    optimizeScale = checkParam("--optimizeScale");
    optimizeAngles = checkParam("--optimizeAngles");
    optimizeDefocus = checkParam("--optimizeDefocus");
    ignoreCTF = checkParam("--ignoreCTF");
    phaseFlipped = checkParam("--phaseFlipped");
    // penalization = getDoubleParam("--penalization");
    fnResiduals = getParam("--oresiduals");
    fnProjections = getParam("--oprojections");
	sameDefocus = checkParam("--sameDefocus");
	keep_input_columns = true; // each output metadata row is a deep copy of the input metadata row
}

// Show ====================================================================
void ProgContinuousCreateResiduals::show()
{
    if (!verbose)
        return;
	XmippMetadataProgram::show();
    std::cout
    << "Reference volume:    " << fnVol              << std::endl
    << "Max. Shift:          " << maxShift           << std::endl
    << "Max. Scale:          " << maxScale           << std::endl
    << "Max. Angular Change: " << maxAngularChange   << std::endl
    << "Max. Defocus Change: " << maxDefocusChange   << std::endl
	<< "Max. Gray Scale:     " << maxA               << std::endl
	<< "Max. Gray Shift:     " << maxB               << std::endl
    << "Sampling:            " << Ts                 << std::endl
    << "Max. Radius:         " << Rmax               << std::endl
    << "Padding factor:      " << pad                << std::endl
    << "Optimize gray:       " << optimizeGrayValues << std::endl
    << "Optimize shifts:     " << optimizeShift      << std::endl
    << "Optimize scale:      " << optimizeScale      << std::endl
    << "Optimize angles:     " << optimizeAngles     << std::endl
    << "Optimize defocus:    " << optimizeDefocus    << std::endl
    << "Ignore CTF:          " << ignoreCTF          << std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    // << "Penalization:        " << penalization       << std::endl
	<< "Force same defocus:  " << sameDefocus        << std::endl
    << "Output residuals:    " << fnResiduals        << std::endl
    << "Output projections:  " << fnProjections      << std::endl
    ;
}

// usage ===================================================================
void ProgContinuousCreateResiduals::defineParams()
{
    addUsageLine("Make an image residual");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Stack of images prepared for 3D reconstruction");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
    addParamsLine("  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine("  [--max_scale <s=0.02>]       : Maximum scale change");
    addParamsLine("  [--max_angular_change <a=5>] : Maximum angular change allowed (in degrees)");
    addParamsLine("  [--max_defocus_change <d=500>] : Maximum defocus change allowed (in Angstroms)");
    addParamsLine("  [--max_gray_scale <a=0.05>]  : Maximum gray scale change");
    addParamsLine("  [--max_gray_shift <b=0.05>]  : Maximum gray shift change as a factor of the image standard deviation");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--Rmax <R=-1>]              : Maximum radius (px). -1=Half of volume size");
    addParamsLine("  [--padding <p=2>]            : Padding factor");
    addParamsLine("  [--optimizeGray]             : Optimize gray values");
    addParamsLine("  [--optimizeShift]            : Optimize shift");
    addParamsLine("  [--optimizeScale]            : Optimize scale");
    addParamsLine("  [--optimizeAngles]           : Optimize angles");
    addParamsLine("  [--optimizeDefocus]          : Optimize defocus");
    addParamsLine("  [--ignoreCTF]                : Ignore CTF");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    // addParamsLine("  [--penalization <l=100>]     : Penalization for the average term");
	addParamsLine("  [--sameDefocus]              : Force defocusU = defocusV");
    addParamsLine("  [--oresiduals <stack=\"\">]  : Output stack for the residuals");
    addParamsLine("  [--oprojections <stack=\"\">] : Output stack for the projections");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_continuous_create_residuals -i anglesFromDiscreteAssignment.xmd --ref reference.vol -o assigned_angles.stk");
}

void ProgContinuousCreateResiduals::startProcessing()
{
	XmippMetadataProgram::startProcessing();
	if (fnResiduals!="")
		createEmptyFile(fnResiduals, xdimOut, ydimOut, zdimOut, mdInSize, true, WRITE_OVERWRITE);
	if (fnProjections!="")
		createEmptyFile(fnProjections, xdimOut, ydimOut, zdimOut, mdInSize, true, WRITE_OVERWRITE);
}

// Produce side information ================================================
void ProgContinuousCreateResiduals::preProcess()
{
    // Read the reference volume
	Image<double> V;
	if (rank==0)
	{
		V.read(fnVol);
		V().setXmippOrigin();
	    Xdim=XSIZE(V());
	}
	else
	{
		size_t ydim, zdim, ndim;
		getImageSize(fnVol, Xdim, ydim, zdim, ndim);
	}

    E().initZeros(Xdim,Xdim);
    Pp().initZeros(Xdim,Xdim);
    Pp().setXmippOrigin();

    // Construct mask
    if (Rmax<0)
    	Rmax=Xdim/2;
    Mask mask;
    mask.type = BINARY_CIRCULAR_MASK;
    mask.mode = INNER_MASK;
    mask.R1 = Rmax;
    mask.generate_mask(Xdim,Xdim);
    mask2D=mask.get_binary_mask();
    iMask2Dsum=1.0/mask2D.sum();

    // Construct reference covariance
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mask2D)
    if (DIRECT_MULTIDIM_ELEM(mask2D,n))
    	DIRECT_MULTIDIM_ELEM(E(),n)=rnd_gaus(0,1);
    covarianceMatrix(E(),C0);
    FOR_ALL_ELEMENTS_IN_MATRIX2D(C0)
    {
    	double val=MAT_ELEM(C0,i,j);
    	if (val<0.5)
    		MAT_ELEM(C0,i,j)=0;
    	else
    		MAT_ELEM(C0,i,j)=1;
    }

    // Construct projector
    if (rank==0)
    	projector = new FourierProjector(V(),pad,Ts,xmipp_transformation::BSPLINE3);
    else
    	projector = new FourierProjector(pad,Ts,xmipp_transformation::BSPLINE3);

    // Transformation matrix
    A.initIdentity(3);

    // Continuous cost
    if (optimizeGrayValues)
    	contCost = CONTCOST_L1;
    else
    	contCost = CONTCOST_CORR;
}

//#define DEBUG
void ProgContinuousCreateResiduals::updateCTFImage(double defocusU, double defocusV, double angle)
{
	ctf.K=1; // get pure CTF with no envelope
	currentDefocusU=ctf.DeltafU=defocusU;
	currentDefocusV=ctf.DeltafV=defocusV;
	currentAngle=ctf.azimuthal_angle=angle;
	ctf.produceSideInfo();
	if (ctfImage==nullptr)
	{
		ctfImage = new MultidimArray<double>();
		ctfImage->resizeNoCopy(I());
		STARTINGY(*ctfImage)=STARTINGX(*ctfImage)=0;

		ctfEnvelope = std::make_unique<MultidimArray<double>>();
		ctfEnvelope->resizeNoCopy(I());
		STARTINGY(*ctfEnvelope)=STARTINGX(*ctfEnvelope)=0;
	}
	ctf.generateCTF((int)YSIZE(I()),(int)XSIZE(I()),*ctfImage,Ts);
	ctf.generateEnvelope((int)YSIZE(I()),(int)XSIZE(I()),*ctfEnvelope,Ts);
	if (phaseFlipped)
		FOR_ALL_ELEMENTS_IN_ARRAY2D(*ctfImage)
			A2D_ELEM(*ctfImage,i,j)=fabs(A2D_ELEM(*ctfImage,i,j));
}

//#define DEBUG
//#define DEBUG2
double transformImage(ProgContinuousCreateResiduals *prm, double rot, double tilt, double psi,
		double a, double b, const Matrix2D<double> &A, 
		double deltaDefocusU, double deltaDefocusV, double deltaDefocusAngle, int degree)
{
    if (prm->hasCTF)
    {
    	double defocusU=prm->old_defocusU+deltaDefocusU;
		double defocusV;
		if (prm->sameDefocus){
    		defocusV=defocusU;
		}
		else
			defocusV=prm->old_defocusV+deltaDefocusV;
    	double angle=prm->old_defocusAngle+deltaDefocusAngle;
    	if (defocusU!=prm->currentDefocusU || defocusV!=prm->currentDefocusV || angle!=prm->currentAngle)
    		prm->updateCTFImage(defocusU,defocusV,angle);
    }
	projectVolume(*(prm->projector), prm->P, (int)XSIZE(prm->I()), (int)XSIZE(prm->I()),
	              rot, tilt, psi, (const MultidimArray<double> *)prm->ctfImage);
	if (prm->old_flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}

	applyGeometry(degree,prm->Pp(),prm->P(),A,xmipp_transformation::IS_INV,xmipp_transformation::DONT_WRAP,0.);
	const MultidimArray<double> &mP=prm->P();
	const MultidimArray<double> &mI=prm->I();
	const MultidimArray<int> &mMask2D=prm->mask2D;
	MultidimArray<double> &mPp=prm->Pp();
	MultidimArray<double> &mE=prm->E();
	mE.initZeros();
	if (prm->contCost==CONTCOST_L1)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mMask2D)
		{
			if (DIRECT_MULTIDIM_ELEM(mMask2D,n))
			{
				DIRECT_MULTIDIM_ELEM(mPp,n)=DIRECT_MULTIDIM_ELEM(mPp,n);
				double val=(a*DIRECT_MULTIDIM_ELEM(mPp,n)+b)-DIRECT_MULTIDIM_ELEM(mI,n);
				DIRECT_MULTIDIM_ELEM(mE,n)=val;
			}
			else
				DIRECT_MULTIDIM_ELEM(mPp,n)=0;
		}
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mMask2D)
		{
			if (DIRECT_MULTIDIM_ELEM(mMask2D,n))
			{
				double val=DIRECT_MULTIDIM_ELEM(mPp,n)-DIRECT_MULTIDIM_ELEM(mI,n);
				DIRECT_MULTIDIM_ELEM(mE,n)=val;
			}
			else
				DIRECT_MULTIDIM_ELEM(mI,n)=0;
		}
	}

	double cost=0.0;
	if (prm->contCost==CONTCOST_L1)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mE)
			cost+=fabs(DIRECT_A2D_ELEM(mE,i,j));
		cost*=prm->iMask2Dsum;
	}
	else
		cost=-correlationIndex(mI,mPp,&mMask2D);

	//covarianceMatrix(mE, prm->C);
	//double div=computeCovarianceMatrixDivergence(prm->C0,prm->C)/MAT_XSIZE(prm->C);
#ifdef DEBUG
	std::cout << "A=" << A << std::endl;
	Image<double> save;
	save()=a*prm->P()+b;
	save.write("PPPtheo.xmp");
	save()=prm->Pp();
	save.write("PPPp.xmp");
	save.write("PPP.xmp");
	save()=prm->E();
	save.write("PPPe.xmp");
	//save()=prm->C;
	//save.write("PPPc.xmp");
	//save()=prm->C0;
	//save.write("PPPc0.xmp");
	//std::cout << "Cost=" << cost << " Div=" << div << " avg=" << avg << std::endl;
	std::cout << "Cost=" << cost << std::endl;
	std::cout << "Press any key" << std::endl;
	char c; std::cin >> c;
#endif
	//return div+prm->penalization*fabs(avg);
	//return cost+prm->penalization*fabs(avg);
	//return div;
	return cost;
}


double continuousCost(double *x, void *_prm)
{
	double a=x[1];
	double b=x[2];
	double deltax=x[3];
	double deltay=x[4];
	double scalex=x[5];
	double scaley=x[6];
	double scaleAngle=x[7];
	double deltaRot=x[8];
	double deltaTilt=x[9];
	double deltaPsi=x[10];
	double deltaDefocusU=x[11];
	double deltaDefocusV=x[12];
	double deltaDefocusAngle=x[13];
	auto *prm=(ProgContinuousCreateResiduals *)_prm;
	if (prm->maxShift>0 && deltax*deltax+deltay*deltay>prm->maxShift*prm->maxShift)
		return 1e38;
	if (fabs(scalex)>prm->maxScale || fabs(scaley)>prm->maxScale)
		return 1e38;
	if (fabs(deltaRot)>prm->maxAngularChange || fabs(deltaTilt)>prm->maxAngularChange || fabs(deltaPsi)>prm->maxAngularChange)
		return 1e38;
	if (fabs(a-prm->old_grayA)>prm->maxA)
		return 1e38;
	if (fabs(b)>prm->maxB*prm->Istddev)
		return 1e38;
	if (fabs(deltaDefocusU)>prm->maxDefocusChange || fabs(deltaDefocusV)>prm->maxDefocusChange)
		return 1e38;
//	MAT_ELEM(prm->A,0,0)=1+scalex;
//	MAT_ELEM(prm->A,1,1)=1+scaley;
// In Matlab
//	syms sx sy t
//	R=[cos(t) -sin(t); sin(t) cos(t)]
//	S=[1+sx 0; 0 1+sy]
//	simple(transpose(R)*S*R)
//	[ sx - sx*sin(t)^2 + sy*sin(t)^2 + 1,            -sin(2*t)*(sx/2 - sy/2)]
//	[            -sin(2*t)*(sx/2 - sy/2), sy + sx*sin(t)^2 - sy*sin(t)^2 + 1]
	double sin2_t=sin(scaleAngle)*sin(scaleAngle);
	double sin_2t=sin(2*scaleAngle);
	MAT_ELEM(prm->A,0,0)=1+scalex+(scaley-scalex)*sin2_t;
	MAT_ELEM(prm->A,0,1)=0.5*(scaley-scalex)*sin_2t;
	MAT_ELEM(prm->A,1,0)=MAT_ELEM(prm->A,0,1);
	MAT_ELEM(prm->A,1,1)=1+scaley-(scaley-scalex)*sin2_t;
	MAT_ELEM(prm->A,0,2)=prm->old_shiftX+deltax;
	MAT_ELEM(prm->A,1,2)=prm->old_shiftY+deltay;
	return transformImage(prm,prm->old_rot+deltaRot, prm->old_tilt+deltaTilt,
		prm->old_psi+deltaPsi, a, b, prm->A, deltaDefocusU, deltaDefocusV, 
		deltaDefocusAngle, xmipp_transformation::LINEAR);
}

// Predict =================================================================
//#define DEBUG
void ProgContinuousCreateResiduals::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    // Read input image and initial parameters
//  ApplyGeoParams geoParams;
//	geoParams.only_apply_shifts=false;
//	geoParams.wrap=DONT_WRAP;

    //AJ some definitions
    double corrIdx = 0.0;
    double corrMask = 0.0;
    double corrWeight = 0.0;
    double imedDist = 0.0;

	if (verbose>=2)
		std::cout << "Processing " << fnImg << std::endl;
	I.read(fnImg);
	I().setXmippOrigin();
	Istddev=I().computeStddev();

	old_rot = rowIn.getValueOrDefault(MDL_ANGLE_ROT, 0.);
	old_tilt = rowIn.getValueOrDefault(MDL_ANGLE_TILT, 0.);
	old_psi = rowIn.getValueOrDefault(MDL_ANGLE_PSI, 0.);
	old_shiftX = rowIn.getValueOrDefault(MDL_SHIFT_X, 0.);
	old_shiftY = rowIn.getValueOrDefault(MDL_SHIFT_Y, 0.);
	old_flip = rowIn.getValueOrDefault(MDL_FLIP, false);
	double old_scaleX=0, old_scaleY=0, old_scaleAngle=0;
	old_grayA=1;
	old_grayB=0;
	if (rowIn.containsLabel(MDL_CONTINUOUS_SCALE_X))
	{
		old_scaleX = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_X);
		old_scaleY = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_Y);
		if (rowIn.containsLabel(MDL_CONTINUOUS_SCALE_ANGLE))
			old_scaleAngle = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_ANGLE);
		old_shiftX = rowIn.getValue<double>(MDL_CONTINUOUS_X);
		old_shiftY = rowIn.getValue<double>(MDL_CONTINUOUS_Y);
		old_flip = rowIn.getValue<bool>(MDL_CONTINUOUS_FLIP);
	}

	if (optimizeGrayValues && rowIn.containsLabel(MDL_CONTINUOUS_GRAY_A))
	{
		old_grayA = rowIn.getValue<double>(MDL_CONTINUOUS_GRAY_A);
		old_grayB = rowIn.getValue<double>(MDL_CONTINUOUS_GRAY_B);
	}

	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && !ignoreCTF)
	{
		hasCTF=true;
		ctf.readFromMdRow(rowIn);
		ctf.produceSideInfo();
		old_defocusU=ctf.DeltafU;
		old_defocusV=ctf.DeltafV;
		old_defocusAngle=ctf.azimuthal_angle;
		updateCTFImage(old_defocusU,old_defocusV,old_defocusAngle);
		fftTransformer.FourierTransform(I(),fftE,false);
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(fftE)
			DIRECT_A2D_ELEM(fftE,i,j)*=DIRECT_A2D_ELEM(*ctfEnvelope,i,j);
		fftTransformer.inverseFourierTransform();
	}
	else
		hasCTF=false;

    Matrix1D<double> p(13), steps(13);
    // COSS: Gray values are optimized in transform_image_adjust_gray_values
    if (optimizeGrayValues)
    {
		p(0)=old_grayA; // a in P'=a*P+b
		p(1)=old_grayB; // b in P'=a*P+b
    }
    else
    {
		p(0)=1; // a in P'=a*P+b
		p(1)=0; // b in P'=a*P+b
    }
    p(4)=old_scaleX;
    p(5)=old_scaleY;
    p(6)=old_scaleAngle;

    // default values
    if (fnResiduals != "") {
      rowOut.setValue<String>(MDL_IMAGE_RESIDUAL, "");
    }
    if (fnProjections != "") {
      rowOut.setValue<String>(MDL_IMAGE_REF, "");
    }

    // Optimize
	double cost=-1;
	if (fabs(old_scaleX)>maxScale || fabs(old_scaleY)>maxScale)
    	rowOut.setValue(MDL_ENABLED,-1);
	else
	{
		try
		{
			cost=1e38;
			int iter;
			steps.initZeros();
			if (optimizeGrayValues)
				steps(0)=steps(1)=1.;
			if (optimizeShift)
				steps(2)=steps(3)=1.;
			if (optimizeScale)
				steps(4)=steps(5)=steps(6)=1.;
			if (optimizeAngles)
				steps(7)=steps(8)=steps(9)=1.;
			if (optimizeDefocus) {
				if (sameDefocus)
					steps(10)=steps(12)=1.; 
				else {
					steps(10)=steps(11)=steps(12)=1.;
					if (hasCTF)
					{
						currentDefocusU = old_defocusU;
						currentDefocusV = old_defocusV;
						currentAngle = old_defocusAngle;
					}
					else
						currentDefocusU = currentDefocusV = currentAngle = 0;
				}
			}
			powellOptimizer(p, 1, 13, &continuousCost, this, 0.01, cost, iter, steps, verbose>=2);
			if (cost>1e30 || (cost>0 && contCost==CONTCOST_CORR))
			{
				rowOut.setValue(MDL_ENABLED,-1);
				p.initZeros();
				if (optimizeGrayValues)
				{
					p(0)=old_grayA; // a in P'=a*P+b
					p(1)=old_grayB; // b in P'=a*P+b
				}
				else
				{
					p(0)=1;
					p(1)=0;
				}
			    p(4)=old_scaleX;
			    p(5)=old_scaleY;
			    p(6)=old_scaleAngle;
			}
			else
			{
				//Calculating several similarity measures between I and Pp (correlations and imed)
				corrIdx = correlationIndex(I(), Pp());
				corrMask = correlationMasked(I(), Pp());
				corrWeight = correlationWeighted(I(), Pp());
				imedDist = imedDistance(I(), Pp());

				if (fnResiduals!="")
				{
					FileName fnResidual;
					fnResidual.compose(fnImgOut.getPrefixNumber(),fnResiduals);
					E.write(fnResidual);
					rowOut.setValue(MDL_IMAGE_RESIDUAL,fnResidual);
				}
				if (fnProjections!="")
				{
					FileName fnProjection;
					fnProjection.compose(fnImgOut.getPrefixNumber(),fnProjections);
					P.write(fnProjection);
					rowOut.setValue(MDL_IMAGE_REF,fnProjection);
				}
			}
			if (contCost==CONTCOST_CORR)
				cost=-cost;
			if (verbose>=2)
				std::cout << "I'=" << p(0) << "*I" << "+" << p(1) << " Dshift=(" << p(2) << "," << p(3) << ") "
				          << "scale=(" << 1+p(4) << "," << 1+p(5) << ", angle=" << p(6) << ") Drot=" << p(7) << " Dtilt=" << p(8)
				          << " Dpsi=" << p(9) << " DU=" << p(10) << " DV=" << p(11) << " Dalpha=" << p(12) << std::endl;
			A(0,2)=p(2)+old_shiftX;
			A(1,2)=p(3)+old_shiftY;
			double scalex=p(4);
			double scaley=p(5);
			double scaleAngle=p(6);
			double sin2_t=sin(scaleAngle)*sin(scaleAngle);
			double sin_2t=sin(2*scaleAngle);
			A(0,0)=1+scalex+(scaley-scalex)*sin2_t;
			A(0,1)=0.5*(scaley-scalex)*sin_2t;
			A(1,0)=A(0,1);
			A(1,1)=1+scaley-(scaley-scalex)*sin2_t;
//			A(0,0)=1+p(4);
//			A(1,1)=1+p(5);

			if (old_flip)
			{
				MAT_ELEM(A,0,0)*=-1;
				MAT_ELEM(A,0,1)*=-1;
				MAT_ELEM(A,0,2)*=-1;
			}
		}
		catch (XmippError &XE)
		{
			std::cerr << XE.what() << std::endl;
			std::cerr << "Warning: Cannot refine " << fnImg << std::endl;
			rowOut.setValue(MDL_ENABLED,-1);
		}
	}
    rowOut.setValue(MDL_IMAGE_ORIGINAL, fnImg);
    rowOut.setValue(MDL_ANGLE_ROT,  old_rot+p(7));
    rowOut.setValue(MDL_ANGLE_TILT, old_tilt+p(8));
    rowOut.setValue(MDL_ANGLE_PSI,  old_psi+p(9));
    rowOut.setValue(MDL_SHIFT_X,    0.);
    rowOut.setValue(MDL_SHIFT_Y,    0.);
    rowOut.setValue(MDL_FLIP,       false);
    rowOut.setValue(MDL_COST,       cost);
    if (optimizeGrayValues)
    {
		rowOut.setValue(MDL_CONTINUOUS_GRAY_A,p(0));
		rowOut.setValue(MDL_CONTINUOUS_GRAY_B,p(1));
    }
    rowOut.setValue(MDL_CONTINUOUS_SCALE_X,p(4));
    rowOut.setValue(MDL_CONTINUOUS_SCALE_Y,p(5));
    rowOut.setValue(MDL_CONTINUOUS_SCALE_ANGLE,p(6));
    rowOut.setValue(MDL_CONTINUOUS_X,p(2)+old_shiftX);
    rowOut.setValue(MDL_CONTINUOUS_Y,p(3)+old_shiftY);
    rowOut.setValue(MDL_CONTINUOUS_FLIP,old_flip);
    if (hasCTF)
    {
    	rowOut.setValue(MDL_CTF_DEFOCUSU,old_defocusU+p(10));
		if (sameDefocus)
    		rowOut.setValue(MDL_CTF_DEFOCUSV,old_defocusU+p(10)); 
		else
			rowOut.setValue(MDL_CTF_DEFOCUSV,old_defocusV+p(11));
    	rowOut.setValue(MDL_CTF_DEFOCUS_ANGLE,old_defocusAngle+p(12));
		if (sameDefocus)
			rowOut.setValue(MDL_CTF_DEFOCUS_CHANGE,0.5*(p(10)+p(10)));
		else
			rowOut.setValue(MDL_CTF_DEFOCUS_CHANGE,0.5*(p(10)+p(11)));
    	if (old_defocusU+p(10)<0 || old_defocusU+p(11)<0)
    		rowOut.setValue(MDL_ENABLED,-1);
    }
    //Saving correlation and imed values in the metadata
    rowOut.setValue(MDL_CORRELATION_IDX, corrIdx);
    rowOut.setValue(MDL_CORRELATION_MASK, corrMask);
    rowOut.setValue(MDL_CORRELATION_WEIGHT, corrWeight);
    rowOut.setValue(MDL_IMED, imedDist);


#ifdef DEBUG
    std::cout << "p=" << p << std::endl;
    MetaDataVec MDaux;
    MDaux.addRow(rowOut);
    MDaux.write("PPPmd.xmd");
    Image<double> save;
    save()=P();
    save.write("PPPprojection.xmp");
    save()=I();
    save.write("PPPexperimental.xmp");
    //save()=C;
    //save.write("PPPC.xmp");
    Pp.write("PPPprojectionp.xmp");
    E.write("PPPresidual.xmp");
    std::cout << A << std::endl;
    std::cout << fnImgOut << " rewritten\n";
    std::cout << "Press any key" << std::endl;
    char c; std::cin >> c;
#endif
}
#undef DEBUG

void ProgContinuousCreateResiduals::postProcess()
{
	MetaData &ptrMdOut = getOutputMd();
	if (contCost==CONTCOST_L1)
	{
		double minCost=1e38;
        for (size_t objId : ptrMdOut.ids())
		{
			double cost;
			ptrMdOut.getValue(MDL_COST,cost,objId);
			if (cost<minCost)
				minCost=cost;
		}
        for (size_t objId : ptrMdOut.ids())
		{
			double cost;
			ptrMdOut.getValue(MDL_COST,cost,objId);
			ptrMdOut.setValue(MDL_WEIGHT_CONTINUOUS2,minCost/cost,objId);
		}
	}
	else
	{
		double maxCost=-1e38;
        for (size_t objId : ptrMdOut.ids())
		{
			double cost;
			ptrMdOut.getValue(MDL_COST,cost,objId);
			if (cost>maxCost)
				maxCost=cost;
		}
        for (size_t objId : ptrMdOut.ids())
		{
			double cost;
			ptrMdOut.getValue(MDL_COST,cost,objId);
			ptrMdOut.setValue(MDL_WEIGHT_CONTINUOUS2,cost/maxCost,objId);
		}
	}

	ptrMdOut.write(fn_out.replaceExtension("xmd"));
}
