/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 		jlvilas@cnb.csic.es
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

#ifndef _PROG_RESBFACTOR
#define _PROG_RESBFACTOR

#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include "data/pdb.h"




/**@defgroup Resolution B-Factor
   @ingroup ReconsLibrary */
//@{

class ProgResBFactor : public XmippProgram
{
private:
	 /** Filenames */
	FileName fnOut, fn_pdb, fn_locres;

	/** sampling rate*/
	double sampling;

	/** Number of atoms in the pdb or alpha-carbons*/
	int numberOfAtoms;

	bool medianTrue;
	bool centered;

	std::vector<double> residuesToChimera;
	double fscResolution;

    pdbInfo at_pos;


private:

    void defineParams();
    void readParams();

    /**
    * SORT_INDEXES: This function takes a vector, it is sorted from low to high and the permutation
    * indexes are returned. Example, the vector (5, 4, 7, 3), is sorted as (3, 4, 5, 7) and the output
    * of the function will be (3, 1, 0, 2), because the number 3, is in the fourth position, number 4 is
    * in the second position, and so on. Remeber that natural numbers start at 0.
    */
    template<typename T>
    std::vector<size_t> sort_indexes(const std::vector<T> &v);
 
    /**
    * SWEEPBYRESIDUE: This function creates a vector with the normalized local resolution per residue.
    * Later, this vector will be used by generateOutput to create a pdb to visualize the normalized
    * local resolution in chimera using on the pdb.
    * Also the Normalized local resolution in stored in an output metadata
    */
    void sweepByResidue(std::vector<double> &residuesToChimera_aux, 
									std::vector<double> &resNumberList, 
									std::vector<double> &resolution_per_residue, 
									std::vector<double> &bfactor_per_residue);

    /**
    * GENERATEOUTPUTPDB: The normalized local resolution per residue is taken, residuesToChimera,
    * and the values are stored in and output pdb file by substituting the bfactor column by the
    * normalized local resolution of each residue. This file has visualization purpose (in Chimera).
    */
    void generateOutputPDB(const std::vector<double> &residuesToChimera, std::vector<double> &resNumberList, 
									std::vector<double> &resolution_per_residue, 
									std::vector<double> &bfactor_per_residue);

    /**
    * For each atom position k, i, j, and a radius, totRad, the local resolution values of a local
    * resolution map, resvol, around such position and inside a sphere of radiue totRad are taken to
    * compute the mean resolution, the number of voxels, of the sphere. Also a Mask is created with
    * the sphere and the local resolution values are stored in a vector, resolution_to_estimate, to 
    * compute the median resolution in a later step.
    */
    void estimatingResolutionOfResidue(int k, int i, int j, int totRad, MultidimArray<int> &mask, const MultidimArray<double> &resvol, double &resolution_mean, int &N_elems, std::vector<double> &resolution_to_estimate);


    void run();
};
//@}
#endif
