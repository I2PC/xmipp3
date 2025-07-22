/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#include "mpi_classify_partial_occupancy.h"

void MpiProgClassifyPartialOccupancy::defineParams()
{
    ProgClassifyPartialOccupancy::defineParams();
    MpiMetadataProgram::defineParams();
}
void MpiProgClassifyPartialOccupancy::readParams()
{
    MpiMetadataProgram::readParams();
    ProgClassifyPartialOccupancy::readParams();
}
void MpiProgClassifyPartialOccupancy::read(int argc, char **argv, bool reportErrors)
{
    MpiMetadataProgram::read(argc, argv);
}
void MpiProgClassifyPartialOccupancy::preProcess()
{
    rank = (int)node->rank;

    ProgClassifyPartialOccupancy::preProcess();

    // Initialize noise power and masks for MPI
    int particleFreqMapSizeX;
    int particleFreqMapSizeY;
    int particleFreqMapOrigin;

    int noiseSizeX;
    int noiseSizeY;
    int noiseSizeOrigin;

    int radialAvgFTSize;
    double* radialAvgFTOrigin;

    if (node->rank == 0)
    {

        noiseSizeX = (int)XSIZE(powerNoise());
        noiseSizeY = (int)YSIZE(powerNoise());
        noiseSizeOrigin = STARTINGX(powerNoise());

        particleFreqMapSizeX = (int)XSIZE(particleFreqMap);
        particleFreqMapSizeY = (int)YSIZE(particleFreqMap);
        particleFreqMapOrigin = STARTINGX(particleFreqMap);

        radialAvgFTSize = radialAvgVolFT.size();
        radialAvgFTOrigin = radialAvgVolFT.data();
    }

    MPI_Bcast(&noiseSizeX,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&noiseSizeY,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&noiseSizeOrigin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&particleFreqMapSizeX,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&particleFreqMapSizeY,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&particleFreqMapOrigin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&radialAvgFTSize,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radialAvgFTOrigin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        powerNoise().resizeNoCopy(noiseSizeY,noiseSizeX);
        STARTINGX(powerNoise())=STARTINGY(powerNoise())=noiseSizeOrigin;

        particleFreqMap.resizeNoCopy(particleFreqMapSizeY, particleFreqMapSizeX);
        STARTINGX(particleFreqMap)=STARTINGY(particleFreqMap)=particleFreqMapOrigin;

        radialAvgVolFT.resize(radialAvgFTSize);
    }

    MPI_Bcast(MULTIDIM_ARRAY(powerNoise()),    (int)MULTIDIM_SIZE(powerNoise()),    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(MULTIDIM_ARRAY(particleFreqMap), (int)MULTIDIM_SIZE(particleFreqMap), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(radialAvgVolFT.data(), radialAvgVolFT.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize Fourier projector for MPI
    int realSizeX;
    int realSizeY;
    int realSizeZ;
    int origin;

    if (!realSpaceProjector)
    {
        if (node->rank == 0)
        {
            realSizeX = (int)XSIZE(projector->VfourierRealCoefs);
            realSizeY = (int)YSIZE(projector->VfourierRealCoefs);
            realSizeZ = (int)ZSIZE(projector->VfourierRealCoefs);
            origin = STARTINGX(projector->VfourierRealCoefs);
        }

        MPI_Bcast(&realSizeX, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&realSizeY, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&realSizeZ, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&origin, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(projector->volumePaddedSize), 1, MPI_INT, 0, MPI_COMM_WORLD); 
        MPI_Bcast(&projector->volumeSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            projector->VfourierRealCoefs.resizeNoCopy(realSizeZ,realSizeY,realSizeX);
            projector->VfourierImagCoefs.resizeNoCopy(realSizeZ,realSizeY,realSizeX);
            STARTINGX(projector->VfourierRealCoefs)=STARTINGY(projector->VfourierRealCoefs)=STARTINGZ(projector->VfourierRealCoefs)=origin;
            STARTINGX(projector->VfourierImagCoefs)=STARTINGY(projector->VfourierImagCoefs)=STARTINGZ(projector->VfourierImagCoefs)=origin;
        }

        MPI_Bcast(MULTIDIM_ARRAY(projector->VfourierRealCoefs), (int)MULTIDIM_SIZE(projector->VfourierRealCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(MULTIDIM_ARRAY(projector->VfourierImagCoefs), (int)MULTIDIM_SIZE(projector->VfourierImagCoefs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            projector->produceSideInfoProjection();
        }
    }

    MetaData &mdIn = *getInputMd();
    mdIn.addLabel(MDL_GATHER_ID);
    mdIn.fillLinear(MDL_GATHER_ID, 1, 1);
    createTaskDistributor(mdIn, blockSize);
}
void MpiProgClassifyPartialOccupancy::startProcessing()
{
    if (node->rank == 1)
    {
        verbose = 1;
        ProgClassifyPartialOccupancy::startProcessing();
    }
    node->barrierWait();
}
void MpiProgClassifyPartialOccupancy::showProgress() 
{
    if (node->rank == 1)
    {
        time_bar_done = first + 1;
        ProgClassifyPartialOccupancy::showProgress();
    }
}

void MpiProgClassifyPartialOccupancy::finishProcessing()
{
    node->gatherMetadatas(getOutputMd(), fn_out);
    MetaDataVec MDaux;
    MDaux.sort(getOutputMd(), MDL_GATHER_ID);
    MDaux.removeLabel(MDL_GATHER_ID);
    getOutputMd() = MDaux;
    if (node->isMaster())
        ProgClassifyPartialOccupancy::finishProcessing();
}
void MpiProgClassifyPartialOccupancy::wait()
{
    distributor->wait();
}
