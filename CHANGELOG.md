## Release 5.0.0 - Bxxxx
   ### Xmipp Programs 
   - New programs
      - alignPCA_2D: an alignment method applied for 2D classification using PCA and Euclidean distance.‎

   ### Installation and user guide
   - Redirect all documentation to the official and fancy one: https://i2pc.github.io/docs/
   - ??pyXmipp?
     
   ### More Xmipp  
   - ??CUDA 13 is now fully compatible with Xmipp, enabling improved GPU acceleration and performance on modern hardware.


## Release 4.0.0 - Apoferritin
   ### Xmipp Programs 
   - Programs updated
      - align volume and particles: Allow considering mirrors
      - miccCleaner: updated enviroment to tensorflow2
        
   - Programs fixed
      - tranform geometry: Apply geometry program now considers the original geometry in the metadata when the parameter --shift_to is passed.
      - ‎angular_project_library: Fix indexing problem in angular_project_library
      - ctf_correct_phase fix and two constants for tomography were added
      - angular_project_library: Fix indexing bug

   ### Installation and user guide
   - Fixed issue with CUDA 12.5
     
   ### More Xmipp  
   - Forced compiler GCC >= 9 
   - getPSF added to the xmipp binding
   - replace "conda env export" to avoid deprecation issue.
   - Fixing read when encountering NaN-s and infinite values


## Release 3.25.06.0 - Rhea
   ### Xmipp Programs 
   - New programs
      - continuous_create_residuals: Make better residuals by continuous assignment updates to projections
      - tomo_detect_landmarks: to detect landmarks in tilt series (TS)
      - tomo_calculate_landmark_residuals: to calculate residuals in TS
      - tomo_detect_misalignment_residuals: to detect misalignment from residuals in TS
      - batch_tomo_misalignment_resid_statistics: Python script to calculate statistical metrics from a set of residuals
      - tomo_tiltseries_detect_misalignment_corr: (to legacy) for calculating relative shifts between tilt images, as a prealignment strategy

   - Programs updated
      - Particle subtraction: now generates a single stack + includes noise power estimation
      - classify_pca: allows setting CUDA_DEVICE for NVIDIA drivers ≤ 470
        
   - Programs fixed
      - Fixed missing b in angular_continuous_assign2_gpu
      - CUDAFFT test fixed
      - Minor compatibility fixes for CUDA 12.6
      - DT_SChar added to micrographs
        
   ### Installation and user guide
   - Fixed broken links in documentation
   - Enhanced data collection steps (context suggests improvements in reproducibility or logging)
     
   ### More Xmipp  
   - DLTK environments: channels limited to conda-forge and pip. Updated deepEMhancer version and models, update all enviroments
   - Allowing to skip sorting in metadata_vec::split

## Release 3.24.12.0 - Poseidon
   ### Xmipp Programs 
   - New programs
        - tomo_average_subtomos: This method carries out a subtomogram averaging. It means, given a set of subtomogram the program will estimate the average map
        - denoising_tv: The program will make denoising using total variation
        - tomo_ctf_wiener2d_correction: This method performs a CTF correction of a tilt series
        - angular_continuous_assign2_gpu: Make a continuous angular assignment with GPU
        - cuda_fourier_projection
        - predict_deep_center
        - cl2d_clustering

   - Programs updated
      - tomogram_reconstruction:  averaging with gold standard, tigre with internal interpolation, enviroment of tigre updated
      - mpi_subtract_projection / subtract_projection: Improvements on projection subtraction a particle boosting both in MPI and single thread version

   - Programs fixed
      - resolution_directional
      - ctf_estimate_from_psd: Bug fix with no defocus
      - mpi_subtract_projection / subtract_projection: subtraction a particle boosting both in MPI and single thread version
      - volume_halves_restoration_gpu: Fixed failure when using masks

   ### Installation and user guide
   - Total new installer based on CMake. More robust and simplified
   - New posting of statistics about installations
   - Allow re-compilation without internet
   - Added next generation of libstdcxx to DLTKv0.3 env to enhance compatibility
   - Forcing to compile Xmipp with the libstdc++ availabe in the scipion3 env if exist
     
   ### More Xmipp  
   - Added support for reading EER
   - Handling .gain files correctly
   - Added MADV_HUGEPAGE support check
   - Fix for rare axis orderings in MRC/MAP files
   - Local average label added (used to measure the intensity of the particles in the original micrographs)
 
## Release 3.24.06 - Oceanus
   ### Xmipp Programs 
   - New programs
      - xtomo tigre reconstruction: This program provides a variety of algorithms to reconstruct tomogram from a set of tilt series
      - graph_max_cut: split volume program
      - tomo_simulate_tilt_series: simualtion of ts, tomograms and coordinates
   - Programs updated
      - Deep_center: interface simplified, calculate shift for particles
      - deep_misalignment_detection: update model
      - pdb_label_from_volume:Added CIF format support
      - resolution_monotomo: Fast subtomo extraction
      - resolution_monotomo: accelarating monotomo with float
      - tomogram_reconstruction: tigre reconstruction added
      - tomo_extract_subtomograms: solving some bugs and enhancing performance
      - tomo_extract_particlestacks: solving some bugs and enhancing performance
   - Programs fixed
      - movie_estimate_gain: Initialize sigma values
      - resolution_localfilter: double not integer
      - movie_estimate_gain: Initialize sigma values
      - project_tomography: internal loop
      - pdb: Fixed PDB write function when no categories are present
      - fourier_filter:Fix WEDGE option in FourierFilter
      - resolution_pdb_bfactor:fix unaccounted column in res bfac pdb
      - fourier_filter: Fixing missing cone
   ### Installation and user guide
   - Fixed PyXmipp installation
   - Update pythorch version on DLTK pyTorch enviroment
   - Fix DLTK installation
   - Removed force flag on conda installations (new conda version)
   ### More Xmipp 
   - API for collecting xmipp installation statistics
   - Local average added: Now the micrograph scissor annotates the local intensity value around the particles in the original micrographs.
   - Added support for half precision float micrographs
   - Wipe out memory ouptut in cuFFT
   - Deprecated deep_denoising
   - Fix for rare axis orderings in MRC/MAP files
   - Adding fixed extension to tomo

  
## Release 3.23.11 - Nereus
   ### Xmipp Programs 
   
   - New programs
      - local_volume_adjustment: This program modifies a volume in order to adjust its intensity locally to a reference volume
      - subtomo_subtraction: New program with mpi version
   - Programs updated
      - tomo_extract_subtomogram: Accept target box size
      - convert_from_pdb: Added cif support
      - phantom_movies: Adding more options: circle and cross
      - angular_neighbourhood: Modified to output the result in the best symmetry group
      - batch_deep_misalignment: Detection: GPU option 
   - Programs fixed
      - volume_from_pdb: Bug fix in dealing with orig=0, Detection of contrast inverted maps
      - tomo_extract_particles: Stack fixes, dose fixes
      - image_convert: Fixed issue when providing a metadata file as input 
      - volume_substraction: Change NaNs generated by a division between 0 by 0s.
        
   ### Installation and user guide
   - Disabled downloading 380MB file not required
   - Clarifying the requirements 
   - More robust thread execution in the installator
   - Fixed bug collecting versions
   - Removed auto updater from libcifpp

   ### More Xmipp 
   - Upgraded CUDA toolkit in DLTK 1.0
   - Fix PDB read: This fixes a bug where pdbs whose atom name start in position 12 (in base 0)
   - CUDA 12.2 supported (not full tested)
   - Fixed PDB reading for shorter PDBs
   - Allow not compiling xmipp to debug changes in python scripts
   - Xmipp will now be able to read any volume with any values of those fields correctly.
   - Fixed pixel size not being properly stored
   - Fixed MetaData to avoid dangling references
   - Added compatibility for undefined axis sequences in MRC files
   - Added option to write mirrors in angularDistance


   ## Release 3.23.07 - Morpheus
   ### Xmipp Programs 
   - New programs
      - image_peak_high_contrast(for detecting high contrast regions in tomographic reconstruction)
      - misaligment_detection (to detect misalignment in tomographic reconstructions from high-contrast regions)
      - deep_global_assignment
      - deep_center
   - Programs updated
      - xmipp_angular_distance: new features
      - tomo_extract_particles: new features
      - subtract_projection: parallelization with mpi
      - tomo_extract_subtomograms: allow downsampling of features
      - angular_resolution_alignment:  Detect misalignment with resolution
   - Programs fixed
      - align_volume_and_particles: Fixed error  
      - angular_project_library: Removed deterministic behaviour (mpi)
      - volumen_subtraction: fixed bug
   - Deprecated programs (For more details [visit](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols)) 
      - classify_kmeans_2D
      - rotational_spectra
      - particle_boxsize
   
   ### Installation and user guide
   - New clearer, more intuitive and informative installer. It also creates a file to facilitate user support.
   - Refactored the deep learning toolkit, more robust and new environment with updated tensorflow.
   - Updated requirement.
     
   ### More Xmipp 
   - Added half precission suport to numpy
   - Added the ability to read and write CIF files
   - Modular design of winner filter
   - Decoupling XmippTomo from XmippSPA
   - Fix Zernike equation
   - Removed all occurrences of non base-class default destructors
   - Improved MultidimArray performance
   - Added support for half precision floating point numbers in MRCs
   - Assign tiff to gain files
   - adding half maps labels  


   
## Release 3.23.03 - Kratos
   ### Xmipp Programs 
   
  - New programs: tomo_confidence_map, tomo_extract_particlestacks, tomo_extract_subtomograms, tomo_tiltseries_dose_filter, psd_estimatator
  - Deprecated programs (for more details visit the [wiki](https://github.com/I2PC/xmipp/wiki/Deprecating-programs)):
    angular_distribution_show, apropos
    ctf_correct_idr, ctf_create_ctfdat , ctf_show , idr_xray_tomo , image_common_lines , metadata_convert_to_spider , metadata_selfile_create , mlf_refine_3d, ml_refine_3d, ml_tomo , mrc_create_metadata , pdb_construct_dictionary, pdb_restore_with_dictionary , reconstruct_admn , reconstruct_art_pseudo , resolution_ibw , resolution_ssnr , score_micrograph , reconstruct_fourier_starpu , tomo_align_tilt_series, tomo_align_dual_tilt_series, tomo_align_refinement, tomo_align_refinement, tomo_extract_subvolume, tomo_project_main, tomo_remove_fluctuations , tomo_align_tilt_series,transform_range_adjust , validation_tilt_pairs , volume_pca , volume_validate_pca , work_test , 6f4d983 , evaulate_coordinates , extract_subset , image_separate_objects , volume_enhance_contrast , volume_reslice , xray_import , xray_project , xray_psf_create , xray_reconstruct_art , gpu_correlation, gpu_utils, classify_significant, deepAlign.

  - volume_from_pdb: fixing input pdb file being overwritten when '-centerPDB' flag was set
  - xmipp_phantom_movie:  adding support for fixed step shift & gain and dark image generation
  - CTF simulation allows astigmatism
  - xmipp_metadata_utility: Now join operations with an empty set will return a new empty set (previously no output file was generated).
  - xmipp_matrix_dimred: Program help improved. Exception is now thrown when the number of output dimensions is larger than the input dimensions
  - xmipp_angular_distance: Added itemId column to the output

   ### Installation and user guide
   - Refactor and simplified Readme page.
   - Updating CUDA version compatibility
   - Updating gcc version availables
   - Fixed Matlab installation
   - Added missing array include to fix compilation error with g++12
   - Alert and not block compilation if gcc - CUDA are not compatible
   - Avoid compilation warnings
   - Required pyworkflow==3.0.31


   ### Others
   - Maintenance: Recovered python binding tests
   - Maintenance: fixing dangling pointer in xmipp_error
   - Maintenance: Cleaned includes in xmipp_image_base
   - PSD estimation: templating function, improving performance
   - Flag cleanDeprecate in the installation; clean all deprecated executables programs 
   - python binding: fixed bug when Numpy arrays created by slicing were badly interpretted
   - Removed "seed" library
   - Fixed memory pinning CUDA bug
   - Fixed compilation errors on CUDA 9



   ## Release 3.22.11 - Iris

  ### Xmipp Programs
  -  Speeding up iterations in some xmipp programs (xmipp_ctf_group, xmipp_image_histogram, xmipp_mpi_angular_class_average, xmipp_angular_distance, xmipp_angular_estimate_tilt_axis, xmipp_ctf_create_ctfdat, xmipp_resolution_ssnr)
  -  New Zernike3D programs
  -  angular_project_library: Reported some error if there are no images in the range
  -  angular_discrete_assign.cpp: Removed memory leak and uninitialized values
  -  angular_distance: Fixing condition to avoid iteration behind the end of the MD in cases when input data have different sizes. Optimized performance
  -  Pdb_reduce_pseudoatoms: Produced pdb is one-based indexed
  -  xmipp_micrograph_automatic_picking: Fixing memory leak
  -  subtract_projection: Fixed several bugs (improved results), added circular mask to avoid edge artifacts, added option to boost particles instead of subtract

  
  ### Installation and user guide
  - Various bug fixing
  - More information about hdf5 library
  - Updating CUDA - GCC compatibility. Added CUDA 11.7 (not tested)
  - Updating Readme
  
  ### Others
  - Performance optimization (metadata binding)
  - Python binding: adding methods to directly set / get entire MD row
  - g++ >= 8 required
  - In viewers used pwutils 
  - The pdb data library now has all the right fields and should write the record type ("ATOM " or "HETATM") correctly at the beginning of the line and the atomType (element) and charge (if applicable) correctly at the end of the line.
  - Removal of an artifact of symmetrization related to the z pitch (symmetries.cpp)
  - Using the same identical Deprecated param from pyworkflow.


## Release 3.22.07 - Helios


### Scripts Xmipp
- **xmipp_image_operate**: taked into account non existing files
- **angular_continuous_assign2**: Bug fixed
- **volume_consensus**: Bug fixed
- **ctf.h and angular_continuous_assign_2**: Changes for local defocus estimation #578

### Installation and user guide
- Version info printed at the end of the installation
- Removed empty folder with cleanBin command
- Clarifing linking to Scipion and removed a bug with the build link
- New flag (OPENCV_VERSION) in xmipp.config
- Updated Readme (explain OpenCV-CUDA support)

### Others
- Validation server:  Merged what remains
- Replaced sincos to sin and cos
- Handling of pointers in MPI programs
- "nullptr" used to denote the null pointer not "NULL"
- Check if nvidiaDriverVer is None

## Release 3.22.04 - Gaia

### Installation and user guide
- Updated readme
  - Updated hdf5 info troubleshoting
  - Updated Standalone installation 
  - Updated Scons installation 
- xmipp get_models: fixing the run and download path
- Updating xmipp links for Scipion on installation
- Removed fatal message in installation
- Reported error if happen on installation - runjob
- Ensuring that target directory for the libraries exists


### Protocols scipion-em-xmipp
- **protocol_core_analysis**: New protocol
- **protocol_compare_angles**: Bug fix in compare angles under some conditions
- **protocol_center_particles**: protocol simplified (removed setofCoordinates as output)
- **protocol_CTF_consensus**: concurrency error fixed
- **protocol_convert_pdb**: remove size if deactivated
- **protocol_resolution_deepres**: binary masked not stored in Extra folder and avoiding memory problems on GPUs
- **protocol_add_noise**: fixes
- **protocol_compare_reprojections**: improve computation of residuals + tests + fix + formatting
- **protocol_screen_deepConsensus**: multiple fixes in batch processing, trainging and streaming mode
- **protocol_shift_particles**: apply transform is now optional
### Others
- New XMIPP logo
- subtract_projection: adding new flag + fix
- Add intersection size metadata (bindings/python)
- Fixed unitialized unique pointers (bindings/python)
- Bug fixing: Resolution directional and anisotropic filtering fixing the test
- Removed SonarCloud issues
  - Replaced defines with constexpr
  - Removing Unused funtion parameters
  - Division by zero
  - Memory management
  - Removed field shadowing
  - Destructors should not throw exceptions

## Release 3.22.01 - Eris


- Updating to C++17
- Support newer versions of CUDA and gcc
- Zernike programs compatible with Cuda 8.x
- Fixed Sonar Cloud issues and bugs
- Matlab compilation Fixed
- Fixed importing pwem.metadata
- nma_alignment: Fixed arguments for the xmipp_angular_projection_matching invocation
- Fixed test  fails: ResolutionSsnr, ReconstructArtMpi, ReconstructArt, MlfRefine3dMpi, MlfRefine3d, MlRefine3dMpi, MlRefine3d, xmipp_test_pocs_main & volume_subtraction
- xmipp_micrograph_automatic_picking: Fixed tests, avoid possible memory corruption
- resolution_pdb_bfactor: bug fixed -  error with multiple chains
- FlexAlign: Fixed crash when binning > 1
- Bug fixed and allowed controlling high sampling rate
- Volume consensus: Fixed number of levels in the wavelet transform
- Compilation: Fixed compilation of starpu programs
- xmipp_transform_dimred: Fixed output metadata in append mode, adding MDL_DIMRED label
- Config file generation: Fixed config version detection outside of the git repo,  refactored check_CUDA and managed gcc compiler if it is installed out of /usr/bin/, check and exit if xmipp.conf does not exist
- Compilation: Fixed detection of the last commit changed the config script
- Resolution_fso: Bingham test implemented
- Opencv not detected. Added include to user/include/opencv4 folder on config file
- Compilation: asking whether to continue with compilation even though the config file is outdated
- XMIPP install: Linked libsvm to scipion
- Installation: Referenced 'global' xmipp.conf instead of using local copy of it
- Multiple MPI programs: replaced CREATE_MPI_METADATA_PROGRAM macro by templated class
- python_constants: add defocus labels
- Metadata: added new nmaEigenval label
- Python binding: added new function -  correlationAfterAlignment, MDL_RESOLUTION_ANISOTROPY, MDL_RESOLUTION_ANISOTROPY
- Matlab binding dependencies: set XMIPP as a hard dependency
- Projections subtraction: new program
- FFTwT: added mutex for plan handling
- Multiple programs: Added a common implementation of the rerun
- Phantom_create: update info link
- Multiple programs: Added a common implementation of the rerun
- Transform Geometry: save new shifted coordinates in option "shift to" + enterOfMass to python binding 
- Readme info: add virtual machine info
- Removal of the SVM from inside the XMIPP repository and downloading it as an external dependence
- Solved a configuration problem with CUDA
- ml_tomo: Using .mrc instead of .vol ; volume_align: Addded wrapping during alignment


## Release 3.21.06 - Caerus

- CUDA-11 support
- New protocol: Deep align
- ChimeraX support
- Improvements of streaming process
- Several performance optimizations
- Build time optimization
- Multiple bug fixes
- Improved documentation


## Release 3.20.07 - Boreas

- Fast CTF estimation
- CTF includes phase shifts now
- Selection of alpha helices or beta sheets from a PDB (xmipp_pdb_select)
- Centering a PDB (xmipp_pdb_center)
- New Protocol: MicrographCleaner is a new algorithm that removes coordinates picked from carbon edges, aggregations, ice crystals and other contaminations
- New functionality: The protocol compare reprojections can now compute the residuals after alignment
- New protocol: Split frames divide input movies into odd and even movies so that they can be processed independently
- New protocol: Continuous heterogeneity analysis using spherical harmonics (not ready to be used)
- Bug fixing when some micrograph has no coordinates in the consensus-picking.
- New functionalities: Different architectures and training modes
- Normal Mode Analysis protocols have been moved to the plugin ContinuousFlex
- Fixing MPI version of the Fourier Reconstruction
- New protocol: local CTF integration and consensus protocol for local ctf (also the viewers)
- Local CTF analysis tools: Not yet ready for general public
- New functionallity: Introducing the posibility of automatic estimation of the gain orientation.
- Bugs fixings regarding stability on streaming processing
- Support of heterogeneous movie sets
- New protocol: Clustering of subtomogram coordinates into connected components that can be processed independently
- New Protocol: Removing duplicated coordinates
- New protocol: Subtomograms can be projected in several ways to 2D images so that 2D clustering tools can be used
- New protocol: Regions of Interest can be defined in tomograms (e.g., membranes)
- Bug fixing in mask3d protocol
- Bug fix: in helical search symmetry protocol
- Enhanced precision of the FlexAlign program
- Now, deepLearningToolkit is under its own conda environment
- Multiple protocols accelerated using GPU
- New functionality: Xmipp CTF estimation can now take a previous defocus and do not change it
- New functionallity: CTF-consensus is able to take the primary main values or an average of the two.
- New functionallity: CTF-consensus is able to append metadata from the secondary input
- New functionality: Xmipp Highres can now work with non-phase flipped images
- New functionality: Xmipp Preprocess particles can now phase flip the images
- New protocol: Tool to evaluate the quality of a map-model fitting
- Allowing multi-GPU processing using FlexAlign
- Improvement in monores and localdeblur
- Randomize phases also available for images
- Change the plugin to the new Scipion structure
- Migrating the code to python3
