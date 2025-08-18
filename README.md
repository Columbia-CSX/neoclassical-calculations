# neoclassical-calculations
Collection of helper scripts for post-processing of SFINCS ouputs

This repository was created by Michael Campagna, Summer 2025. 

## Dependencies
Aspects of these scripts depend on [simsopt](https://github.com/hiddenSymmetries/simsopt), [sfincs](https://github.com/landreman/sfincs), [sfincsProjectsAndTools](https://github.com/landreman/sfincsProjectsAndTools.git) and [stelloptPlusSfincs](https://github.com/leebr48/stelloptPlusSfincs). Some changes have been made to sfincs and stelloptPlusSfincs to make it compatible with this code. The most notable difference is that the branch [sfincs\_boozer\_scaling](https://github.com/landreman/sfincs/tree/sfincs_boozer_scaling) must be merged with the current version of sfincs for a few of these scripts to run as expected. Additional dependencies include the python libraries tqdm, SciPy, and Astropy.  

Just in case, any modified .f90 files from sfincs are included in /sfincs/version3files. Overwriting the current version (as of July 16, 2025) with these .f90 files before making sfincs will ensure the scripts contained in this repo function as intended by including the necessary elements of sfincs\_boozer\_scaling. Note some additional adjustments to the code will likely be necessary, as machine-specific changes were made to the stelloptPlusSfincs repository as well.

## How to use

Some of the python scripts accept (or require) arguments when called from the terminal. Running any of these scripts with the --help tag will give a description of each of the possible arguments.

In order to run, sfincs needs information about the density and temperature gradients in the plasma volume. Profiles can be specified using profiles.py. A VMEC wout file and a .bc file are additionally necessary to specify the magnetic geometry. [I will update the repo to add all of the ones I used soon] Given just the wout, a .bc file can be created using the wout2bc.py script [currently adapting this script to be more user friendly].

A typical workflow might be to use runRadErScan.py to run sfincs over a range of flux surfaces, varying the radial electric field Er on each. After this is done, runAmbipolar.py may be used to parse the outputs from runRadErScan.py to run sfincs at the ambipolar Er value for each flux surface. Quantities of interest can be calculated by running processOutputs.py in either the ./ambipolar directory.

If you'd also like to scan over different values of the symmetry breaking factor $\epsilon_{sb}$, this can be done similarly by using the scripts runSBRadErScan.py, runSBAmbipolar.py, and processSBOutputs.py. Note that you should edit \_\_main\_\_ in processOutputs.py and/or processSBOutputs.py to ensure it's plotting/calculating whatever it is you're actually interested in.

Feel free to email me if you have any questions : )

mc5893@columbia.edu    <-- preferred until September 2025
macampagna@wm.edu      <-- preffered until June 2026
michampagn13@gmail.com <-- just in case
