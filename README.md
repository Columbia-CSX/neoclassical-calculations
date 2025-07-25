# neoclassical-calculations
Collection of helper scripts for post-processing of SFINCS ouputs

This repository was created by Michael Campagna, Summer 2025. 

## Dependencies
Aspects of these scripts depend on [simsopt](https://github.com/hiddenSymmetries/simsopt), [sfincs](https://github.com/landreman/sfincs), [sfincsProjectsAndTools](https://github.com/landreman/sfincsProjectsAndTools.git) and [stelloptPlusSfincs](https://github.com/leebr48/stelloptPlusSfincs). Some changes have been made to sfincs and stelloptPlusSfincs to make it compatible with this code. The most notable difference is that the branch [sfincs_boozer_scaling](https://github.com/landreman/sfincs/tree/sfincs_boozer_scaling) must be merged with the current version of sfincs for a few of these scripts to run as expected. Additional dependencies include the python libraries tqdm, SciPy, and Astropy.  

Just in case, any modified .f90 files from sfincs are included in /sfincs/version3files. Overwriting the current version (as of July 16, 2025) with these .f90 files before making sfincs will ensure the scripts contained in this repo function as intended by including the necessary elements of sfincs_boozer_scaling. Note some additional adjustments to the code will likely be necessary, as machine-specific changes were made to the stelloptPlusSfincs repository as well.

## How to use

I don't think anyone else will use these, but I'll add some explanation to this section later just in case. Send me an email at macampagna@wm.edu if you have any questions :)
