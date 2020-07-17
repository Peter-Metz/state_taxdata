# bash commands for installing your package

git clone https://github.com/Peter-Metz/state_taxdata
cd state_taxdata
conda install PSLmodels::taxcalc conda-forge::paramtools
pip install -e .