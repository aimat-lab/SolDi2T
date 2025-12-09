# Sol(Di)2T: A Differentiable Digital Twin Framework for Solar Cell Energy Yield Optimization

Please follow the following instructions for installation

## Create and activate the conda environment
``conda create -n SolDi2T python=3.13 -y``

``conda activate SolDi2T``

## Clone the repository
``git clone https://github.com/aimat-lab/SolDi2T.git``

``cd SolDi2T``

## Install the project
``pip install .``

This [Zenodo link](https://zenodo.org/records/17790703?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU4MmRkNTViLTdhZDYtNGE0My05YzhiLWY3Mzg2ZDFkYjFkNiIsImRhdGEiOnt9LCJyYW5kb20iOiIzZDYyYTQxNTU5YmU1Zjg1YTAwNGE3YjM0OGRhZWExZCJ9.SyGwDC8WcHHE4iwM4PgrhcQrlSp1qeNC3IyBtO2uBR8kN-Ep-cDIyq1buRE6DlTmXpKiAEPobgDUQvelQ3jOHQ) contains the irradiance files necessary to reproduce the code in various locations (Honolulu, Seattle, Miami and Phoenix). To include these files, just extract the necessary ``.pkl`` files from the ``.zip`` file on the Zenodo link and put them in a defined ``data/irradiance/`` folder.

## Training the machine learning models
To retrain the various machine learning models employed in this study, please consult the data available in the same following [Zenodo link](https://zenodo.org/records/17790703?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU4MmRkNTViLTdhZDYtNGE0My05YzhiLWY3Mzg2ZDFkYjFkNiIsImRhdGEiOnt9LCJyYW5kb20iOiIzZDYyYTQxNTU5YmU1Zjg1YTAwNGE3YjM0OGRhZWExZCJ9.SyGwDC8WcHHE4iwM4PgrhcQrlSp1qeNC3IyBtO2uBR8kN-Ep-cDIyq1buRE6DlTmXpKiAEPobgDUQvelQ3jOHQ).

