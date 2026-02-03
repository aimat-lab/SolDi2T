# Sol(Di)²T: A Differentiable Digital Twin Framework for Solar Cell Energy Yield Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

<p align="center">
  <img src="overview.png" alt="Project Overview" width="500"/>
  <br>
  <em>Overview of the digital twin framework Sol(Di)²T.</em>
</p>

Sol(Di)²T (a **So**lar **Di**fferentiable **Di**gital **T**win) is a differentiable digital twin framework for solar cells that predicts the total energy yield of a given device architecture and material choice. It supports data-driven decision-making and accelerates development and optimization, making it relevant for industrial R&D, scale-up, and application-specific PV design.

Please follow the following instructions for installation.

## Create and activate the conda environment
``conda create -n SolDi2T python=3.13 -y``

``conda activate SolDi2T``

## Clone the repository
``git clone https://github.com/aimat-lab/SolDi2T.git``

``cd SolDi2T``

## Install the project
``pip install .``

## Training the machine learning models
To retrain the various machine learning models employed in this study, please consult the data available in the same [Zenodo link](https://zenodo.org/records/17790703?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU4MmRkNTViLTdhZDYtNGE0My05YzhiLWY3Mzg2ZDFkYjFkNiIsImRhdGEiOnt9LCJyYW5kb20iOiIzZDYyYTQxNTU5YmU1Zjg1YTAwNGE3YjM0OGRhZWExZCJ9.SyGwDC8WcHHE4iwM4PgrhcQrlSp1qeNC3IyBtO2uBR8kN-Ep-cDIyq1buRE6DlTmXpKiAEPobgDUQvelQ3jOHQ).

## Citation

If you use this code or data in your research, please cite the following paper:

> **Towards a fully differentiable digital twin for solar cells** > Marie Louise Schubert, Houssam Metni, et al. (2025).  
> *arXiv preprint arXiv:2512.02904*.  
> Available at: [https://arxiv.org/abs/2512.02904](https://arxiv.org/abs/2512.02904)

**BibTeX:**

```bibtex
@misc{schubert2025fullydifferentiabledigitaltwin,
      title={Towards a fully differentiable digital twin for solar cells}, 
      author={Marie Louise Schubert and Houssam Metni and Jan David Fischbach and Benedikt Zerulla and Marjan Krstić and Ulrich W. Paetzold and Seyedamir Orooji and Olivier J. J. Ronsin and Yasin Ameslon and Jens Harting and Thomas Kirchartz and Sandheep Ravishankar and Chris Dreessen and Eunchi Kim and Christian Sprau and Mohamed Hussein and Alexander Colsmann and Karen Forberich and Klaus Jäger and Pascal Friederich and Carsten Rockstuhl},
      year={2025},
      eprint={2512.02904},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={[https://arxiv.org/abs/2512.02904](https://arxiv.org/abs/2512.02904)}, 
}