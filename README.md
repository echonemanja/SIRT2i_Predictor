
# SIRT2i_Predictor: A machine learning-based tool to facilitate the discovery of novel SIRT2 inhibitors

This repository contains code and accompanyed ML models described in this work:<br/> 
> Djokovic, N., Rahnasto-Rilla, M., Lougiakis, N., Lahtela-Kakkonen, M. & Nikolic, K. SIRT2i_Predictor: A machine learning-based tool to facilitate the discovery of novel SIRT2 inhibitors. (2022) [doi:10.26434/chemrxiv-2022-8q749](https://doi.org/10.26434/chemrxiv-2022-8q749).

Application is deployed on the Streamlit cloud for demo: [SIRT2i_Predictor](https://echonemanja-sirt2i-predictor-sl-cloud-sirt2i-predictor-88n0qw.streamlitapp.com/)

The code is written in Python. 
Current installation instructions for SIRT2i_Predictor are tested on Linux/UNIX and Windows. 

## Installation instructions for SIRT2i_Predictor

Desired way to run the application is within conda environments with installed dependences (requires prior installation of [Anaconda](https://www.anaconda.com/), or [Miniconda](https://conda.io/miniconda.html)).<br />
Installing the dependences: <br />
	 1. `conda create -n sirt2i_predictor_test python=3.10` <br />
	 2. `conda activate sirt2i_predictor_test` <br />
	 3. `conda install rdkit=2022.03 -c conda-forge` <br />
	 4. `conda install -c rdkit -c mordred-descriptor mordred=1.2` <br />
	 5. `conda install -c conda-forge py-xgboost-cpu` <br />
	 6. `conda install tensorflow=2.8.2` <br />
	 7. `conda install streamlit=1.10.0 -c conda-forge` <br />

To run the SIRT2i_Predictor, unzip the SIRT2i_Predictor and navigate to the SIRT2i_Predictor's 
directory using the terminal with active sirt2i_predictor_test environment (step 2 from instructions).
Then run:
	`streamlit run ./SIRT2i_Predictor.py`

SIRT2i_Predictor is then previewed in your default web browser.

Results of predictions are written to the ./results folder. 

### Video tutorial

https://user-images.githubusercontent.com/109313212/187770205-1e1d844a-0e62-4261-929d-bca7c091d1c8.mp4


Good luck!

