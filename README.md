
# SIRT2i_Predictor: A machine learning-based tool to facilitate the discovery of novel SIRT2 inhibitors

This repository contains code and accompanyed ML models described in this work:<br/> 
> Djokovic, N.; Rahnasto-Rilla, M.; Lougiakis, N.; Lahtela-Kakkonen, M.; Nikolic, K. SIRT2i_Predictor: A Machine Learning-Based Tool to Facilitate the Discovery of Novel SIRT2 Inhibitors. Pharmaceuticals 2023, 16 (1), 127. https://doi.org/10.3390/ph16010127.

Application is deployed on the Streamlit cloud for demo: [SIRT2i_Predictor](https://echonemanja-sirt2i-predictor-sl-cloud-sirt2i-predictor-88n0qw.streamlitapp.com/)

The code is written in Python. 
Current installation instructions for SIRT2i_Predictor are tested on Linux/UNIX and Windows. 

## Installation instructions for SIRT2i_Predictor

Desired way to run the application is within conda environment with installed dependences (requires prior installation of [Anaconda](https://www.anaconda.com/), or [Miniconda](https://conda.io/miniconda.html)).<br />
Installing the dependences: <br />
	 1. `conda create -n sirt2i_predictor python=3.9` <br />
	 2. `conda activate sirt2i_predictor` <br />
	 3. `conda install -c conda-forge rdkit` <br />
	 4. `conda install -c conda-forge scikit-learn` <br />
	 5. `python -m pip install tensorflow-cpu` or `conda install -c conda-forge tensorflow-cpu` (Using `tensorflow` instead `tensorflow-cpu` works as well, but sometimes requires more memory); <br />
	 6. `conda install -c conda-forge mordred` <br />
	 7. `conda install -c conda-forge xgboost` <br />
	 8. `python -m pip install streamlit`

To run the SIRT2i_Predictor, unzip the SIRT2i_Predictor and navigate to the SIRT2i_Predictor's 
directory using the terminal with active sirt2i_predictor environment (step 2 from instructions).
Then run:
	`streamlit run ./SIRT2i_Predictor.py`

SIRT2i_Predictor is then previewed in your default web browser.

Results of predictions are written to the ./results folder. 

### Video tutorial

https://user-images.githubusercontent.com/109313212/187770205-1e1d844a-0e62-4261-929d-bca7c091d1c8.mp4


Good luck!

