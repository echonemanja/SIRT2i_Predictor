# SIRT2i_Predictor
SIRT2i_Predictor: A machine learning-based tool to facilitate the discovery of novel SIRT2 inhibitors

Installation instructions for SIRT2i_Predictor

SIRT2i_Predictor was tested on Linux and Windows, but it should be
platform-independent since it was written in Python.

For proper function, all dependences should be installed. 

Installing the dependences:
	Desired way is to so is by using conda environment.
	1. Install Anaconda for your OS (https://docs.anaconda.com/anaconda/installl)
	2. Open Anaconda Powershell Prompt (or just terminal for Linux users)
	3. conda create -n sirt2i_predictor_test python=3.10
	4. conda activate sirt2i_predictor_test
	5. conda install rdkit=2022.03 -c conda-forge
	6. conda install -c rdkit -c mordred-descriptor mordred=1.2
	7. conda install -c conda-forge py-xgboost-cpu
	8. conda install tensorflow=2.8.2
	9. conda install streamlit=1.10.0 -c conda-forge

To run the SIRT2i_Predictor, unizip the SIRT2i_Predictor and navigate to the SIRT2i_Predictor's 
directory using the terminal with active sirt2i_predictor_test environment (step 4 from dependences instructions).
Then run command:
	streamlit run .\SIRT2i_Predictor.py

SIRT2i_Predictor is now previewed in your default web browser.
Results of predictions are written to the ./results folder. 


Good luck!

