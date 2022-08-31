# SIRT2i_Predictor
## SIRT2i_Predictor: A machine learning-based tool to facilitate the discovery of novel SIRT2 inhibitors

Installation instructions for SIRT2i_Predictor

SIRT2i_Predictor was tested on Linux and Windows, but it should be
platform-independent since it was written in Python.

For proper function, all dependences should be installed. 

Installing the dependences: <br />
	Desired way is to so is by using conda environment. <br />
	 1. Install Anaconda for your OS (https://docs.anaconda.com/anaconda/installl) <br />
	 2. Open Anaconda Powershell Prompt (or just terminal for Linux users) <br />
	 3. conda create -n sirt2i_predictor_test python=3.10 <br />
	 4. conda activate sirt2i_predictor_test <br />
	 5. conda install rdkit=2022.03 -c conda-forge <br />
	 6. conda install -c rdkit -c mordred-descriptor mordred=1.2 <br />
	 7. conda install -c conda-forge py-xgboost-cpu <br />
	 8. conda install tensorflow=2.8.2 <br />
	 9. conda install streamlit=1.10.0 -c conda-forge <br />

To run the SIRT2i_Predictor, unizip the SIRT2i_Predictor and navigate to the SIRT2i_Predictor's 
directory using the terminal with active sirt2i_predictor_test environment (step 4 from dependences instructions).
Then run command:
	streamlit run .\SIRT2i_Predictor.py

SIRT2i_Predictor is now previewed in your default web browser.
Results of predictions are written to the ./results folder. 

### Video tutorial

https://user-images.githubusercontent.com/109313212/187770205-1e1d844a-0e62-4261-929d-bca7c091d1c8.mp4


Good luck!

