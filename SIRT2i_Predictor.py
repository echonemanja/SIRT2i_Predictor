import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import model_selection
from xgboost import XGBRegressor
#import joblib
from math import pi
from io import BytesIO
from PIL import Image
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
#from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem, PandasTools, Descriptors
from rdkit.Chem import DataStructs
from mordred import Calculator, descriptors
from joblib import dump, load
from matplotlib.lines import Line2D
from functools import partial
from rdkit.Chem.Draw import SimilarityMaps
from datetime import datetime
from tensorflow.keras.models import load_model


class ADLeverage(object):
    def __init__(self, training_molecules, test_molecules):
        self.training_molecules = training_molecules
        self.test_molecules = test_molecules
        self.threshold = (3*(1+training_molecules.shape[1])) / training_molecules.shape[0]
    def __hat_matrix(self):
        H = self.test_molecules.dot(np.linalg.inv(self.training_molecules.T.dot(self.training_molecules)).dot(self.test_molecules.T))
        return H
    def calculate(self):
        leverages = np.diagonal(self.__hat_matrix())
        limit = np.repeat(self.threshold, self.test_molecules.shape[0])
        summary = {'Leverage':leverages, 'Limit':limit}
        return summary

def radar_chart(radar_df):
    categories = ['\n          pIC50/10', 'Probability of \n SIRT2 activity',
                  '\n\n\nProbability of       \n SIRT2 selectivity            \nover SIRT1',
                  'Probability of \n SIRT2 selectivity over SIRT3']
    N = len(categories)

    values = radar_df.loc[radar_df['ID'] == choice].iloc[:, 1:].values.flatten().tolist()
    values += values[:1]
    # values
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=8)
    ax.tick_params(axis='both', which='major', pad=11)

    # Draw ylabels
    ax.set_rlabel_position(1)
    plt.yticks([0.1, 0.5, 1], ["0.1", "0.5", "1"], color="grey", size=9)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'p', alpha=0.1)
    # Show the graph
    # st.pyplot(plt, figsize = (1,1))
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)
    st.subheader('Detailed analysis of probabilities')
    st.text('''Each plot represent different classification model explained in the reference.''')

def tanimoto_ad(ligand_smi):
    tanimoto_base = pd.read_csv('./helpers/tanimoto_base.csv')
    mols = [Chem.MolFromSmiles(smi) for smi in tanimoto_base.SMILES]
    # zameni ligands_mol[1] sa kodom za selektovano jedinjenje
    ligands_mol = Chem.MolFromSmiles(ligand_smi)
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ligands_mol, 2, nBits=1024)
    tanimoto_score = []
    for mol in mols:
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        tan_score = DataStructs.TanimotoSimilarity(ref_fp, fp2)
        tanimoto_score.append(tan_score)
    tan_max = max(tanimoto_score)
    n = np.array(tanimoto_score)
    tan_max_index = np.argmax(n)

    ad_X = pd.DataFrame(list(ref_fp)).T
    X_internal = pd.read_csv('./helpers/X_internal.csv').set_index('Unnamed: 0')
    list_importance = pd.read_csv('./helpers/list_importance.csv').set_index('Unnamed: 0')
    list_important_descriptors = [str(x) for x in list(list_importance.iloc[:, 0])]
    X_internal = X_internal[list_important_descriptors]
    ad_X = ad_X[list(list_importance.iloc[:, 0])]
    leverage_knn = ADLeverage(X_internal, ad_X)
    leverages_knn = leverage_knn.calculate()

    leverage = pd.DataFrame(leverages_knn)
    limit = leverage['Limit'].iloc[0]

    f, ax = plt.subplots(figsize=(5, 4))
    plt.ylim(0, 1.1)
    # plt.grid()
    legend_elements = [Line2D([0], [0], linestyle='--', color='black', lw=2, label='cut-off line')]
    ax.legend(handles=legend_elements, loc='center')
    plt.xlabel("Leverage")
    plt.ylabel("Tanimoto similrity index")
    ax.scatter(leverage['Leverage'], tan_max, marker='o', alpha=1, c='b', s=95)
    plt.plot([limit, limit], [0, 1.1], 'k--')

    buf_adplot = BytesIO()
    plt.savefig(buf_adplot, format="png")
    st.image(buf_adplot)
    st.write('**The most similar SIRT2 inhibitor from ChEMBL database is:**', tanimoto_base.ID.iloc[tan_max_index])
    st.write('**Tanimoto score =**', tan_max)
    st.write('**Chemical structure of**', tanimoto_base.ID.iloc[tan_max_index], ':')
    image_ad = Draw.MolToImage(mols[tan_max_index], size=(250, 250), fitImage=True)
    buffered_ad = BytesIO()
    image_ad.save(buffered_ad, format="png")
    st.image(buffered_ad)

def barplots(binary_proba_list, sirt12_proba_list, sirt23_proba_list):
    fig, axis = plt.subplots(1, 3, figsize=(10, 8))
    axis[0].bar('Model 1: SIRT2 inhibitory activity', binary_proba_list[0], label='probability of inactive class',
                color='#ffdd99')
    axis[0].bar('Model 1: SIRT2 inhibitory activity', binary_proba_list[1], bottom=binary_proba_list[0],
                label='probability of active class', color='#52527a')
    # axis[0].bar('binary', n[2], bottom = n[1] + n[0], label = 'proba3')
    axis[1].bar('Model 3: SIRT1/2 selectivity', sirt12_proba_list[0],
                label='probability of SIRT1/2 nonselective prediction', color='#ff9999')
    axis[1].bar('Model 3: SIRT1/2 selectivity', sirt12_proba_list[1], bottom=sirt12_proba_list[0],
                label='probability of SIRT2/SIRT1 selectivity', color='#00ff00')
    axis[1].bar('Model 3: SIRT1/2 selectivity', sirt12_proba_list[2],
                bottom=sirt12_proba_list[1] + sirt12_proba_list[0], label='probability of SIRT1/SIRT2 inactivity',
                color='#c68c53')
    axis[2].bar('Model 4: SIRT2/3 selectivity', sirt23_proba_list[0],
                label='probability of SIRT2/3 nonselective prediction', color='#ff3300')
    axis[2].bar('Model 4: SIRT2/3 selectivity', sirt23_proba_list[1], bottom=sirt23_proba_list[0],
                label='probability of SIRT2/SIRT3 selectivity', color='#000033')
    axis[2].bar('Model 4: SIRT2/3 selectivity', sirt23_proba_list[2],
                bottom=sirt23_proba_list[1] + sirt23_proba_list[0], label='probability of SIRT2/SIRT3 inactivity',
                color='#009999')
    axis[0].set_ylabel('Probability')
    # axis[0].set_title('Predictions')
    axis[0].legend(bbox_to_anchor=(0.5, 0.9, 0.6, 0.1), loc=4)
    axis[1].legend(bbox_to_anchor=(0.3, 0.5, 0.6, 0.1), loc=8)
    axis[2].legend(bbox_to_anchor=(1, 0, 0.6, 0.1), loc=4)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=1.1,
                        hspace=1.9)
    buf_barplot = BytesIO()
    plt.savefig(buf_barplot, format="png")
    st.image(buf_barplot, use_column_width = 'auto')

#functions for similarity maps
def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0]
def getProba_binary(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]
fpfunc = partial(SimilarityMaps.GetMorganFingerprint, nBits=1024, radius=2)

def drawmol(mol):
    fig_map, maxWeight = SimilarityMaps.GetSimilarityMapForModel(mol,fpfunc,
                                                           lambda x: getProba(x, model_qsar.predict),
                                                           colorMap='bwr',
                                                           step=0.001,
                                                           alpha=0.5)
    buffered_ = BytesIO()
    fig_map.savefig(buffered_, format="png", bbox_inches='tight')
    st.image(buffered_, use_column_width=True)
def drawmol_binary(mol):
    fig_map_, maxWeight = SimilarityMaps.GetSimilarityMapForModel(mol,fpfunc,
                                                           lambda x: getProba_binary(x, model_binary.predict_proba),
                                                           colorMap='bwr',
                                                           step=0.001,
                                                           alpha=0.5)
    buffered__ = BytesIO()
    fig_map_.savefig(buffered__, format="png", bbox_inches='tight')
    st.image(buffered__, use_column_width=True)
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
#########################################################
st.set_page_config(page_icon='🔮', page_title = "SIRT2i_Predictor", layout = 'wide')
image = Image.open('./helpers/logo_final.png')
st.image(image, width = 256)
st.title("SIRT2i_Predictor")
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.header("Virtual Screening Module")
    with st.form(key='multi-structure module'):
        uploaded_file = st.file_uploader('Upload CSV file with IDs (first column) and SMILES (second column)', type='csv')
        submit_button_vs = st.form_submit_button(label='Submit')
        st.caption('Results of the virtual screening are written to the file "results_vs_*.csv".')
        if uploaded_file is not None:
            if submit_button_vs:
                try:
                    df = pd.read_csv(uploaded_file)
                    df.dropna(inplace=True)
                    df.reset_index(drop = True, inplace=True)
                    mols1 = []
                    for index, row in df.iterrows():
                        mols1.append(Chem.MolFromSmiles(row[1]))
                    df['Mols'] = mols1
                    nonan_smiles = []
                    nonan_ids = []
                    for index, moll in df['Mols'].iteritems():
                        if moll is not None:
                            nonan_smiles.append(Chem.MolToSmiles(moll))
                            nonan_ids.append(df.iloc[index,0])
                    # sledici korak je kreirati novi df a obrisati prethodni
                    df_smiles_novi = pd.DataFrame(nonan_smiles, columns=['SMILES'])
                    df_smiles_novi['ID'] = nonan_ids
                    # df_smiles_novi
                    #####################################
                    # canonization
                    smiles_canon = []
                    df_smiles_novi['SMILES_canon'] = [Chem.MolToSmiles(Chem.MolFromSmiles(n)) for n in nonan_smiles]
                    # df_smiles_novi
                    # Uncharger
                    un = rdMolStandardize.Uncharger()
                    mols2 = []
                    for smile in df_smiles_novi['SMILES_canon']:
                        mols2.append(Chem.MolFromSmiles(smile))
                    smiles_uni = [Chem.MolToSmiles(un.uncharge(mol)) for mol in mols2]
                    df_smiles_novi['SMILES_uni'] = smiles_uni
                    df_smiles_novi.drop_duplicates(subset=['SMILES_uni'], inplace=True)
                    df_smiles_novi.reset_index(drop = True, inplace = True)
                    df_smiles_novi.drop(labels=['SMILES', 'SMILES_canon'], inplace=True, axis=1)
                    df_smiles_novi.rename(columns={'SMILES_uni': 'SMILES'}, inplace=True)
                    mols = [Chem.MolFromSmiles(smi) for smi in df_smiles_novi.SMILES]
                    locs = list_duplicates_of(mols, None)
                    df_smiles_novi.drop(labels = locs, inplace = True)
                    df_smiles_novi.reset_index(inplace=True)
                    df_smiles_novi.drop(columns=['index'], inplace=True)
                    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]
                    df_FP = pd.DataFrame(np.array(fp))
                    df_FP = pd.concat([df_smiles_novi, df_FP], axis=1)
                    #reset index
                    df_FP.reset_index(drop = True, inplace=True)
                    X = df_FP.iloc[:, 2:]
                    ##################################
                    model_qsar = XGBRegressor()
                    model_qsar.load_model('./models/XGBoost_ecfp4_qsar.json')
                    #model_binary = XGBClassifier()
                    #model_binary.load_model('XGBoost_smote.json')
                    model_binary = load('./models/RF_binary_ecfp4_model.sav')
                    Y_binary = pd.DataFrame(model_binary.predict(X), columns=['Model 1: SIRT2 inhibitory activity'])
                    Y_binary.replace({0: 'No', 1: 'Yes'}, inplace=True)
                    Y_binary_prob = pd.DataFrame(model_binary.predict_proba(X)[:, 1],
                                                 columns=['Probability of Model 1 "Yes" class prediction'])
                    #Y_binary_prob_for_radchart = pd.DataFrame(list(Y_binary_prob.iloc[:, 0]),
                    #                                          columns=['Probability of SIRT2 activity'])
                    #for index, report in Y_binary_prob.iterrows():
                    #    if Y_binary.loc[index][0] == 'No':
                    #        Y_binary_prob.loc[index] = model_binary.predict_proba(X)[index, 0]
                    #Y_qsar = pd.DataFrame(model_qsar.predict(X), columns=['Model 2: pIC50 (SIRT2)'])
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                    solutions = pd.concat(
                        [df_FP['ID'], df_FP['SMILES'], Y_binary,Y_binary_prob['Probability of Model 1 "Yes" class prediction']], axis=1)
                    solutions
                    solutions.to_csv('./results/results_vs_'+ str(dt_string)+'.csv')
                    st.write('Results should be ranked according to the Probability score.')
                    st.write(
                        'For more detailed analysis of potency and selectivity, copy specific SMILES (or list of SMILES) to the "SMILES-Analyzer Module".')
                except Exception as e:
                    print(e)
with col2:
    st.header("SMILES-Analyzer Module")
    with st.form('Form_descriptors_calc'):
        multi_smiles = st.text_area('Accepts SMILES, or list of SMILES. Each SMILES entry should be placed to a new line.')
        submitted3 = st.form_submit_button('Predict')
        if submitted3:
            list_of_inputs = []
            #Ovaj deo koda bi trebalo da hendluje i prazan red u sredini...proveri
            for line in multi_smiles.split('\n'):
                if line != '':
                    list_of_inputs.append(line)
            df_smiles = pd.DataFrame(list_of_inputs, columns=['SMILES'])
            ids = ['ID_' + str(x) for x in range(len(list_of_inputs))]
            df_smiles['ID'] = ids
            # if smiles are incorrect
            mols1 = []
            for smile in df_smiles['SMILES']:
                mols1.append(Chem.MolFromSmiles(smile))
            nonan_smiles = []
            for moll in mols1:
                if moll is not None:
                    nonan_smiles.append(Chem.MolToSmiles(moll))
            # nonan_smiles
            df_smiles_novi = pd.DataFrame(nonan_smiles, columns=['SMILES'])
            ids2 = ['ID_' + str(x) for x in range(len(nonan_smiles))]
            df_smiles_novi['ID'] = ids2
            # df_smiles_novi
            # canonization
            smiles_canon = []
            df_smiles_novi['SMILES_canon'] = [Chem.MolToSmiles(Chem.MolFromSmiles(n)) for n in nonan_smiles]
            # df_smiles_novi
            # Uncharger
            un = rdMolStandardize.Uncharger()
            mols2 = []
            for smile in df_smiles_novi['SMILES_canon']:
                mols2.append(Chem.MolFromSmiles(smile))
            smiles_uni = [Chem.MolToSmiles(un.uncharge(mol)) for mol in mols2]
            df_smiles_novi['SMILES_uni'] = smiles_uni
            df_smiles_novi.drop_duplicates(subset=['SMILES_uni'], inplace=True)
            df_smiles_novi.reset_index(drop=True, inplace=True)
            df_smiles_novi.drop(labels=['SMILES', 'SMILES_canon'], inplace=True, axis=1)
            df_smiles_novi.rename(columns={'SMILES_uni': 'SMILES'}, inplace=True)
            mols = [Chem.MolFromSmiles(smi) for smi in df_smiles_novi.SMILES]
            locs = list_duplicates_of(mols, None)
            df_smiles_novi.drop(labels=locs, inplace=True)
            df_smiles_novi.reset_index(inplace=True)
            df_smiles_novi.drop(columns=['index'], inplace=True)
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]
            #fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
            df_FP = pd.DataFrame(np.array(fp))
            df_FP = pd.concat([df_smiles_novi, df_FP], axis=1)
            X = df_FP.iloc[:, 2:]
            ###################### OLD CODE ##################
            #calc = Calculator(descriptors, ignore_3D=True)
            #mols_sirt12 = [Chem.MolFromSmiles(smi) for smi in df_smiles_novi.SMILES]
            #df_descript_sirt12 = calc.pandas(mols_sirt12)
            #df_numeric = pd.read_csv('descriptors_pool_sirt12_no4.csv')
            #data = pd.read_csv('finalno_odabrani_deskriptori_sirt12.csv')
            #scaler = load('std_scaler.bin')
            #selected_descr = df_descript_sirt12[df_numeric.iloc[:, 4:].columns]
            #df1_normalized = pd.DataFrame(scaler.transform(selected_descr), columns=selected_descr.columns)
            #final_descr_sirt12 = df1_normalized[data.iloc[:, 4:].columns]
            #X_sirt12 = final_descr_sirt12.iloc[:, :]
            #X_sirt12 = X_sirt12.replace(np.nan, 0)
            ##########################Descriptors calc for sirt23 model######################
            calc = Calculator(descriptors, ignore_3D=True)
            mols_sirt23 = [Chem.MolFromSmiles(smi) for smi in df_smiles_novi.SMILES]
            df_descript_sirt23 = calc.pandas(mols_sirt23)
            df_numeric_23 = pd.read_csv('./helpers/descriptors_pool_sirt23_forGUI.csv')
            data_23 = pd.read_csv('./helpers/final_selected_descriptors_sirt23.csv')
            scaler_23 = load('./helpers/std_scaler_sirt23.bin')
            selected_descr_23 = df_descript_sirt23[df_numeric_23.iloc[:, 4:].columns]
            df1_normalized_23 = pd.DataFrame(scaler_23.transform(selected_descr_23), columns=selected_descr_23.columns)
            final_descr_sirt23 = df1_normalized_23[data_23.iloc[:, 4:].columns]
            X_sirt23 = final_descr_sirt23.iloc[:, :]
            X_sirt23 = X_sirt23.replace(np.nan, 0)
            ##########################################################################
            model_qsar = XGBRegressor()
            model_qsar.load_model('./models/XGBoost_ecfp4_qsar.json')
            #model_binary = XGBClassifier()
            #model_binary.load_model('XGBoost_smote.json')
            model_binary = load('./models/RF_binary_ecfp4_model.sav')
            model_sirt12 = load('./models/RF_sirt12_ecfp4_model.sav')
            model_sirt23 = load_model('./models/DNN2_sirt23_descr_small_model.h5')
            Y_binary = pd.DataFrame(model_binary.predict(X), columns=['Model 1: SIRT2 inhibitory activity'])
            Y_binary.replace({0: 'No', 1: 'Yes'}, inplace=True)
            Y_binary_prob = pd.DataFrame(model_binary.predict_proba(X)[:, 1], columns=['Probability of Model 1 "Yes" class prediction'])
            Y_binary_prob_for_radchart = pd.DataFrame(list(Y_binary_prob.iloc[:, 0]),
                                                      columns=['Probability of SIRT2 activity'])
            #for index, report in Y_binary_prob.iterrows():
            #    if Y_binary.loc[index][0] == 'No':
            #        Y_binary_prob.loc[index] = model_binary.predict_proba(X)[index, 0]
            Y_qsar = pd.DataFrame(model_qsar.predict(X), columns=['Model 2: pIC50 (SIRT2)'])
            Y_sirt12 = pd.DataFrame(model_sirt12.predict(X), columns=['Model 3: SIRT1/2 selectivity'])
            Y_sirt12.replace({0: 'SIRT1/2 nonselective', 1: 'Selective for SIRT2 over SIRT1',
                              2: 'Predicted to be inactive on both SIRT1/SIRT2'}, inplace=True)
            
            Y_sirt12_prob = pd.DataFrame(model_sirt12.predict_proba(X)[:, 0], columns=['Probability of Model 3 SIRT1/2 nonselective prediction'])
            Y_sirt12_prob['Probability of Model 3 SIRT2/1 selectivity'] = model_sirt12.predict_proba(X)[:, 1]
            Y_sirt12_prob['Probability of Model 3 SIRT1/2 inactivity'] = model_sirt12.predict_proba(X)[:, 2]
            # add sirt23 predictions
            Y_sirt23 = pd.DataFrame(np.argmax(model_sirt23.predict(X_sirt23), axis=-1),
                                    columns=['Model 4: SIRT2/3 selectivity'])
            Y_sirt23.replace({0: 'SIRT2/3 nonselective', 1: 'Selective for SIRT2 over SIRT3',
                              2: 'Predicted to be inactive on both SIRT2/SIRT3'}, inplace=True)
       
            Y_sirt23_prob = pd.DataFrame(model_sirt23.predict(X_sirt23)[:, 0],
                                         columns=['Probability of Model 4 SIRT2/3 nonselective prediction'])
            Y_sirt23_prob['Probability of Model 4 SIRT2/3 selectivity'] = model_sirt23.predict(X_sirt23)[:, 1]
            Y_sirt23_prob['Probability of SIRT2/3 inactivity'] = model_sirt23.predict(X_sirt23)[:, 2]
            model_sirt23.predict(X_sirt23)
            solutions = pd.concat([df_FP['ID'], df_FP['SMILES'], Y_binary, Y_qsar['Model 2: pIC50 (SIRT2)'], Y_sirt12,
                                   Y_sirt23['Model 4: SIRT2/3 selectivity'],
                                   Y_binary_prob['Probability of Model 1 "Yes" class prediction'],
                                   Y_sirt12_prob, Y_sirt23_prob], axis=1)
            # solutions = pd.concat([solutions, ids], axis = 1)
            solutions.to_csv('./results/results_analyzer.csv')
            radar_df = pd.concat(
                [df_FP['ID'], Y_qsar / 10, Y_binary_prob_for_radchart,
                 Y_sirt12_prob['Probability of Model 3 SIRT2/1 selectivity'],
                 Y_sirt23_prob['Probability of Model 4 SIRT2/3 selectivity']], axis=1)
            radar_df.to_csv('./helpers/radar_df.csv')
    with st.form('Form2'):
        #multi_smiles = st.text_area('multi SMILES separeted each in new line')
        #submitted3 = st.form_submit_button('Predict')
        #list_of_inputs = []
        #for line in multi_smiles.split('\n'):
        #    list_of_inputs.append(line)
        if len(multi_smiles) != 0:

            solutions = pd.read_csv('./results/results_analyzer.csv').set_index('Unnamed: 0')
            st.header('Summary of the results')
            st.caption('Results of the analysis are written to the file "results_analyzer.csv".')
            st.dataframe(solutions)
            st.subheader('Analyse results for selected compound')
            st.caption('Select compound from the dropdown menu:')
            choice = st.selectbox('', solutions)
            submit_button = st.form_submit_button(label='Calculate for selected')
            if submit_button:
         ####################### molecule img######################
                with col3:
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    #st.title(" \n  ")
                    st.text('_______________________________________________')
                    st.write('**Chemical structure of the selected compound:**')
                    fig_smi = solutions.loc[solutions['ID'] == choice].iloc[:, 1]
                    for figsmi in fig_smi:
                        image = Draw.MolToImage(Chem.MolFromSmiles(figsmi), size = (250, 250), fitImage = True)
                        buffered = BytesIO()
                        image.save(buffered, format="png")
                        st.image(buffered)
                    st.text('_______________________________________________')
                st.subheader('Predictions for selected compound')
                solutions.loc[solutions['ID'] == choice]
        ############# radar chart############################################33
                radar_df = pd.read_csv('./helpers/radar_df.csv').set_index('Unnamed: 0')
                radar_chart(radar_df)
                #Plot histogram of probabilities
                binary_proba_list = [1-solutions.loc[solutions['ID'] == choice].iloc[:, 6], solutions.loc[solutions['ID'] == choice].iloc[:, 6]]
                #if solutions.loc[solutions['ID'] == choice].iloc[:, 2].iloc[0] == 'No':
                #    binary_proba_list = [solutions.loc[solutions['ID'] == choice].iloc[:, 6], 1-solutions.loc[solutions['ID'] == choice].iloc[:, 6]]
                sirt12_proba_list = [solutions.loc[solutions['ID'] == choice].iloc[:, 7], solutions.loc[solutions['ID'] == choice].iloc[:, 8], solutions.loc[solutions['ID'] == choice].iloc[:, 9]]
                sirt23_proba_list = [solutions.loc[solutions['ID'] == choice].iloc[:, 10], solutions.loc[solutions['ID'] == choice].iloc[:, 11], solutions.loc[solutions['ID'] == choice].iloc[:, 12]]
                barplots(binary_proba_list, sirt12_proba_list, sirt23_proba_list)

                ligand_smi = solutions.loc[solutions['ID'] == choice].SMILES.iloc[0]
                st.subheader('Applicability domain analysis of pIC50 predictions')
                st.text('''pIC50 predictions from outside cut-off value are 
considered unreliable.''')
                tanimoto_ad(ligand_smi)
                with col3:
                    st.write('**Binary model per-atom importance**')
                    st.write('A red glow indicates region that has a positive influence on the property; a blue glow indicates a negative influence.')
                    for figsmi in fig_smi:
                        model_binary = load('./models/RF_binary_ecfp4_model.sav')
                        drawmol_binary(Chem.MolFromSmiles(figsmi))
                    st.text('_______________________________________________')
                    st.write('**Regression model per-atom importance**')
                    st.write('A red glow indicates region that has a positive influence on the property; a blue glow indicates a negative influence.')
                    for figsmi in fig_smi:
                        model_qsar = XGBRegressor()
                        model_qsar.load_model('./models/XGBoost_ecfp4_qsar.json')
                        drawmol(Chem.MolFromSmiles(figsmi))







