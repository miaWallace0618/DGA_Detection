from flask import Flask, jsonify, request, make_response
from flask_compress import Compress
import pandas as pd
import os
import numpy as np
from numpy import loadtxt
import pickle
import keras
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

app = Flask(__name__)
Compress(app)

# clean raw dga files
def clean_file(filename):
    """
    Take .txt file as input, output cleaned file

    """
    # read common TLD information from TLDlist.txt(downloaded from ICANN)
    # skip comments, use '/n' as delimiter, convert into lowercase
    tld = loadtxt("TLDlist.txt", dtype=str, comments="#", delimiter="/n", unpack=False)
    tld = [x.lower() for x in tld]

    # read the dga-dataset using 'ctrl-A' as delimiter 
    # add the original column
    dga = pd.read_csv(filename.stream, sep = chr(1), header = None)
    dga.columns = ['domain']
    # convert all values into lowercase
    dga = dga.apply(lambda x: x.astype(str).str.lower())
    # add a column named 'domain_list' -  in order to seperate it into detailed domain information later
    dga['domain_list'] = dga['domain'].apply(lambda x: x.split('.'))
    
    df_list = []
    # analyse domain_list
    # first, deal with instances with only two elements(pri+tld) in 'domain_list'
    dga2 = dga[dga['domain_list'].apply(lambda x: True if len(x)==2 else False)]
    # split them into two columns as primary1 and tld1
    # add two empty columns:primary2 and tld2
    dga2['primary1'] = dga['domain_list'].apply(lambda x: x[0])
    dga2['primary2'] = ''
    dga2['tld1'] = dga['domain_list'].apply(lambda x: x[1])
    dga2['tld2'] = ''
    df_list.append(dga2)

    
    # then, deal with instances with three elements in 'domain_list'
    dga3 = dga[dga['domain_list'].apply(lambda x: True if len(x)==3 else False)]
    if dga3.shape[0]>0:
        # for instances with three domain elements(pri1+tld1+tld2, both tld1 and tld2 belong to common TLD)
        # split them into three columns as primary1, tld1, tld2
        # add one empty column primary2
        dga3_1 = dga3[dga3['domain_list'].apply(lambda x: True if x[-2] in tld and x[-1] in tld else False)]
        dga3_1['primary1'] = dga3['domain_list'].apply(lambda x: x[0])
        dga3_1['primary2'] = ''
        dga3_1['tld1'] = dga3['domain_list'].apply(lambda x: x[1])
        dga3_1['tld2'] = dga3['domain_list'].apply(lambda x: x[2])
        # for instances with three domain elements(sub+pri+tld1, tld1 belongs to common TLD)
        dga3_2 = dga3[dga3['domain_list'].apply(lambda x: True if x[-2] not in tld and x[-1] in tld else False)]
        # split them into two columns as primary1, tld1
        # add two emtpy columns primary2, tld2
        dga3_2['primary1'] = dga3_2['domain_list'].apply(lambda x: x[1])
        dga3_2['primary2'] = ''
        dga3_2['tld1'] = dga3_2['domain_list'].apply(lambda x: x[2])
        dga3_2['tld2'] = ''
        # for instances with three domain elements(sub+pri+tld1, tld1 doesn't belong to common TLD)
        dga3_3 = dga3[dga3['domain_list'].apply(lambda x: True if x[-2] not in tld and x[-1] not in tld else False)]
        # split them into two columns as primary1, tld1
        # add two emtpy columns primary2, tld2
        dga3_3['primary1'] = dga3_3['domain_list'].apply(lambda x: x[1])
        dga3_3['primary2'] = ''
        dga3_3['tld1'] = dga3_3['domain_list'].apply(lambda x: x[2])
        dga3_3['tld2'] = ''
        df_list.extend([dga3_1, dga3_2, dga3_3])
    
    # next, deal with instances with four element in 'domain_list'
    dga4 = dga[dga['domain_list'].apply(lambda x: True if len(x)== 4 else False)]
    if dga4.shape[0]>0:
        # for instances with four domain elements(sub+pri1+tld1+tld2, both tld1 and tld2 belong to common TLD)
        # split them into three columns as primary1, tld1, tld2
        # add one empty column primary2
        dga4_1 = dga4[dga4['domain_list'].apply(lambda x: True if x[-2] in tld and x[-1] in tld else False)]
        dga4_1['primary1'] = dga4_1['domain_list'].apply(lambda x: x[1])
        dga4_1['primary2'] = ''
        dga4_1['tld1'] = dga4_1['domain_list'].apply(lambda x: x[2])
        dga4_1['tld2'] = dga4_1['domain_list'].apply(lambda x: x[3])
        # for instances with four domain elements(sub+pri2+pri1+tld1, tld1 belongs to common TLD)
        # split them into three columns as primary1, primary2, tld1
        # add one empty column tld2
        dga4_2 = dga4[dga4['domain_list'].apply(lambda x: True if x[-2] not in tld and x[-1] in tld else False)]
        dga4_2['primary1'] = dga4_2['domain_list'].apply(lambda x: x[2])
        dga4_2['primary2'] = dga4_2['domain_list'].apply(lambda x: x[1])
        dga4_2['tld1'] = dga4_2['domain_list'].apply(lambda x: x[3])
        dga4_2['tld2'] = ''
        # for instances with four domain elements(sub+pri2+pri1+tld1, tld1 doesn't belong to common TLD)
        # split them into three columns as primary1, primary2, tld1
        # add one empty column tld2
        dga4_3 = dga4[dga4['domain_list'].apply(lambda x: True if x[-2] not in tld and x[-1] not in tld else False)]
        dga4_3['primary1'] = dga4_3['domain_list'].apply(lambda x: x[2])
        dga4_3['primary2'] = dga4_3['domain_list'].apply(lambda x: x[1])
        dga4_3['tld1'] = dga4_3['domain_list'].apply(lambda x: x[3])
        dga4_3['tld2'] = ''
        df_list.extend([dga4_1, dga4_2, dga4_3])
    
    # deal with instances with five elements in ‘domain_list’
    dga5 = dga[dga['domain_list'].apply(lambda x: True if len(x)== 5 else False)]
    if dga5.shape[0]>0:
        # select instances with structure of sub1+pri2+pri1+tld1+tld2
        # split them into four columns primary1, primary2, tld1, tld2
        dga5['primary1'] = dga5['domain_list'].apply(lambda x: x[2])
        dga5['primary2'] = dga5['domain_list'].apply(lambda x: x[1])
        dga5['tld1'] = dga5['domain_list'].apply(lambda x: x[3])
        dga5['tld2'] = dga5['domain_list'].apply(lambda x: x[4])
        df_list.append(dga5)
    
    # organize above datasets into a final dataset as dga_new
    dga_new = pd.concat(df_list, ignore_index=True)

    # select four columns to build DGA_Detection model
    dga_cleaned = dga_new[['primary1','primary2','tld1','tld2']]
    
    return dga_cleaned



# get predicted probabilites of lstm model
def predict_lstm(filename):
    
    """
    Take .txt files as input, output predicted probabilities by lstm model and cleaned input data

    """
    dga_input = clean_file(filename)
    
    # extract 
    dga_base = dga_input['primary1']
    X  = dga_base.values.tolist()
    
    # load the valid chars from traiend lstm model
    file_validchar = 'valid_chars.pkl'
    valid_chars = pickle.load(open(file_validchar, 'rb'))
    max_features = len(valid_chars) + 1

    # convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=56)

    # load the lstm model and get predicted probabilities 
    lstm_model = load_model('lstm_model.h5')
    probs = lstm_model.predict_proba(X)
    
    return probs, dga_input

# get predicted results from ensemble model
def predict_ensemble(filename):
    
    """
    Take .txt files as input, output predicted classes by ensemble model

    """
    
    # get predicited probabilites of lstm model and cleaned file
    probs, dga_input = predict_lstm(filename)
    
    # perform one hot encoding on top 250 tld
    file_labelbinarizer = 'labelbinarizer.pkl'
    lb = pickle.load(open(file_labelbinarizer, 'rb'))
    
    # transform column tld1 of cleaned file
    dga_tld = pd.DataFrame(lb.transform(dga_input['tld1']))
    dga_tld.reset_index(drop=True, inplace=True)
    
    # merge probabilities of lstm and one-hot encoding of tld
    lstm_probs = pd.DataFrame(probs)
    lstm_probs.columns = ['lstm_probs']
    lstm_probs.reset_index(drop=True, inplace=True)
    dga_cleaned = pd.concat([dga_tld, lstm_probs], axis=1)
    X = dga_cleaned.values
    
    # load ensemble model
    file_ensemble = 'lrensemble_model.pkl'
    ensemble_model = pickle.load(open(file_ensemble, 'rb'))
    
    # get predictions
    predictions = ensemble_model.predict(X)
    
    return predictions


@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Accept,Content-Type'
    return response

@app.route('/predict', methods=['POST'])
def predict():
   
    #try:
    f = request.files['file']
    predictions = predict_ensemble(f)

    return jsonify({"prediction": predictions.tolist()})
    
#    except Exception as ex:
#        error_response = {
#            'error_message': "Unexpected error",
#            'stack_trace': str(ex)
#        }
#        return jsonify(error_response), 50

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
