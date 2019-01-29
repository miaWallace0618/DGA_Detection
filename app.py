from flask import Flask, jsonify, request, make_response
from flask_compress import Compress
import pandas as pd
import os
import numpy as np
from numpy import loadtxt
import pickle
import gc
import keras
from tensorflow import reset_default_graph
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras import backend as K

app = Flask(__name__)
Compress(app)

# read common TLD information from TLDlist.txt(downloaded from ICANN)
# skip comments, use '/n' as delimiter, convert into lowercase
tld = loadtxt("TLDlist.txt", dtype=str, comments="#", delimiter="/n", unpack=False)
tld = [x.lower() for x in tld]

def preprocessing(domain_list):
    primary1 = ''
    primary2 = ''
    tld1     = ''
    tld2     = ''

    if len(domain_list)==2: # first, deal with instances with only two elements(pri+tld) in 'domain_list'
        # split them into two columns as primary1 and tld1
        # add two empty columns:primary2 and tld2
        primary1 = domain_list[0]
        tld1 = domain_list[1]
    elif len(domain_list)==3: # then, deal with instances with three elements in 'domain_list'
        # for instances with three domain elements(pri1+tld1+tld2, both tld1 and tld2 belong to common TLD)
        # split them into three columns as primary1, tld1, tld2
        # add one empty column primary2
        if domain_list[-2] in tld and domain_list[-1] in tld:
            primary1 = domain_list[0]
            tld1 = domain_list[1]
            tld2 = domain_list[2]
        # for instances with three domain elements(sub+pri+tld1, tld1 belongs to common TLD)
        elif domain_list[-2] not in tld and domain_list[-1] in tld:
            # split them into two columns as primary1, tld1
            # add two emtpy columns primary2, tld2
            primary1 = domain_list[1]
            tld1 = domain_list[2]
        # for instances with three domain elements(sub+pri+tld1, tld1 doesn't belong to common TLD)
        elif domain_list[-2] not in tld and domain_list[-1] not in tld:
            # split them into two columns as primary1, tld1
            # add two emtpy columns primary2, tld2
            primary1 = domain_list[1]
            tld1 = domain_list[2]
    elif len(domain_list)== 4: # next, deal with instances with four element in 'domain_list'
        # for instances with four domain elements(sub+pri1+tld1+tld2, both tld1 and tld2 belong to common TLD)
        # split them into three columns as primary1, tld1, tld2
        # add one empty column primary2
        if domain_list[-2] in tld and domain_list[-1] in tld:
            primary1 = domain_list[1]
            tld1 = domain_list[2]
            tld2 = domain_list[3]
        # for instances with four domain elements(sub+pri2+pri1+tld1, tld1 belongs to common TLD)
        # split them into three columns as primary1, primary2, tld1
        # add one empty column tld2
        elif domain_list[-2] not in tld and domain_list[-1] in tld:
            primary1 = domain_list[2]
            primary2 = domain_list[1]
            tld1 = domain_list[3]
        # for instances with four domain elements(sub+pri2+pri1+tld1, tld1 doesn't belong to common TLD)
        # split them into three columns as primary1, primary2, tld1
        # add one empty column tld2
        elif domain_list[-2] not in tld and domain_list[-1] not in tld:
            primary1 = domain_list[2]
            primary2 = domain_list[1]
            tld1 = domain_list[3]
    # deal with instances with five elements in ‘domain_list’
    elif len(domain_list)== 5:
        # select instances with structure of sub1+pri2+pri1+tld1+tld2
        # split them into four columns primary1, primary2, tld1, tld2
        primary1 = domain_list[2]
        primary2 = domain_list[1]
        tld1 = domain_list[3]
        tld2 = domain_list[4]
    
    return pd.Series([primary1, primary2, tld1, tld2])

def clean_file(filename):
    """
    Take .txt file as input, output cleaned file
    """

    # read the dga-dataset using 'ctrl-A' as delimiter 
    # add the original column
    dga = pd.read_csv(filename.stream, sep = chr(1), header = None)
    dga.columns = ['domain']
    # convert all values into lowercase
    dga = dga.apply(lambda x: x.astype(str).str.lower())
    # add a column named 'domain_list' -  in order to seperate it into detailed domain information later
    dga['domain_list'] = dga['domain'].apply(lambda x: x.split('.'))

    dga['primary1'] = ''
    dga['primary2'] = ''
    dga['tld1'] = ''
    dga['tld2'] = ''
    
    # analyse domain_list
    dga[['primary1', 'primary2', 'tld1', 'tld2']] = dga['domain_list'].apply(preprocessing)


    # select four columns to build DGA_Detection model
    return dga[['primary1','primary2','tld1','tld2']]



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
    K.clear_session()
    
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
    reset_default_graph()
    
    try:
        f = request.files['file']
        predictions = predict_ensemble(f)

        return jsonify({"prediction": predictions.tolist()})
    
    except Exception as ex:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(ex)
        }
        return jsonify(error_response), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
