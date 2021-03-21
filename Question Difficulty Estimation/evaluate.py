import time
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from LevelClassificationModel import LevelClassificationModel
from socket import *
import pickle

def infer():

    bert_config_file = "checkpoints/bert_config.json"
    vocab_file = "checkpoints/vocab.txt"
    dropout_prob = 0.0
    memory_model_path = "checkpoints/Memory_level_model.bin"
    logical_model_path = "checkpoints/Logical_level_model.bin"
    
    model = LevelClassificationModel(bert_config_file, vocab_file, dropout_prob, memory_model_path, logical_model_path)

    description = "" # model seems to require this but it is empty as dataset does not provide descriptions

    df = pd.read_pickle('validation.pickle') # contains all validation questions and ground truth memory and logic levels
    with open('validation.pickle', 'rb') as f:
        data = pickle.load(f)
    
    print(data.values)
    exit()
    df['memory_predicted'] = -1 # add empty columns for predictions
    df['logic_predicted'] = -1

    start = time.time()
    print("Predicting...")

    # iterate over all questions and make the predictions
    for index, row in df.iterrows():
        question = row['question']
        utterance = row['utterances']
        memory_level, logic_level = model.predict(question, description, utterance)    
        df.at[index, 'memory_predicted'] = memory_level
        df.at[index, 'logic_predicted'] = logic_level

    end = time.time()
    print("Predicted memory levels for all validation questions.")
    print("Total processing time: {}".format(end-start))
    print("")
    print("Writing results to file...")
    df.to_pickle("validation_predictions.pickle")
    print(df.tail())
    
    print(20*"-")
    print("Memory accuracy:", metrics.accuracy_score(df['memory'], df['memory_predicted']))
    print("Logic accuracy:", metrics.accuracy_score(df['logic'], df['logic_predicted']))
    print(20*"-")
    print("Ground truth memory level mean:", np.mean(df['memory']))
    print("Ground truth memory level variance:", np.var(df['memory']))
    print("Predicted memory level mean:", np.mean(df['memory_predicted']))
    print("Predicted memory level variance:", np.var(df['memory_predicted']))
    print(20*"-")
    print("Ground truth logic level mean:", np.mean(df['logic']))
    print("Ground truth logic level variance:", np.var(df['logic']))
    print("Predicted logic level mean:", np.mean(df['logic_predicted']))
    print("Predicted logic level variance:", np.var(df['logic_predicted']))

if __name__=="__main__":
    infer()
