import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, Conv1D, Flatten
from sklearn.metrics import f1_score
import sys
import pickle


#read the test file
test_csv = sys.argv[1]
test = pd.read_csv(test_csv)

action_list = []
action_dict = {
    'increase' : 0,
    'decrease' : 1,
    'activate' : 2,
    'deactivate' : 3,
    'change language' : 4,
    'bring' : 5
}
for action in test['action']:
    one_hot_action = [0] * len(action_dict.keys())
    one_hot_action[action_dict[action]] = 1
    action_list.append(np.array(one_hot_action))


object_list = []
object_dict = {
    'heat' : 0,
    'lights' : 1,
    'volume' : 2,
    'music' : 3,
    'none' : 4,
    'lamp' : 5,
    'newspaper' : 6,
    'shoes' : 7,
    'socks' : 8,
    'juice' : 9,
    'Chinese' : 10,
    'English' : 11,
    'Korean' : 12,
    'German' : 13
}
for _object in test['object']:
    one_hot_object = [0] * len(object_dict.keys())
    one_hot_object[object_dict[_object]] = 1
    object_list.append(np.array(one_hot_object))


location_list = []
location_dict = {
    'none' : 0,
    'washroom' : 1,
    'kitchen' : 2,
    'bedroom' : 3
}

def get_key(val, my_dict):
   
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"


for location in test['location']:
    one_hot_location = [0] * len(location_dict.keys())
    one_hot_location[location_dict[location]] = 1
    location_list.append(np.array(one_hot_location))

action_list = np.array(action_list)
location_list = np.array(location_list)
object_list = np.array(object_list)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences(list(test['transcription']))
pad_seq = pad_sequences(seq, maxlen=12, padding = 'post')


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


prediction = loaded_model.predict(pad_seq)

actual_list = [1]*len(test)
pred_list = []
for i in range(len(test)):
    action_pred = get_key(np.argmax(prediction[0][i]), action_dict)
    location_pred = get_key(np.argmax(prediction[1][i]), location_dict)
    object_pred = get_key(np.argmax(prediction[2][i]), object_dict)
    
    if action_pred == test['action'].iloc[i] and location_pred == test['location'].iloc[i] and object_pred == test['object'].iloc[i]:
        pred_list.append(1)
    else:
    	pred_list.append(0)
    	print(action_pred + " " + test['action'].iloc[i])
    	print(location_pred + " " + test['location'].iloc[i])
    	print(object_pred + " " + test['object'].iloc[i])

print("F1 Score " + str(f1_score(actual_list, pred_list, average='binary')))
