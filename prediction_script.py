import io
import sys

import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
import numpy as np

category = sys.argv[1]
num_of_class = sys.argv[2]
data_string = sys.argv[3]

data_array = data_string.split(";")
data_array = np.array(data_array)
data_array = data_array.reshape(1, num_of_class)

df = pd.DataFrame(data_array)

encoded_data = None
for i in df:
    label_encoder = LabelEncoder()
    file_name = 'encoder/' + category + '/' + 'classes' + str(i + 1) + '.npy'
    label_encoder.classes_ = np.load(file_name, allow_pickle=True)
    feature = label_encoder.transform(df[i])
    
    feature_df = pd.DataFrame(feature)
    if encoded_data is None:
        encoded_data = feature_df
    else:
        encoded_data = pd.concat((encoded_data, feature_df), axis=1)

model = XGBClassifier()
model.load_model(category + ".json")

prediction_result = model.predict(encoded_data)
print(prediction_result)
