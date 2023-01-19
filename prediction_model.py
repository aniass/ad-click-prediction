import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import load
import warnings
warnings.simplefilter('ignore')


URL = '\Data\Ad_Click_prediciton_test.csv'
MODELSPATH = '\models\adaboost_model.pkl'

label_encoder = LabelEncoder()


def load_model():
    '''Loading pretrained model'''
    with open(MODELSPATH, 'rb') as file:
        model = load(file)
        return model
    

def clean_data(df):
    '''Deleting missing data, feature engineering for date time features'''
    df = df.dropna(subset=['gender','age_level', 'user_group_id', 'user_depth'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.assign(hour = df.DateTime.dt.hour,
                   day_of_week = df.DateTime.apply(lambda x: x.dayofweek),)
    return df


def data_transformation(data):
    '''Filling missing values and convert non numeric values '''
    df = clean_data(data)
    df = df.assign(city_development_index = df.city_development_index.fillna('0'),
                   product_category_2 = df.product_category_2.fillna("0"),
                   gender = df.gender.map({'Male' : 0, 'Female' : 1}),)
    df['product'] = label_encoder.fit_transform(df['product'])
    data = df.drop(['session_id','user_id', 'DateTime'], axis=1) 
    return data


def read_data(path):
    '''Function to read data'''
    data = pd.read_csv(path)
    df = data_transformation(data)
    return df


def get_prediction(test):
    '''Generating predictions from test data'''
    test_X = np.array(test)
    model = load_model()
    predicted = model.predict(test_X)
    test['click_ prediction'] = predicted
    print(test.head())


if __name__ == '__main__':
    data = read_data(URL)
    get_prediction(data)