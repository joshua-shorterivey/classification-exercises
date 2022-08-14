# import section ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

import acquire
# ---

def prep_iris(df_iris):
    """
    """
    df_iris.drop(['species_id', 'measurement_id'], axis=1, inplace=True)
    df_iris.rename({'species_name':'species'}, axis='columns', inplace=True)
    
    dummy_df = pd.get_dummies(df_iris[['species']], dummy_na=False, drop_first=True)
    df_iris = pd.concat([df_iris, dummy_df], axis=1)
    
    return df_iris


def prep_titanic(df_titanic):
    """
    """
    df_titanic.drop(['Unnamed: 0', 'embarked', 'class', 'age', 'deck'], axis=1, inplace=True)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer =  imputer.fit(df_titanic[['embark_town']])
    df_titanic[['embark_town']] = imputer.transform(df_titanic[['embark_town']])

    dummy_titanic = pd.get_dummies(df_titanic[['sex', 'embark_town']], dummy_na=False, drop_first=['True', 'True'])
    df_titanic = pd.concat([df_titanic, dummy_titanic], axis=1)
    
    return df_titanic

#duplicated and made adjustments to bring it in line with function from exploratory analysis lesson

def prep_titanicb(df_titanic):
    """
    """
    df_titanic = df_titanic[(df_titanic.age.notna()) & (df_titanic.embarked.notna())]
    df_titanic.drop(['Unnamed: 0', 'class', 'deck'], axis=1, inplace=True)

    dummy_titanic = pd.get_dummies(df_titanic[['sex', 'embark_town']], dummy_na=False, prefix=['sex', 'embark'])
    
    df_titanic = pd.concat([df_titanic, dummy_titanic], axis=1)
    
    df_titanic.drop(['sex', 'embark_town', 'sex_male'], axis=1, inplace=True)
    
    df_titanic.rename(columns={"sex_female": "is_female"}, inplace=True)
    
    return df_titanic

def prep_telco(df_telco):
    """
    """
    df_telco = df_telco.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    
    df_telco['gender_encoded'] = df_telco.gender.map({'Female':1, 'Male':0})
    df_telco['partner_encoded'] = df_telco.partner.map({'Yes':1, 'No':0})
    df_telco['dependents_encoded'] = df_telco.dependents.map({'Yes':1, 'No':0})
    df_telco['phone_service_encoded'] = df_telco.phone_service.map({'Yes':1, 'No':0})
    df_telco['paperless_billing_encoded'] = df_telco.paperless_billing.map({'Yes':1, 'No':0})
    df_telco['churn_encoded'] = df_telco.churn.map({'Yes':1, 'No':0})
    
    dummy_telco = pd.get_dummies(df_telco[['contract_type',\
                                           'payment_type', \
                                           'internet_service_type', \
                                           'multiple_lines', \
                                           'online_security',\
                                           'online_backup',\
                                           'device_protection',\
                                           'tech_support',\
                                           'streaming_tv',\
                                           'streaming_movies']],\
                                 dummy_na=False, drop_first=True)
    
    df_telco = pd.concat([df_telco, dummy_telco], axis=1)
    df_telco.total_charges = pd.to_numeric(df_telco.total_charges.str.strip())
    
    return df_telco

def split_data(df, target):
    """
    """
    
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    
    #verify shapes of prepared df, train, validate, and test subsets
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

