import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def get_data() -> pd.DataFrame:
    path = r'https://storage.googleapis.com/eg3311/'
    file_name = 'Customer Sentiment Data.csv'
    data = os.path.join(path, file_name)
    return pd.read_csv(data)

def rename_columns(df: pd.DataFrame, lst:dict) -> pd.DataFrame:
    return df.rename(columns=lst)

def filter_data(df:pd.DataFrame, lst:list) -> pd.DataFrame:
    return df.filter(lst)

def as_type(df:pd.DataFrame, type_string:str, column:str) -> pd.DataFrame:
    return df.astype({column:type_string})

def remove_value(df:pd.DataFrame, value:str, column:str) -> pd.DataFrame:
    return df[df[column] != value]

def main():
    pass