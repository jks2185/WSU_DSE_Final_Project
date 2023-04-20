import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def get_data() -> pd.DataFrame:
    path = r'C:\Users\e707088\Downloads'
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

def replace_value(df:pd.DataFrame, value, replace_value, column:str) -> pd.DataFrame:
    return df[column].replace({value:replace_value})

def main():
    df = get_data()
    df = rename_columns(df, {
            'CASEID':'CASE IDENTIFICATION NUMBER',
            'YYYY':'SURVEY YEAR',
            'ID':'INTERVIEW ID',
            'ICS':'INDEX OF CONSUMER SENTIMENT',
            'ICC':'INDEX OF CURRENT ECONOMIC CONDITIONS',
            'ICE':'INDEX OF CONSUMER EXPECTATIONS',
            'PAGO':'PERSONAL FINANCES B/W YEAR AGO',
            'PAGO5':'PERSONAL FINANCES B/W 5 YEAR AGO',
            'PEXP':'PERSONAL FINANCES B/W NEXT YEAR',
            'PEXP5':'PERSONAL FINANCES B/W IN 5YRS',
            'BAGO':'ECONOMY BETTER/WORSE YEAR AGO',
            'BEXP':'ECONOMY BETTER/WORSE NEXT YEAR',
            'UNEMP':'UNEMPLOYMENT MORE/LESS NEXT YEAR',
            'GOVT':'GOVERNMENT ECONOMIC POLICY',
            'RATEX':'INTEREST RATES UP/DOWN NEXT YEAR',
            'PX1Q1':'PRICES UP/DOWN NEXT YEAR',
            'DUR':'DURABLES BUYING ATTITUDES',
            'HOM':'HOME BUYING ATTITUDES',
            'SHOM':'G/B SELL HOUSE',
            'CAR':'VEHICLE BUYING ATTITUDES',
            'INCOME':'TOTAL HOUSEHOLD INCOME - CURRENT DOLLARS',
            'HOMEOWN':'OWN/RENT HOME',
            'HOMEVAL':'HOME VALUE UP/DOWN',
            'AGE':'AGE OF RESPONDENT',
            'REGION':'REGION OF RESIDENCE',
            'SEX':'SEX OF RESPONDENT',
            'MARRY':'MARITAL STATUS OF RESPONDENT',
            'EDUC':'EDUCATION OF RESPONDENT',
            'ECLGRD':'EDUCATION: COLLEGE GRADUATE',
            'POLAFF':'POLITICAL AFFILIATION'})
    df = filter_data(df, ['CASE IDENTIFICATION NUMBER','SURVEY YEAR','INTERVIEW ID','INDEX OF CONSUMER SENTIMENT','INDEX OF CURRENT ECONOMIC CONDITIONS','INDEX OF CONSUMER EXPECTATIONS','PERSONAL FINANCES B/W YEAR AGO',
                      'PERSONAL FINANCES B/W 5 YEAR AGO','PERSONAL FINANCES B/W NEXT YEAR','PERSONAL FINANCES B/W IN 5YRS','ECONOMY BETTER/WORSE YEAR AGO','ECONOMY BETTER/WORSE NEXT YEAR','UNEMPLOYMENT MORE/LESS NEXT YEAR',
                      'GOVERNMENT ECONOMIC POLICY','INTEREST RATES UP/DOWN NEXT YEAR','PRICES UP/DOWN NEXT YEAR','DURABLES BUYING ATTITUDES','HOME BUYING ATTITUDES','G/B SELL HOUSE','VEHICLE BUYING ATTITUDES','TOTAL HOUSEHOLD INCOME - CURRENT DOLLARS',
                      'OWN/RENT HOME','HOME VALUE UP/DOWN','AGE OF RESPONDENT','REGION OF RESIDENCE','SEX OF RESPONDENT','MARITAL STATUS OF RESPONDENT','EDUCATION OF RESPONDENT','EDUCATION: COLLEGE GRADUATE','POLITICAL AFFILIATION'])
    df = remove_value(df, '  ', 'AGE OF RESPONDENT')
    df = remove_value(df, '      ', 'TOTAL HOUSEHOLD INCOME - CURRENT DOLLARS')
    
    


if '__name__' == '__main__':
    main()