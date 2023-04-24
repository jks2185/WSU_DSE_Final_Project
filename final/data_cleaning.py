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
                            'CASEID':'CASE_IDENTIFICATION_NUMBER',
                            'YYYY':'SURVEY_YEAR',
                            'ID':'INTERVIEW_ID',
                            'ICS':'INDEX_OF_CONSUMER_SENTIMENT',
                            'ICC':'INDEX_OF_CURRENT_ECONOMIC_CONDITIONS',
                            'ICE':'INDEX_OF_CONSUMER_EXPECTATIONS',
                            'PAGO':'PERSONAL_FINANCES_B/W_YEAR_AGO',
                            'PAGO5':'PERSONAL_FINANCES_B/W_5_YEAR_AGO',
                            'PEXP':'PERSONAL_FINANCES_B/W_NEXT_YEAR',
                            'PEXP5':'PERSONAL_FINANCES_B/W_IN_5YRS',
                            'BAGO':'ECONOMY_BETTER/WORSE_YEAR_AGO',
                            'BEXP':'ECONOMY_BETTER/WORSE_NEXT_YEAR',
                            'UNEMP':'UNEMPLOYMENT_MORE/LESS_NEXT_YEAR',
                            'GOVT':'GOVERNMENT_ECONOMIC_POLICY',
                            'RATEX':'INTEREST_RATES_UP/DOWN_NEXT_YEAR',
                            'PX1Q1':'PRICES_UP/DOWN_NEXT_YEAR',
                            'DUR':'DURABLES_BUYING_ATTITUDES',
                            'HOM':'HOME_BUYING_ATTITUDES',
                            'SHOM':'G/B_SELL_HOUSE',
                            'CAR':'VEHICLE_BUYING_ATTITUDES',
                            'INCOME':'TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS',
                            'HOMEOWN':'OWN/RENT_HOME',
                            'HOMEVAL':'HOME_VALUE_UP/DOWN',
                            'AGE':'AGE_OF_RESPONDENT',
                            'REGION':'REGION_OF_RESIDENCE',
                            'SEX':'SEX_OF_RESPONDENT',
                            'MARRY':'MARITAL_STATUS_OF_RESPONDENT',
                            'EDUC':'EDUCATION_OF_RESPONDENT',
                            'ECLGRD':'EDUCATION:_COLLEGE_GRADUATE',
                            'POLAFF':'POLITICAL_AFFILIATION'})

    df = filter_data(df,['CASE_IDENTIFICATION_NUMBER','SURVEY_YEAR','INTERVIEW_ID','INDEX_OF_CONSUMER_SENTIMENT','INDEX_OF_CURRENT_ECONOMIC_CONDITIONS','INDEX_OF_CONSUMER_EXPECTATIONS','PERSONAL_FINANCES_B/W_YEAR_AGO',
                        'PERSONAL_FINANCES_B/W_5_YEAR_AGO','PERSONAL_FINANCES_B/W_NEXT_YEAR','PERSONAL_FINANCES_B/W_IN_5YRS','ECONOMY_BETTER/WORSE_YEAR_AGO','ECONOMY_BETTER/WORSE_NEXT_YEAR','UNEMPLOYMENT_MORE/LESS_NEXT_YEAR',
                        'GOVERNMENT_ECONOMIC_POLICY','INTEREST_RATES_UP/DOWN_NEXT_YEAR','PRICES_UP/DOWN_NEXT_YEAR','DURABLES_BUYING_ATTITUDES','HOME_BUYING_ATTITUDES','G/B_SELL_HOUSE','VEHICLE_BUYING_ATTITUDES','TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS',
                        'OWN/RENT_HOME','HOME_VALUE_UP/DOWN','AGE_OF_RESPONDENT','REGION_OF_RESIDENCE','SEX_OF_RESPONDENT','MARITAL_STATUS_OF_RESPONDENT','EDUCATION_OF_RESPONDENT','EDUCATION:_COLLEGE_GRADUATE','POLITICAL_AFFILIATION'])
    df = remove_value(df, '  ', 'AGE_OF_RESPONDENT')
    df = remove_value(df, '      ', 'TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')

    return df
    
    


if __name__ == '__main__':
    main()