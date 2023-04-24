import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import data_cleaning
import pickle

def get_data() -> pd.DataFrame:
    return data_cleaning.main()

def get_split(df):
    X = df[['SURVEY_YEAR', 'PERSONAL_FINANCES_B/W_YEAR_AGO']]
    y = df['INDEX_OF_CONSUMER_SENTIMENT']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(df : pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = get_split(df)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Fit a random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    

    # Fit an XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
        
    return linear_model, rf_model, xgb_model

def main():
   df = get_data()
   print(df.head())
   print(type(df))
   X_train, X_test, y_train, y_test  = get_split(df)
   linear_model, rf_model, xgb_model = train_model(df)
   linear_pred = linear_model.predict(X_test)
   print(linear_pred)

   with open("linear_model.pkl", "wb") as file:
        pickle.dump(linear_model, file)

   with open("rf_model.pkl", "wb") as file:
        pickle.dump(rf_model, file)

   with open("xgb_model.pkl", "wb") as file:
        pickle.dump(xgb_model, file)


if __name__ == '__main__':
    main()