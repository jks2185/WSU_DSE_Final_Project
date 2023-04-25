import dash
import dash_bootstrap_components as dbc
from dash import html as html
from dash import Input, Output
from dash import dcc as dcc
import pandas as pd
import plotly.express as px
import numpy as np
from ml_model import get_split
import pickle
from sklearn.metrics import mean_squared_error
import create_charts
import data_cleaning

# Load df dataset
df = data_cleaning.main()
df = data_cleaning.as_type(df, 'int', 'TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
df = data_cleaning.as_type(df, 'int', 'AGE_OF_RESPONDENT')

fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig17, fig18, fig19, fig20 = create_charts.main(df)

#ml models creation
X_train, X_test, y_train, y_test = get_split(df)
with open('linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)
# rf_model = pickle.load(open('rf_model.pkl', 'rb'))
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
# xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
linear_pred = linear_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
ml_data = pd.DataFrame({'y_test' : y_test, 'prediction' : xgb_pred })

#Create ml models
fig16 = px.scatter(ml_data, x='y_test', y='prediction', trendline='ols')
fig16.update_traces(marker=dict(color='green'))
fig16.update_xaxes(title='y test')
fig16.update_yaxes(title='Prediction')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define dropdown options
dropdown_options = {
    'AGE_OF_RESPONDENT': 'Age',
    'REGION_OF_RESIDENCE': 'Region of Residence',
    'SEX_OF_RESPONDENT': 'Sex',
    'MARITAL_STATUS_OF_RESPONDENT': 'Marital Status'
}

# Define groupby options
groupby_options = {
    'EDUCATION_OF_RESPONDENT': 'Education level',
    'EDUCATION:_COLLEGE_GRADUATE': 'Education: college grad',
    'POLITICAL_AFFILIATION': 'Political Affiliation'
}

@app.callback(
    Output('output', 'children'),
    [Input(component_id='submit-button', component_property='n_clicks'),
    Input('input-1', 'value')]
)
def update_output(n_clicks, value):

    if int(n_clicks) > 0:
        value1, value2 = value.split(',')
        value1 = int(value1)
        value2 = int(value2)
        features = np.array([[value1, value2]])
        result = xgb_model.predict(features)
        return f'Prediction: {result}'
#n_clicks = 0
#input_value = ''
# Define function to create data table
def create_table(df):
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H5('df Dashboard', className='text-center'),
            html.Hr(),
            html.H4('Select X-Axis'),
            dcc.Dropdown(
                id='dropdown-x',
                options=[{'label': label, 'value': value} for value, label in dropdown_options.items()],
                value='AGE_OF_RESPONDENT'
            ),
            html.H4('Select Group By'),
            dcc.Dropdown(
                id='dropdown-groupby',
                options=[{'label': label, 'value': value} for value, label in groupby_options.items()],
                value='EDUCATION_OF_RESPONDENT'
            ),
        ], md=4),
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label='Table', value='table'),
                dcc.Tab(label='Income', value='graph_money'),
                dcc.Tab(label='Demographics', value='graph_demographics'),
                dcc.Tab(label='Sentiment', value='graph_sentiment'),
                dcc.Tab(label='ML Prediction', value='ml')
            ], id='tabs', value='table'),
            html.Div(id='display-page'),
        ], md=8)
    ])
])

# Define callback to update display page
@app.callback(
    Output('display-page', 'children'),
    [
     Input('dropdown-x', 'value'),
     Input('dropdown-groupby', 'value'),
     Input('tabs', 'value')
     ]
)

def display_page(x, groupby, tab):
    # Compute groupby
    if tab == 'table':
        # Create data table
        df_grouped = df.groupby(groupby)[x].describe().reset_index()
        table = create_table(df_grouped)
        # Return table
        return table
    elif tab == 'graph_money':
        layout = html.Div([
                html.H5('Personal Income by Age'),
                html.P('WIth this graph we wanted to look at income by age, expecting older people would make more.'),
                html.Div([
                    dcc.Graph(id='fig1',figure=fig1)
                ]),
                html.P("We concluded that people who were older did in fact make more, with a drop-off around retirement age."),
                html.H5('Personal Income by Sex'),
                html.P('We expected men to make more on average, as this is usually the case.'),
                html.Div([
                    dcc.Graph(id='fig5',figure=fig5)
                ]),
                html.P('This is what we found across all years.'),
                html.H5('Correlation between Inflation, home amount, vehile, unemployment, and business conditions'),
                html.P('I would expect these to have positive correlations with each other, as unemployment, buying attitudes, and personal finance should reflect on how the economy is doing.'),
                html.Div([
                    dcc.Graph(id='fig4', figure=fig4)
                ]),
                html.P('This is more or less what we see, as personal finances being better or worse is affected by unemployment, and these all generally have a negative affect on how we view the economy.'),
                html.H5('Correlation between Inflation and Buying Attitudes'),
                html.P('We would think that buying attidues would be negatively related to inflation, as people would be less likely to buy things as price goes up.'),
                html.Div([
                    dcc.Graph(id='fig6', figure=fig6)
                ]),
                html.P('This is what we see here, as they all have a negative correlation with each other.'),
                html.H5('Personal Finace in 2017 v 2022'),
                html.P('We thought there would be a drop-off in how people thought they were doing from 2017 to 2022.'),
                html.Div(children=[
                    dcc.Graph(id='fig2',figure=fig2, style={'width': '49%', 'display': 'inline-block'}),
                    dcc.Graph(id='fig3',figure=fig3, style={'width': '49%', 'display': 'inline-block'})
                ]),
                html.P('There is a slight drop-off, but not as much as expected.')
             ])
        return layout
    elif tab == 'graph_sentiment':
        layout = html.Div([
                html.H5('Index of Consumer Sentiment V Current Economic Conditions'),
                html.P('Here we would think sentiment would be in line with the economic conditions.'),
                html.Div(
                    [
                        dcc.Graph(id='fig8', figure=fig8)
                    ]
                ),
                html.H5('Index of Consumer Sentiment'),
                html.P('We were looking to see if there was anything that stood out in the frequency of good or bad responses.'),
                html.Div(
                    [
                        dcc.Graph(id='fig13', figure=fig13)
                    ]
                ),
                html.P('The Bar graph shows a slightly higher amount of responses of "20" and "120" compared to others. \
                       This could indicate a higher number of respondents answering due to successful economic expansion or in times of recession/depression.'),
                html.H5('Consumer Sentiment by Education'),
                html.P('We were interested to see if there was any relationships between education level and sentiment of respondents.'),
                html.Div(
                    [
                        dcc.Graph(id='fig11', figure=fig11)
                    ]
                ),
                html.P('From this plot, we can see that there is a positive relationship between education level and consumer sentiment. \
                       Respondents with higher education levels tend to have higher consumer sentiment scores, \
                       as indicated by the higher median values for each level of education. \
                       The range of consumer sentiment scores is also wider for respondents with higher education levels, indicating greater variability in the responses.'),
                html.H5('Consumer Sentiment Over Time'),
                html.P('This line graph helps to show how consumer sentiment tracked over the years since the survey began.'),
                html.Div(
                    [
                        dcc.Graph(id='fig12', figure=fig12)
                    ]
                ),
                html.P('The graph illustrates the higher amount of responses of "20" and "120" compared to others from the previous bar chart may be associated with specific events in the economy. \
                       All of the drops in sentiment coincide with recessions over the past 40 years (1980, early 90s, 2001, 2008, 2020).'),
                html.H5('Current Economic Conditions by Region'),
                html.P('We were interested to see if there was any relationships between Consumer expectations and sentiment across regions.'),
                html.Div(
                    [
                        dcc.Graph(id='fig14', figure=fig14)
                    ]
                ),
                html.P('The matrix above shows that regions 1 and 4 are more tightly correlated while 2 and 3 had more variance in their answers. This may indicate greater economic wealth gaps in those regions or different socioeconomic situations.'),
                html.H5('Consumer Expectations V Sentiment'),
                html.P('We were curious to see if there was any variance in the economic conditions of those surveyed.'),
                html.Div(
                    [
                        dcc.Graph(id='fig15', figure=fig15)
                    ]
                ),
                html.P('The histogram shows a roughly normal distribution indicating there is not a significant difference across regions.'),
            ])
        return layout

    elif tab == 'graph_demographics':
        layout = html.Div([
                html.H5('Proportion of College Grads in Respondents'),
                html.P('We would think this would lean towards more non-college graduates.'),
                html.Div([
                    dcc.Graph(id='fig7', figure=fig7)
                ]),
                html.P('This is what we see here, as there are more respondents without degrees than with.'),
                html.H5('Age of Respondent Distribution'),
                html.P('We would think this would be somewhat evenely distributed across all ages.'),
                html.Div([
                    dcc.Graph(id='fig9', figure=fig9)
                ]),
                html.P('This leans more towards then working ages between 20s and 60 year old adults.'),
                html.H5('Education of Respondent v Current Economic Conditions'),
                html.P('We would think that those with a higher education would have a higher view on the economy with more money.'),
                html.Div([
                    dcc.Graph(id='fig11', figure=fig11)
                ]),
                html.P('We see that this is the case, with those with a lower education having a lower view.'),
                html.H5('Home Buying Attitudes by Sex'),
                html.P('Here we would think that men would have a higher view on buying a house.'),
                html.Div([
                    dcc.Graph(id='fig17', figure=fig17)
                ]),
                html.P('This is actually not the case, with women having a higher view on it throughout the years.'),
                html.H5('Car Buying Attitudes by Sex'),
                html.P('Here we would think that men would have a higher view on buying a car.'),
                html.Div([
                    dcc.Graph(id='fig18', figure=fig18)
                ]),
                html.P('This is actually not the case, with women having a higher view on it throughout the years.'),
                html.H5('Home Buying Attitudes by Marital Status'),
                html.P('We would think that those who are married would have the highest sentiment towards buying a house.'),
                html.Div([
                    dcc.Graph(id='fig19', figure=fig19)
                ]),
                html.P('This is not the case, as we see that those who are separated do.'),
                html.H5('Vehicle Buying Attitudes by Marital Status'),
                html.P('Here would think that those who are married would have a higher need for a car and thus have a higher buying attitude.'),
                html.Div([
                    dcc.Graph(id='fig20', figure=fig20)
                ]),
                html.P('Although it fluctuates through the years, separated has the highest some years, followd by widowed, and never married.'),
                ])
                
        return layout

    elif tab == 'ml':
        layout = html.Div([
                dcc.Input(id='input-1', type='text', value=''),
                html.Button('Submit', id='submit-button', n_clicks=0),
                html.Br(),
                html.P('Output:'),
                html.Div(id='output'),
                
                html.H2('Mean Square Error of Models:'),
                html.P(f'Linear MSE: {linear_mse}'),
                html.P(f'RF MSE: {rf_mse}'),
                html.P(f'XGB MSE: {xgb_mse}'),
                html.P('We can see here that the XgBoost model has the lowest mean squared error, so that is the model we have used for this prediction.'),
                html.H5('Residuals'),
                html.Div(
                    [
                        dcc.Graph(id='fig16', figure=fig16)
                    ]
                ),
                html.P('Here we can see that there isnt much of a pattern for the model to follow as far as making predictions accurately. In the future, I would use more dimensions for this model since only having a couple did not create \
                       a viable model.'),
                ])
        return layout

if __name__ == '__main__':
    app.run_server(port=8888)