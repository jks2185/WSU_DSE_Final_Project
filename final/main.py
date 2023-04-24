import dash
import dash_bootstrap_components as dbc
from dash import html as html
from dash import Input, Output
from dash import dcc as dcc
import pandas as pd
import plotly.express as px
import data_cleaning
import plotly.graph_objects as go
import numpy as np
import ml_model
import pickle
from sklearn.metrics import mean_squared_error

# Load df dataset
df = data_cleaning.main()
df = data_cleaning.as_type(df, 'int', 'TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
df = data_cleaning.as_type(df, 'int', 'AGE_OF_RESPONDENT')
#create first figure
df_group = df.groupby('AGE_OF_RESPONDENT')['TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS'].mean().reset_index()
df_group = pd.DataFrame(df_group)
df_group = df_group[['AGE_OF_RESPONDENT', 'TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS']].sort_values(by='AGE_OF_RESPONDENT',ascending=True)
#new graph df
df['SEX_OF_RESPONDENT'] = df['SEX_OF_RESPONDENT'].astype(str)
#replace 1 with M and 2 with F
df['SEX_OF_RESPONDENT'] = df['SEX_OF_RESPONDENT'].replace({'1': 'M', '2': 'F'})
df.dropna(subset=['SEX_OF_RESPONDENT'], inplace=True)
pivot = df.pivot_table(index=['SURVEY_YEAR'], columns='SEX_OF_RESPONDENT', values='TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS', aggfunc='mean')
#reset the index
pivot = pivot.reset_index()
#drop rows where SEX is blank or not M or F
df = df[df['SEX_OF_RESPONDENT'].isin(['M', 'F'])]
#convert INCOME column to numeric type
#create second figure
df['PERSONAL_FINANCES_B/W_5_YEAR_AGO'] = df['PERSONAL_FINANCES_B/W_5_YEAR_AGO'].astype(str)
#replace the financial condition number into str
df['PERSONAL_FINANCES_B/W_5_YEAR_AGO'] = df['PERSONAL_FINANCES_B/W_5_YEAR_AGO'].replace({'1':'Better now','3' :'Same', '5': 'Worse now','8': 'DK','9':'NA'})
df_2022 = df.query("SURVEY_YEAR == 2022")
df_2022_PAGO5 = df_2022['PERSONAL_FINANCES_B/W_5_YEAR_AGO'].value_counts().reset_index().rename(columns={'index': 'sub_cat_values', 'PERSONAL_FINANCES_B/W_5_YEAR_AGO': 'counts'})
df_2017 = df.query("SURVEY_YEAR == 2017")
df_2017_PAGO = df_2017['PERSONAL_FINANCES_B/W_YEAR_AGO'].value_counts().reset_index().rename(columns={'index': 'sub_cat_values', 'PERSONAL_FINANCES_B/W_YEAR_AGO': 'counts'})
#create next graph
df_NewColumn = df[['PERSONAL_FINANCES_B/W_YEAR_AGO','VEHICLE_BUYING_ATTITUDES', 'ECONOMY_BETTER/WORSE_YEAR_AGO', 'UNEMPLOYMENT_MORE/LESS_NEXT_YEAR','ECONOMY_BETTER/WORSE_NEXT_YEAR','DURABLES_BUYING_ATTITUDES']]
corr = df_NewColumn.corr()
corr_columns = ['PRICES_UP/DOWN_NEXT_YEAR', 'DURABLES_BUYING_ATTITUDES', 'HOME_BUYING_ATTITUDES', 'VEHICLE_BUYING_ATTITUDES']
corr_df = df[corr_columns]
corr_matrix = corr_df.corr()
annotations = np.round(corr_matrix.values, 2)
counts = df['EDUCATION:_COLLEGE_GRADUATE'].value_counts()
# Get the sizes and labels for the pie chart
sizes = [counts[0], counts[1]]
labels = ['Non-College Graduates', 'College Graduates']
df_data = df.groupby('SURVEY_YEAR')[['INDEX_OF_CONSUMER_SENTIMENT', 'INDEX_OF_CURRENT_ECONOMIC_CONDITIONS', 'INDEX_OF_CONSUMER_EXPECTATIONS']].mean().reset_index()
df_data = pd.DataFrame(df_data)
df_data  = df_data[['SURVEY_YEAR', 'INDEX_OF_CONSUMER_SENTIMENT', 'INDEX_OF_CURRENT_ECONOMIC_CONDITIONS', 'INDEX_OF_CONSUMER_EXPECTATIONS']].sort_values(by='SURVEY_YEAR')

#ml models creation
X_train, X_test, y_train, y_test = ml_model.get_split(df)
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

#Graph creation
fig1 = px.line(df_group, x='AGE_OF_RESPONDENT', y='TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
#fig2 = px.bar(df_fin, x='AGE_OF_RESPONDENT', y=['PERSONAL_FINANCES_B/W_NEXT_YEAR', 'PERSONAL_FINANCES_B/W_YEAR_AGO'])
fig2 = px.pie(df_2022_PAGO5, values='counts', names='sub_cat_values',
        labels={'sub_cat_values': 'Financial Condition'},
        #labels={'ou': 'ere'},
        color_discrete_sequence=px.colors.qualitative.Plotly)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(font=dict(size=18),
                showlegend=True)
fig3 = px.pie(df_2017_PAGO, values='counts', names='sub_cat_values',
        labels={'sub_cat_values': 'Financial Condition'},
        color_discrete_sequence=px.colors.qualitative.Plotly)
fig3.update_traces(textposition='inside', textinfo='percent+label')
fig3.update_layout(font=dict(size=18),
                showlegend=True)
fig4 = px.imshow(corr, x=corr.columns, y=corr.columns,
        color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect='auto')
fig4.update_layout(
                title={
                    'text': 'Correlation Heatmap',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
#create double bar graph using plotly express
fig5 = px.bar(pivot, x='SURVEY_YEAR', y=['M', 'F'], barmode='group')
#set plot title and labels
fig5.update_layout(title='Year on Year Average Income by Sex and Category', xaxis_title='Year, Sex', yaxis_title='Average Income')
#figure 6
fig6 = px.imshow(corr_matrix.values,
        x=corr_columns, y=corr_columns,
        color_continuous_scale='blues',
        zmin=-1, zmax=1,
        labels=dict(x='', y=''))
# add annotations to the heatmap
for i in range(len(corr_columns)):
    for j in range(len(corr_columns)):
        fig6.add_annotation(x=corr_columns[i], y=corr_columns[j],
                        text=str(annotations[i][j]),
                        showarrow=False, font=dict(color='white'))
# update layout to show the color scale and adjust the margins
fig6.update_layout(title="Correlation Matrix",
                coloraxis_colorbar=dict(title='Correlation'),
                margin=dict(l=100, r=100, t=50, b=100))
#figure 7 pie chart
fig7 = px.pie(values=sizes, names=labels, title='Proportion of College Graduates in Respondents')
fig7.update_traces(textposition='inside', textinfo='percent+label')
#willis graphs
fig8 = px.line(df_data, x='SURVEY_YEAR', y=['INDEX_OF_CURRENT_ECONOMIC_CONDITIONS', 'INDEX_OF_CONSUMER_SENTIMENT'])
fig9 = px.histogram(df, x='AGE_OF_RESPONDENT', nbins=20)
fig10 = px.bar(df, x='MARITAL_STATUS_OF_RESPONDENT', y='TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
fig11 = px.box(df, x='EDUCATION_OF_RESPONDENT', y='INDEX_OF_CONSUMER_SENTIMENT')
fig12 = px.line(df_data, x='SURVEY_YEAR', y='INDEX_OF_CONSUMER_EXPECTATIONS')
fig13 = px.histogram(df, x='INDEX_OF_CONSUMER_EXPECTATIONS', nbins=20)
fig14 = px.box(df, x='REGION_OF_RESIDENCE', y='INDEX_OF_CURRENT_ECONOMIC_CONDITIONS')
fig15 = px.scatter_matrix(df_data, dimensions=['INDEX_OF_CONSUMER_EXPECTATIONS', 'INDEX_OF_CONSUMER_SENTIMENT'])


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
    Input('input-1', 'value')
)
def update_output(value):
    return f'Prediction: {value}'
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
                html.Div([
                    dcc.Graph(id='fig4', figure=fig4)
                ]),
                html.H5('Correlation between Inflation and Buying Attitudes'),
                html.Div([
                    dcc.Graph(id='fig6', figure=fig6)
                ]),
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
                html.Div(
                    [
                        dcc.Graph(id='fig8', figure=fig8)
                    ]
                ),
                html.H5('Index of Consumer Sentiment'),
                html.Div(
                    [
                        dcc.Graph(id='fig13', figure=fig13)
                    ]
                ),
                html.H5('Consumer Sentiment by Education'),
                html.Div(
                    [
                        dcc.Graph(id='fig11', figure=fig11)
                    ]
                ),
                html.H5('Consumer Sentiment Over Time'),
                html.Div(
                    [
                        dcc.Graph(id='fig12', figure=fig12)
                    ]
                ),
                html.H5('Current Economic Conditions by Region'),
                html.Div(
                    [
                        dcc.Graph(id='fig14', figure=fig14)
                    ]
                ),
                html.H5('Current Economic Conditions by Region'),
                html.Div(
                    [
                        dcc.Graph(id='fig15', figure=fig15)
                    ]
                ),
            ])
        return layout

    elif tab == 'graph_demographics':
        layout = html.Div([
                html.H5('Proportion of College Grads in Respondents'),
                html.Div([
                    dcc.Graph(id='fig7', figure=fig7)
                ]),
                html.H5('Age of Respondent Distribution'),
                html.Div([
                    dcc.Graph(id='fig9', figure=fig9)
                ]),
                html.H5('Education of Respondent v Consumer Sentiment'),
                html.Div([
                    dcc.Graph(id='fig11', figure=fig11)
                ])
                ])
        return layout

    elif tab == 'ml':
        layout = html.Div([
                dcc.Input(id='input-1', type='text', value=''),
                html.Br(),
                html.Div(id='output'),
                
                html.H2('Mean Square Error of Models:'),
                html.P(f'Linear MSE: {linear_mse}'),
                html.P(f'RF MSE: {rf_mse}'),
                html.P(f'RF MSE: {xgb_mse}')
                ])
        return layout

if __name__ == '__main__':
    app.run_server(port=8888)