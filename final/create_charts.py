from dash import html as html
from dash import dcc as dcc
import pandas as pd
import plotly.express as px
import data_cleaning
import numpy as np

def main(df):
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
    df11 = df
    df11['EDUCATION_OF_RESPONDENT'] = df11['EDUCATION_OF_RESPONDENT'].astype(str)
    df11 = df11.sort_values('EDUCATION_OF_RESPONDENT',ascending=False)
    df11['EDUCATION_OF_RESPONDENT'] = df11['EDUCATION_OF_RESPONDENT'].replace({'1': 'Grade 0-8 no hs diploma', '2': 'Grade 9-12 no hs diploma', '3': 'Grade 0-12 w/ hs diploma',  '4': 'Grade 13-17 no col degree',  '5': 'Grade 13-16 w/ col degree',  '6': 'Grade 17 w/ col degree'})
    df11 = df11.dropna()
    #create graph 14
    df14 = df
    df14=df14.dropna()
    df14['REGION_OF_RESIDENCE'] = df14['REGION_OF_RESIDENCE'].astype(str)
    df14['REGION_OF_RESIDENCE']= df14['REGION_OF_RESIDENCE'].replace({'1': 'West', '2': 'North Central', '3':'Northest','4':'South'})
    df_buying_attitudes = df.groupby(['SEX_OF_RESPONDENT', 'SURVEY_YEAR'])[['HOME_BUYING_ATTITUDES', 'VEHICLE_BUYING_ATTITUDES']].mean().reset_index()
    df_buying_attitudes = pd.DataFrame(df_buying_attitudes)
    df_buying_attitudes = df_buying_attitudes.sort_values('SURVEY_YEAR', ascending=True)
    df_buying_marry = df.groupby(['MARITAL_STATUS_OF_RESPONDENT', 'SURVEY_YEAR'])[['HOME_BUYING_ATTITUDES', 'VEHICLE_BUYING_ATTITUDES']].mean().reset_index()
    df_buying_marry = pd.DataFrame(df_buying_marry)
    df_buying_marry['MARITAL_STATUS_OF_RESPONDENT'] = df_buying_marry['MARITAL_STATUS_OF_RESPONDENT'].dropna().astype(str).replace({'1':'Married/Partner', '2':'Separated', '3':'Divorced', '4':'Widowed', '5':'Never married'})
    df_buying_marry = df_buying_marry.sort_values('SURVEY_YEAR', ascending=True)



    #Graph creation
    fig1 = px.line(df_group, x='AGE_OF_RESPONDENT', y='TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
    fig1.update_xaxes(title='Age of Respondent')
    fig1.update_yaxes(title='Total Household Income')
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
    fig5.update_xaxes(title='Survey Year')
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
    fig8.update_xaxes(title='Survey Year')
    fig8.update_yaxes(title='Indexes of Sentiment')
    fig9 = px.histogram(df, x='AGE_OF_RESPONDENT', nbins=20)
    fig9.update_xaxes(title='Age of Respondent')
    fig10 = px.bar(df, x='MARITAL_STATUS_OF_RESPONDENT', y='TOTAL_HOUSEHOLD_INCOME_-_CURRENT_DOLLARS')
    fig10.update_xaxes(title='Marital Status')
    fig10.update_yaxes(title='Total Household Income')
    fig11 = px.box(df11, x='EDUCATION_OF_RESPONDENT', y='INDEX_OF_CURRENT_ECONOMIC_CONDITIONS')
    fig11.update_xaxes(title='Education of Respondent')
    fig11.update_yaxes(title='Index of Current Economic Conditions')
    fig12 = px.line(df_data, x='SURVEY_YEAR', y='INDEX_OF_CONSUMER_EXPECTATIONS')
    fig12.update_xaxes(title='Survey Year')
    fig12.update_yaxes(title='Index of Consumer Expectations')
    fig13 = px.histogram(df, x='INDEX_OF_CONSUMER_EXPECTATIONS', nbins=20)
    fig13.update_xaxes(title='Index of Consumer Expectations')
    fig14 = px.box(df, x='REGION_OF_RESIDENCE', y='INDEX_OF_CURRENT_ECONOMIC_CONDITIONS')
    fig14.update_xaxes(title='Region of Residence')
    fig14.update_yaxes(title='Index of Current Economic Conditions')
    fig15 = px.scatter_matrix(df_data, dimensions=['INDEX_OF_CONSUMER_EXPECTATIONS', 'INDEX_OF_CONSUMER_SENTIMENT'])
    fig17 = px.line(df_buying_attitudes, x='SURVEY_YEAR', y='HOME_BUYING_ATTITUDES', color='SEX_OF_RESPONDENT')
    fig18 = px.line(df_buying_attitudes, x='SURVEY_YEAR', y='VEHICLE_BUYING_ATTITUDES', color='SEX_OF_RESPONDENT')
    fig19 = px.line(df_buying_marry, x='SURVEY_YEAR', y='HOME_BUYING_ATTITUDES', color='MARITAL_STATUS_OF_RESPONDENT')
    fig20 = px.line(df_buying_marry, x='SURVEY_YEAR', y='VEHICLE_BUYING_ATTITUDES', color='MARITAL_STATUS_OF_RESPONDENT')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig17, fig18, fig19, fig20

if __name__ == '__main__':
    pass