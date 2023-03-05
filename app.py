import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


infile = Path(__file__).parent / 'encoder'
x_encoder = pickle.load(open(infile, 'rb'))

infile = Path(__file__).parent / 'poly'
polynomial = pickle.load(open(infile, 'rb'))

infile = Path(__file__).parent / 'british_voter'
model = pickle.load(open(infile, 'rb'))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Build a British Voter'
server = app.server


colors = {"Brexit Party/Reform UK":"purple",\
            "Conservative":"blue",\
            "Green Party":"green",\
            "Labour":"red",\
            "Liberal Democrat":"orange",\
            "Scottish National Party (SNP)":"yellow"}


app.layout = \
dbc.Container\
([
    html.Br(),
    dbc.Row(dbc.Card(dbc.CardBody(html.H5('Build a British Voter! Enter demographic characteristics to see the probability of party support at the 2019 election')), style={'width':'100%'})),
    html.Br(),
    dbc.Row(dbc.Stack([
        dbc.Card(dbc.CardBody('A')),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'White British': 'White British,', 'Any other white background': 'other White,', 
                                            'Asian': 'Asian,', 'Black':'Black,', 'Mixed Race': 'Mixed Race,', 'Other ethnic group':'other,'},
                                           id='ethnicity', style={'width': '100%'}, value='White British', clearable=False)),
                 style={'width': '15%'}),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'Anglican': 'Anglican,', 'non-Anglican Protestant': 'non-Anglican Protestant,',\
                                    'Catholic': 'Catholic,', 'Orthodox Christian':'Orthodox Christian,', 'Islam':'Muslim,',\
                                     'Hinduism':'Hindu,', 'Judaism':'Jewish,', 'Buddhism':'Buddhist,', 'Sikhism':'Sikh,',\
                                    'No religion':'non-religious,', 'Other':'other,'},
                                           id='religion', style={'width':'100%'}, value='Anglican', clearable=False)),
                 style={'width':'20%'}), 
        dbc.Card(dbc.CardBody(dcc.Dropdown({'Straight': 'straight', 'LGB+': 'LGB+'},
                                           id='sexuality', style={'width':'100%'}, value='Straight', clearable=False)),
                 style={'width':'12%'}), 
        dbc.Card(dbc.CardBody(dcc.Dropdown({'Male': 'man', 'Female': 'woman'},
                                           id='gender', style={'width':'100%'}, value='Male', clearable=False)),
                 style={'width':'12%'}), 
        dbc.Card(dbc.CardBody(dcc.Dropdown({'18-29':'between 18 and 29.', '30-44':'between 30 and 44.', '45-65':'between 45 and 64.', '65+': '65 or older.'},
                                           id='age_cat', style={'width':'100%'}, value='18-29', clearable=False)),
                style={'width':'20%'}),
        dbc.Card(dbc.CardBody('They are'))
    ], direction='horizontal', gap=2)),
    html.Br(),
    dbc.Row(dbc.Stack([
        dbc.Card(dbc.CardBody(dcc.Dropdown({'Married': 'married,', 'Unmarried': 'unmarried,'},
                                          id='married', style={'width':'100%'}, value='Married', clearable=False)),
                 style={'width':'12%'}),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'Yes': 'have children,', 'No': 'have no children,'},
                                          id='children', style={'width':'100%'}, value='Yes', clearable=False)),
                style={'width':'15%'}),
        dbc.Card(dbc.CardBody('and acheived')),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'No qualifications': 'no formal qualifications',\
                                                  'GCSE or equivalent': 'GCSE or equivalent qualifications',\
                                                  'A-level or equivalent': 'A-level or equivalent qualifications',\
                                                  'Technical or professional qualification': 'a technical or professional qualification',\
                                                  'University degree':'a university degree',\
                                                  'Post-graduate degree': 'a post-graduate degree'},
                                          id='education', style={'width':'100%'}, value='No qualifications', clearable=False)),
                style={'width':'25%'}), 
        dbc.Card(dbc.CardBody('They are')),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'full time': ' working full time', 'part time':'working part time',\
                                        'student':'a full time student', 'retired':'retired', 'unemployed':'unemployed', 'Other':'other'},
                                          id='work', style={'width':'100%'}, value='full time', clearable=False)), 
                style={'width':'15%'}), 
        dbc.Card(dbc.CardBody('and earn'))
    ], direction='horizontal', gap=2)),
    html.Br(),
    dbc.Row(dbc.Stack([
        dbc.Card(dbc.CardBody(dcc.Dropdown({'under £5,000 per year':'<£5k.', '£5,000 to £9,999 per year':'£5-10k.',\
                                          '£10,000 to £14,999 per year': '£10-15k.',\
                                          '£15,000 to £19,999 per year': '£15-20k.',\
                                          '£20,000 to £24,999 per year': '£20-25k.',\
                                          '£25,000 to £29,999 per year': '£25-30k.',\
                                          '£30,000 to £34,999 per year': '£30-35k.',\
                                          '£35,000 to £39,999 per year': '£35-40k.',\
                                          '£40,000 to £44,999 per year': '£40-45k.',\
                                          '£45,000 to £49,999 per year': '£45-50k.',\
                                          '£50,000 to £59,999 per year': '£50-60k.',\
                                          '£60,000 to £69,999 per year': '£60-70k.',\
                                          '£70,000 to £99,999 per year': '£70-100k.',\
                                          '£100,000 to £149,999 per year': '£100-150k.',\
                                          '£150,000 and over': '>£150k,'},
                                          id='income', style={'width':'100%'}, value='under £5,000 per year', clearable=False)),
                style={'width':'11%'}),
        dbc.Card(dbc.CardBody('They live in a home they')), 
        dbc.Card(dbc.CardBody(dcc.Dropdown({'own outright': 'own outright', 'own with mortgage': 'own with mortgage',\
                                                           'private rental': 'rent privately', 'social housing': 'rent socially',\
                                                           'other': 'other'},
                                          id='housing', style={'width':'100%'}, value='own outright', clearable=False)),
                style={'width':'15%'}),
        dbc.Card(dbc.CardBody('in')),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'urban': 'an urban area', 'mostly urban': 'a mostly urban area',\
                                         'mostly rural': 'a mostly rural area', 'rural': 'a rural area'},
                                          id='rural_urban', style={'width':'100%'}, value='urban', clearable=False)),
                style={'width':'18%'}), 
        dbc.Card(dbc.CardBody('in')),
        dbc.Card(dbc.CardBody(dcc.Dropdown({'South East':'the South East.', 'London': 'London.', 'East of England': 'the East of England.',\
                                    'South West': 'the South West.', 'West Midlands': 'the West Midlands.', 'East Midlands': 'the East Midlands.',\
                                    'North West': 'the North West.', 'North East': 'the North East.', 'Yorkshire and The Humber': 'Yorkshire and The Humber.',\
                                     'Wales': 'Wales.', 'Scotland': 'Scotland.'},
                                          id='region', style={'width':'100%'}, value='South East', clearable=False)),
                style={'width':'20%'})
    ], direction='horizontal', gap=2)),
    html.Br(), 
    dbc.Row(dbc.Stack([dcc.Graph(id='fig', style={'width':'100%'})], direction='horizontal'))
])


@app.callback(
    Output('fig', 'figure'),
    Input('ethnicity', 'value'),
    Input('religion', 'value'),
    Input('sexuality', 'value'),
    Input('gender', 'value'),
    Input('age_cat', 'value'),
    Input('married', 'value'),
    Input('children', 'value'),
    Input('education', 'value'),
    Input('work', 'value'),
    Input('income', 'value'),
    Input('housing', 'value'),
    Input('rural_urban', 'value'),
    Input('region', 'value')
)

def update_output(ethnicity, religion, sexuality, gender, age_cat, married, children, education, work, income, housing, rural_urban, region):
    voter_dict = {"ethnicity":[ethnicity], "religion":[religion], "gender":[gender], "housing":[housing],
                      "work":[work], "sexuality":[sexuality], "age_cat":[age_cat], "married":[married],
                      "children":[children], "education":[education], "income":[income], "region":[region],\
                      "rural_urban":[rural_urban]}
    x_transform = x_encoder.transform(pd.DataFrame.from_dict(voter_dict))
    x_transform = polynomial.transform(x_transform)
    probs = model.predict_proba(x_transform)[0]
    prob_dict = {"Brexit Party/Reform UK": probs[0],\
                "Conservative": probs[1],\
                "Green Party": probs[2],\
                "Labour": probs[3],\
                "Liberal Democrat": probs[4],\
                "Scottish National Party (SNP)": probs[5]}
    prob_dict = {party:round(prob*100,2) for party, prob in prob_dict.items()}
    names = list(prob_dict.keys())
    values = list(prob_dict.values())
    
    note = '@hbjroberts<br>Data Source: British Election Study Internet Panel (Wave 19)'

    fig = px.bar(x=names, y=values, color=names,
                 color_discrete_sequence = ["purple", "blue", "green", "red", "orange", "yellow"],
                labels={'x':'Party', 'y':'% Chance of Supporting'},
                height=530)
    fig.update_layout(yaxis_range=[0,100], 
                     legend_title="Party")
    fig.add_annotation(text=note,
                showarrow=False,
                text=note,
                font=dict(size=10), 
                xref='x domain',
                x=0.5,
                yref='y domain',
                y=-0.5)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)

    

