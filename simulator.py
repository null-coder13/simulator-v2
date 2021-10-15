# Author: Thomas Gadacy
# Mentor: Professor Ruby
# Developed during Quip-RS 2021


import os.path

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import base64
import io
from configparser import ConfigParser


def get_config():
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)
    return config

def get_sim_rank(dataframe):
    index = get_index(dataframe)
    return dataframe.iloc[index]['Simulated Rank']
    

def calculate_ranks(dataframe, transform_columns, weights):
    """
    Calculates the ranks for a given program and returns the dataframe
    """
    calc_df = dataframe
    # Transform data if needed
    if transform_columns is not None:
        calc_df = transform(calc_df, transform_columns)
    # Calculate z-score
    calc_df = calc_zscore(calc_df, weights)
    # Calcuate sum of product
    calc_df = sum_product(calc_df)
    # Rank the weighted_zscore
    calc_df = create_sim_rank(calc_df)
    return calc_df


def reorder_dataframe(dataframe):
    reorder_df = dataframe[['Simulated Rank', 'School']].sort_values('Simulated Rank', ascending=True)
    return reorder_df


def calc_zscore(dataframe, weights):
    """
    Calculates the z-scores for each column
    """
    df = dataframe.copy()
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for weight, column in zip(weights, numeric_cols):
        df[column] = ((df[column] - df[column].mean()) / df[column].std(ddof=0)) * weight
    return df


def sum_product(dataframe):
    df = dataframe.copy()
    df['weighted_zscore'] = df.iloc[:, 1:].sum(axis=1)
    return df


def create_sim_rank(dataframe):
    df = dataframe.copy()
    df['Simulated Rank'] = df['weighted_zscore'].rank(ascending=False)
    return df


def config_dropdown():
    """
    Creates a list of dictionarys for a dropdown list
    """
    config = get_config()
    dropdown = []
    for program in list(config['Drop_Down']):
        dropdown.append({"label": str(config['Drop_Down'][program]), "value": str(config['Drop_Down'][program])})
    return dropdown


def create_piechart(dataframe):
    """
    Creates a piechart for weights of each program
    :param dataframe: The current dataframe selected
    :param title: The title of the program
    :return:
    """
    pie = px.pie(
        dataframe,
        names=dataframe.index,
        values=dataframe.values,
        title="Weights",
    )
    pie.update_layout(showlegend=True, title_x=0.485)
    return pie


def create_dropdown(dataframe):
    """
    :param dataframe: the dataframe
    :return: bar chart and dropdown menu
    """
    dropdown = graph_dropdown(dataframe)
    first_value = list(dropdown[0].values())[0]
    children = [
        dcc.Dropdown(
            id={
                'type': 'dynamic-dropdown',
                'index': 0
            },
            options=dropdown,
            value=first_value,
            multi=False,
            className='dbc_light'
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    ]
    return children


def graph_dropdown(dataframe):
    """
    Creates a list of dictionarys for a bar dropdown list
    """
    columns = list(dataframe.columns)
    dropdown = []
    for col in columns:
        dropdown.append({"label": columns[columns.index(col)], "value": columns[columns.index(col)]})
    return dropdown


def table_data(dataframe):
    """
    Converts the data into the proper dataframe for the indicators datatable
    :param dataframe: A created dataframe 
    :return: a dataframe in the proper format for the indicators table
    """
    df = dataframe
    index = get_index(df)
    table_df = []
    for column in df.columns:
        table_df.append(df.loc[index, column])
    data = {'Indicator': list(df.columns), 'Current': table_df, 'New': table_df}
    return pd.DataFrame(data=data)


def get_index(dataframe):
    """
    Returns the index for Quinnipiac
    """
    return dataframe[dataframe['School'].str.contains('Quinnipiac', na=False)].index[0]


def transform_list(dataframe):
    """
    Returns a list of columns that need to be transformed
    If there are no columns None is returned
    """
    data = dataframe.iloc[1].dropna()
    if data.empty:
        return None
    return list(data.index)


def transform(dataframe, columns):
    """
    Preforms logarithmic transformations on given data set
    """
    df = dataframe.copy()
    df.loc[:, columns] = np.log10(dataframe.loc[:, columns] + 1)
    return df


def get_weights(dataframe):
    """
    Returns a series containing the weights of each column
    """
    return dataframe.iloc[0].dropna()


def min_value(dataframe):
    """
    Returns a list of min values for each column
    """
    return dataframe.iloc[2].dropna()


def max_value(dataframe):
    """
    Returns a list of max values for each column
    """
    return dataframe.iloc[3].dropna()


def get_us_news(dataframe):
    return dataframe.iloc[get_index(dataframe)]['Rank']
    

def clean_df(dataframe):
    df = dataframe
    df = df.drop([0, 1, 2, 3], axis=0)
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('Unnamed|Rank')]
    df = df.fillna(0)
    df = df.round(decimals=2)
    return df


def create_dataframe(program, uploaded_data = None):
    """
    Creates the dataframe for the given program
    """
    config = get_config()
    if config.has_section(program):
        d_f = pd.read_excel(config[program]['location'], index_col=None)
    else:
        d_f = uploaded_data
    return d_f


def update_rank_number(df, curr_program, orig_sim_rank, new_sim_rank, us_news_rank):
    """
    This method calculates the new ranks for the Simulated and difference in rank.
    :param df: the calculated dataframe
    :param curr_program: The current program selected
    :return: A Dictionary containing a value for each of the outputs of the update_ranks callback
    """
    us_primary_rank = 122
    us_research_rank = 104
    if curr_program == 'Medical - Research':
        difference = abs(new_sim_rank - orig_sim_rank)
        return [df.to_dict('records'), html.H4('Simulated Rank: {}'.format(new_sim_rank)),
                html.H4("US News Rank: {}".format(us_research_rank)),
                html.H4('Change in Rank: {}'.format(difference))]
    elif curr_program == 'Medical - Primary':
        difference = abs(new_sim_rank - orig_sim_rank)
        return [df.to_dict('records'), html.H4('Simulated Rank: {}'.format(new_sim_rank)),
                html.H4("US News Rank: {}".format(us_primary_rank)),
                html.H4('Change in Rank: {}'.format(difference))]
    else:
        difference = abs(new_sim_rank - orig_sim_rank)
        return [df.to_dict('records'), html.H4('Simulated Rank: {}'.format(new_sim_rank)),
                html.H4("US News Rank: {}".format(us_news_rank)),
                html.H4('Change in Rank: {}'.format(difference))]


def rank_column(dataframe, column):
    """
    :param dataframe: the dataframe
    :param column: The column choosen by the user
    :return: New df ranked by column
    """
    df = dataframe.reindex(columns=["School", column])
    df["Rank"] = df[column].rank(ascending=False)
    df = df.sort_values('Rank', ascending=True)
    df = df.reset_index()
    return df


def create_linechart(dataframe, choice):
    index = get_index(dataframe)
    name = str(dataframe.loc[index, 'School'])
    rank = dataframe.loc[index, 'Rank']
    line_chart = px.line(
        dataframe,
        x="School",
        y='Rank',
        title='Ranking per Indicator',
        height=500,
        hover_data=[choice]
    )
    line_chart.update_traces(mode='markers+lines')
    line_chart.update_layout(
        xaxis={
            'tickmode': 'array',
            'tickvals': [],
            'ticktext': []
        }
    )
    line_chart.add_annotation(
        x=name,
        y=rank,
        text="Quinnipiac University",
        arrowhead=2
    )
    return line_chart


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            upload_name = os.path.splitext(filename)[0]
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            upload_name = os.path.splitext(filename)[0]
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'xlsx' in filename:
            upload_name = os.path.splitext(filename)[0]
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return None
    return df


def validate_value(dataframe, min_values, max_values):
    """
    Checks to see if the entered value is valid based on the min and max values
    """
    table = list(dataframe["New"])
    for min_v, max_v, value in zip(min_values, max_values, table):
        if value == "":
            value = 0
        if value < min_v:
            return True
        if value > max_v:
            return True
    return False


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO],
                # For mobile devices
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.title = "Ranking Simulator"
server = app.server

app.layout = dbc.Container([

    html.Div(
        [
            dbc.Toast(
                [html.P("Invalid number entered into one or more of the indicators.", className="mb-0")],
                id="error-toast",
                header="Invalid Input: ",
                icon='danger',
                is_open=False,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                duration=4000
            )
        ]
    ),
    dbc.Row([
        dbc.Col(html.H1('Ranking Simulator',
                        className='text-center')),
        html.Br(),
        html.Br(),
        html.Br()
    ]),

    dbc.Row([
        dbc.Col(
            html.Img(src=app.get_asset_url('qu.png'))),
        dbc.Col([
            html.P('Select a program:',
                   className='text-center'),
            dcc.Dropdown(id='my-dropdown', multi=False, value='Nursing - Master',
                         options=config_dropdown(), className='dbc_light'),
            html.Br(),
            html.Br()
        ])
    ]),

    dbc.Row([
        dbc.Col(
            html.Div(id='us_rank_output'),
        ),

        dbc.Col(
            html.Div(id='simulated_output')
        ),

        dbc.Col(
            html.Div(id='diff_rank_output'),
        ),

        html.Br(),
        html.Br(),
        html.Br()
    ], className='dbc_light'),

    dbc.Row([
        dbc.Col([
            html.H4(html.B('Instructions:')),
            html.H5("Adjust the indicators by clicking a cell under the 'New' column and typing in a number.")
        ]),

        html.Br()
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='datatable_indicators',
                columns=[
                    {"name": 'Indicator', "id": 'Indicator', "editable": False},
                    {"name": 'Current', "id": 'Current', "editable": False},
                    {"name": 'New', "id": 'New', "editable": True, "type": 'numeric',
                     "on_change": {"action": "coerce", "failure": "reject"}}
                ],
#                data=selected_df.to_dict('records'),
                style_cell={
                    'minWidth': 100, 'maxWidth': 150, 'width': 100
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'border': '1px solid black'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Current'},
                     'textAlign': 'center', 'width': '50%'},
                    {'if': {'column_id': 'New'},
                     'textAlign': 'center', 'width': '50%'},
                    {'if': {'column_id': 'Indicator'},
                     'textAlign': 'left', 'width': '50%'},
                ],
                style_data_conditional=[
                    {
                        "if": {"state": "active"},
                        "border": ".5px solid ",
                        "fontWeight": 1000,
                    },
                    {"if": {"state": "selected"},
                     "fontWeight": 700,
                     },
                ],
                style_header={
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'border': '1px solid black'
                },
                style_table={
                    'height': '500px', 'overflowY': 'auto'
                },
                fixed_rows={
                    'headers': True
                },
            ),
            html.Br()
        ], width=6, className='dbc_light'),

        dbc.Col([
            dash_table.DataTable(
                id='datatable_ranks',
                columns=[
                    {"name": 'Simulated Rank', "id": 'Simulated Rank', 'editable': False},
                    {"name": 'School', "id": 'School', 'editable': False, 'type': 'text'}
                ],
#                data=rank_df.to_dict('records'),
                style_data={
                    'whiteSpace': 'normal',
                    'border': '1px solid black'
                },
                style_cell={
                    'minWidth': 100, 'maxWidth': 150, 'width': 100
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Simulated Rank'},
                     'textAlign': 'center', 'width': '17%'},
                    {'if': {'column_id': 'School'},
                     'textAlign': 'left', 'width': '50%'},
                ],
                style_data_conditional=
                [
                    {
                        "if": {"state": "active"},
                        "border": ".5px solid ",
                        "fontWeight": 1000,
                    },
                    {"if": {"state": "selected"},
                     "fontWeight": 700,
                     },
                    {
                        'if': {
                            'filter_query': '{School} contains "Quinnipiac"'
                        },
                        'backgroundColor': '#ffff00'
                    }
                ],

                style_header={
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'border': '1px solid black'
                },
                page_action='none',
                fixed_rows={
                    'headers': True
                },
                style_table={
                    'height': '600px', 'overflowY': 'auto'
                }
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
        ], width=6, className='dbc_light')
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id={
                    'type': 'dynamic-graph',
                    'index': 0
                }
            ),
            html.Br(),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                id='bar_dropdown',
                style={
                    'width': '100%',
                    'height': '100%'
                }
            )
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='pie_chart'),
            html.Div("These are the weights for each indicator and how much they affect the overall rank.",
                     className='text-center'),
            html.Br(),
            html.Br(),
            html.Br()
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '1100px',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
            )
        ], className='text-center'),
        html.Br(),
        html.Br(),
        html.Br(),
    ]),
    ## May need to use storage_type= local or session
    dcc.Store(id='memory-output'),
    dcc.Store(id='current-program'),
    dcc.Store(id='uploaded_data')
])


@app.callback([Output('memory-output', 'data'),
               Output('current-program', 'data')],
               Input('my-dropdown', 'value'),
               State('uploaded_data', 'data')
)
def store_dataset(val_chosen, data):
    """
    Updates the data stored on the browser
    """
    if data != None:
        print('There is no data in the uploaded_data')
        uploaded_df = pd.read_json(data, orient='split')
    else:
        uploaded_df = None
    df = create_dataframe(val_chosen, uploaded_df)
    data = [{'name':val_chosen}]
    return df.to_json(date_format='iso', orient='split'), data


@app.callback(
        [Output('datatable_indicators', 'data'),
         Output('pie_chart','figure'),
         Output('bar_dropdown', 'children')],
        Input('memory-output','data')
)
def update_charts(stored_data):
    """
    Updates the indicator table and graphs when a different program is chosen
    """
    df = pd.read_json(stored_data, orient='split')
    
    weights = get_weights(df)
    pie_chart = create_piechart(weights)

    df = clean_df(df)

    df_copy = df.drop(['School'], axis=1)
    dff = table_data(df)
    children = create_dropdown(df_copy)
    dff = dff.drop(0)
    return dff.to_dict('records'), pie_chart, children


@app.callback(
        [Output('datatable_ranks', 'data'),
         Output('simulated_output', 'children'),
         Output('us_rank_output', 'children'),
         Output('diff_rank_output', 'children')],
        Input('datatable_indicators', 'data'),
        [State('memory-output','data'),
         State('current-program', 'data')]
)
def update_ranks(data, stored_data, program):
    df = pd.read_json(stored_data, orient='split')
    weights = get_weights(df)
    transform_col = transform_list(df)
    us_news_rank = get_us_news(df)
    df = clean_df(df)

    dff = df
    dff = calculate_ranks(dff, transform_col, weights)
    original_sim_rank = get_sim_rank(dff)
    
    updated_table = pd.DataFrame(data)
    
    columns = []
    values = []
    index = get_index(df)

    for row in range(len(updated_table)):
        columns.append(updated_table.iat[row,0])
        values.append(updated_table.iat[row, 2])
    df.loc[index,columns] = values

    df = calculate_ranks(df, transform_col, weights)
    new_sim_rank = get_sim_rank(df)
    current_program = program[0]['name']
    df = reorder_dataframe(df)
    return update_rank_number(df, current_program, original_sim_rank, new_sim_rank, us_news_rank)


@app.callback(
    Output({'type': 'dynamic-graph', 'index': 0}, 'figure'),
    Input({'type': 'dynamic-dropdown', 'index': 0}, 'value'),
    State('memory-output', 'data')
)
def update_bar(choice, data):
    df = pd.read_json(data, orient='split')
    df = clean_df(df)
    dff = rank_column(df, choice)
    line = create_linechart(dff, choice)
    return line


# Error checking. Comment out for faster preformance
@app.callback(
     Output("error-toast", "is_open"),
     [Input("datatable_indicators", "active_cell"),
      Input("datatable_indicators", 'data')],
     State('memory-output', 'data'),
     prevent_initial_call=True
)
def open_toast(cell, data, memory_data):
     df = pd.read_json(memory_data, orient='split')
     min = min_value(df)
     max = max_value(df)
     if cell is not None:
         new_entry = pd.DataFrame(data)
         return validate_value(new_entry, min, max)
     else:
         return dash.no_update


@app.callback(
    [Output('my-dropdown', 'options'),
     Output('uploaded_data', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('my-dropdown', 'options'),
    prevent_initial_call=True
)
def update_options(contents, filename, options):
    if contents is not None:
        uploaded_df = parse_contents(contents, filename)
        upload_name = os.path.splitext(filename)[0]
        dropdown_options = options
        new_option = {'label': str(upload_name), 'value': str(upload_name)}
        dropdown_options.append(new_option)
        return dropdown_options, uploaded_df.to_json(date_format='iso', orient='split')


# Start the app
if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
