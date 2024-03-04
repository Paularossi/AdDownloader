import datetime
import base64
import io
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table, no_update
from matplotlib import table
import plotly.express as px
import pandas as pd
from AdDownloader import analysis

# add to dependencies: plotly==5.18.0(?)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

colors = {
    'background': '#fbe09c',
    'text': '#000000'
}


# define app layout
app.layout = html.Div([
    html.H1('AdDownloader Analytics'),
    html.H5('Add a file to start analysis.'),
    dcc.Upload(
        id = 'upload-data',
        children = html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style = {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple = False
    ),
    html.Div(id='output-datatable'),
    html.Div(id='output-div'),
])


def parse_contents(contents, filename, date):
    # parse the contents of the uploaded file and return a table of the data
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # assume that the user uploaded an excel file
            # df = pd.read_excel(io.BytesIO(decoded))
            df = analysis.load_data(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    try:
        project_name = filename.split('_')[0]
    except:
        project_name = filename

    table_children = html.Div([
        html.H5(f"Currently showing project {project_name}."),
        # html.P("Inset X axis data"),
        # dcc.Dropdown(id='xaxis-data',
        #              options=[{'label':x, 'value':x} for x in df.columns]),
        # html.P("Inset Y axis data"),
        # dcc.Dropdown(id='yaxis-data',
        #              options=[{'label':x, 'value':x} for x in df.columns]),
        html.Button(id="submit-button", children="Generate Graphs"),
        html.Hr(),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            fixed_rows={'headers': True},
            fixed_columns={'headers': True},
            page_size=15,
            # add scrolling option
            style_table={'overflowX': 'auto', 'overflowY': 'auto', 'height': '300px'},
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'minWidth': '150px', 
                'width': '150px', 
                'maxWidth': '150px',
            },
            #TODO: add date filter

            # hover over a cell to see its contents
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()
                } for row in df.to_dict('records')
            ],
            tooltip_duration=None
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr()

    ])

    return table_children


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, filename, last_modified):
    if contents is not None:
        #children = [parse_contents(c, n, d) for c, n, d in zip(contents, filename, last_modified)]
        children = parse_contents(contents, filename, last_modified)
        return children


@app.callback(Output('output-div', 'children'),
              Input('submit-button', 'n_clicks'),
              State('stored-data', 'data'))
def make_graphs(n, data):
    if n is None:
        return no_update
    
    else:
        try:
            df = pd.DataFrame(data)
            # not working - list has no attribute groupby
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = analysis.get_graphs(df)
        except Exception as e:
            return html.Div([html.H5(f"Failed to get graphs. Try uploading another file. Error: {e}")])
        
    graphs_children = html.Div([
        html.H2('Quick stats for your data'),
        html.Div([
            html.Div([
                html.Div('Total ads', style={'textAlign': 'center'}),
                html.Div(len(df), style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns'),
            html.Div([
                html.Div('Unique pages', style={'textAlign': 'center'}),
                html.Div(df["page_id"].nunique(), style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns'),
            # html.Div([
            #     html.Div('Ads targeted at teenagers', style={'textAlign': 'center'}),
            #     html.Div(len(df[df['target_ages'].str.contains("'13'")]), style={'textAlign': 'center', 'fontWeight': 'bold'})
            # ], className='three columns'),
            html.Div([
                html.Div('Longest ad campaign', style={'textAlign': 'center'}),
                html.Div(max(df["campaign_duration"]), style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns'),
            html.Div([
                html.Div('Biggest EU reach', style={'textAlign': 'center'}),
                html.Div(max(df["eu_total_reach"]), style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns')
        ], className='row'),

        html.H2('Total Ad Reach'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph1', figure=fig1)
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph2', figure=fig2)
            ], className='six columns')
        ], className='row'),

        html.H2('Number of Ads per Page'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph3', figure=fig3)
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph4', figure=fig4)
            ], className='six columns')
        ], className='row'),

        html.H2('Total EU Reach per Page'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph5', figure=fig5)
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph6', figure=fig6)
            ], className='six columns')
        ], className='row'),

        html.H2('Campaign Duration'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph7', figure=fig7)
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph8', figure=fig8)
            ], className='six columns')
        ], className='row'),

        html.H2('Ad Creative Analysis.'),
        html.H6('To be continued...', style={"color": "red"})
    ])
        
    return graphs_children


if __name__ == '__main__':
    app.run_server(debug=True)


""" 
    html.Div([
        html.H2(f'Creating analytics for project {project_name}.', className='sub-header'),

        html.Div([
            html.H3('First glance on your data', className='sub-header'),
            html.Div([
                html.Pre(data.head(20).to_html(), style={'overflowX': 'scroll'}),
            ]),
        ]),

        html.Div([
            html.H3('Filter your results by date:', className='sub-header'),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=data[DATE_MIN].min(),
                max_date_allowed=data[DATE_MAX].max(),
                initial_visible_month=data[DATE_MIN].min(),
                start_date=data[DATE_MIN].min(),
                end_date=data[DATE_MAX].max(),
                display_format='YYYY-MM-DD'
            ),
        ]),

        html.Div([
            html.H3('Quick stats for your data', className='sub-header'),
            html.Div([
                html.Div([
                    html.H4('Total ads'),
                    html.P(len(data))
                ]),
                html.Div([
                    html.H4('Unique pages'),
                    html.P(data['page_id'].nunique())
                ]),
                html.Div([
                    html.H4('Ads targeted at teenagers'),
                    html.P(len(data[data['target_ages'].str.contains("'13'")]))
                ]),
                html.Div([
                    html.H4('Longest ad campaign'),
                    html.P(max(data['campaign_duration']))
                ]),
                html.Div([
                    html.H4('Biggest EU reach'),
                    html.P(max(data['eu_total_reach']))
                ]),
            ]),
        ]),
    ]), """
