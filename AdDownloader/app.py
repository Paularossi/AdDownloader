import base64
import io
import os
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table, no_update
from matplotlib.pyplot import sca
import plotly.express as px
import pandas as pd
from AdDownloader import analysis
from AdDownloader.helpers import update_access_token
from AdDownloader.media_download import start_media_download


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
    html.Div(id='output-graphs'),
    html.Div(id='output-text-analysis'),
    html.Div(id='output-topic-analysis'),
    html.Div(id='output-image-download'),
    html.Div(id='output-image-analysis'),
])


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return "data:image/png;base64," + encoded_string


def parse_contents(contents, filename):
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
        html.Hr(),

        html.H3("Table with your data:"),
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
        dcc.Store(id='stored-project-name', data = project_name),

        html.H2("Choose a task by clicking a button below."),
        html.Div([
            # generate graphs
            html.Button(id="graphs-button", children="Generate Graphs")
        ]),
        html.Div([
            # generate text analysis
            html.Button(id="text-analysis-button", children="Generate Text Analysis"),
            html.H5('Text analysis takes on average 15 seconds for every 1000 ads.', style={'marginLeft': '20px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            # generate topic modeling
            html.Button(id="topic-button", children="Generate Topic Analysis"),
            html.H5('Topic modeling takes on average 30 seconds for every 1000 ads.', style={'marginLeft': '20px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),

        html.Hr(),

        # start media download
        html.Div([
            html.H5('Or provide your access token and desired number of images to download.'),
            dcc.Input(id="access-tkn", type="text", placeholder="Insert your access token", debounce=True),
            dcc.Input(id="nr-ads", type="number", placeholder="Insert a number"),
            html.Button(id="media-download-button", children="Generate Image Analysis"),
            html.H5('The media download takes on average 1.5-2 minutes for every 50 ads.')
        ]),

        html.Hr(),
    ])

    return table_children


# update output of the table
@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        children = parse_contents(contents, filename)
        return children
    

@app.callback(Output("download-sent-dataframe", "data"),
             Input("btn_xlsx_sent", "n_clicks"),
             State('sentiment-data', 'data'),
             State('stored-project-name', 'data'),
             prevent_initial_call=True)
def download_sent_data(n_clicks, sent_data, project_name):
    try:
        df = pd.DataFrame(sent_data)
        return dcc.send_data_frame(df.to_excel, f"{project_name}_sentiment_data.xlsx", index=False)
    except:
        print('Unable to save sentiment data.')


@app.callback(Output("download-topic-dataframe", "data"),
             Input("btn_xlsx_topic", "n_clicks"),
             State('topic-data', 'data'),
             State('stored-project-name', 'data'),
             prevent_initial_call=True)
def download_topic_data(n_clicks, topic_data, project_name):
    try:
        df = pd.DataFrame(topic_data)
        return dcc.send_data_frame(df.to_excel, f"{project_name}_topic_data.xlsx", index=False)
    except:
        print('Unable to save topic data.')


@app.callback(Output("download-img-dataframe", "data"),
             Input("btn_xlsx_img_ft", "n_clicks"),
             State('all-img-features-data', 'data'),
             State('stored-project-name', 'data'),
             prevent_initial_call=True)
def download_img_features_data(n_clicks, data, project_name):
    try:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_excel, f"{project_name}_img_features_data.xlsx", index=False)
    except:
        print('Unable to save image features data.')


@app.callback(Output('output-graphs', 'children'),
              Input('graphs-button', 'n_clicks'),
              State('stored-data', 'data'))
def make_graphs(n, data):
    if n is None:
        return no_update
    
    else:
        try:
            df = pd.DataFrame(data)
            # not working - list has no attribute groupby
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = analysis.get_graphs(df)

            title = "Highest Average Impressions" if "impressions" in df.columns else "Biggest EU Reach"
            max_nr = max(df["impressions_avg"]) if "impressions" in df.columns else max(df["eu_total_reach"])

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
            html.Div([
                html.Div('Longest ad campaign', style={'textAlign': 'center'}),
                html.Div(max(df["campaign_duration"]), style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns'),
            html.Div([
                html.Div(title, style={'textAlign': 'center'}),
                html.Div(max_nr, style={'textAlign': 'center', 'fontWeight': 'bold'})
            ], className='three columns')
        ], className='row'),

        html.H2('Total Ad Reach'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph1', figure=fig1),
                html.H6('In this line graph, the ads were grouped by their `ad_delivery_start_date`, and for each of these ad campaign start dates, the total impressions/reach were computed.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph2', figure=fig2),
                html.H6('This histogram illustrates the distribution of the total average impressions/reach across all ads.')
            ], className='six columns')
        ], className='row'),

        html.H2('Number of Ads per Page'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph3', figure=fig3),
                html.H6('For every unique pair of page name and page ID, the number of published ads was counted. This histogram shows the distribution of the number of ads posted per page.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph4', figure=fig4),
                html.H6('Based on the total number of ads posted, this bar plot displays the Top 15 most active pages.')
            ], className='six columns')
        ], className='row'),

        html.H2('Total EU Reach per Page'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph5', figure=fig5),
                html.H6('For every unique pair of page name and page ID, the total number of impressions/reach were summed up. This histogram shows the distribution of the total average impressions/reach per page.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph6', figure=fig6),
                html.H6('Based on the total number impressions/reach, this bar plot displays the Top 15 most engaged with pages.')
            ], className='six columns')
        ], className='row'),

        html.H2('Campaign Duration'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph7', figure=fig7),
                html.H6('By taking the difference between `ad_delivery_stop_date` and `ad_delivery_start_date`, the ad campaign duration was obtained. This histogram illustrates the distribution of lenghts of ad campaigns.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph8', figure=fig8),
                html.H6('This scatter plot explores the relationship between the duration of ad campaigns and their total average impressions/reach.')
            ], className='six columns')
        ], className='row'),

        html.H2('Reach data by age and gender'),
        html.Div([
            html.Div([
                dcc.Graph(id='graph9', figure=fig9),
                html.H6('This violin plot displays the distribution of ad impressions/reach across all the age ranges (colored by age group). The shape of each age range provides insight into the density and spread of the reach.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='graph10', figure=fig10),
                html.H6('This violin plot shows the distribution of ad impressions/reach across all genders (colored by gender). The central tendency and variability of each gender offer a comparative view of engagement levels.')
            ], className='six columns')
        ], className='row'),

        html.Hr(),
        
    ])
        
    return graphs_children


@app.callback(Output('output-image-download', 'children'),
              Input('access-tkn', 'value'),
              Input('nr-ads', 'value'),
              Input('media-download-button', 'n_clicks'),
              State('stored-data', 'data'),
              State('stored-project-name', 'data'))
def start_image_download(access_token, nr_ads, n, data, project_name):
    if n is None or access_token is None or nr_ads is None:
        return no_update
    
    else:
        try:
            df = pd.DataFrame(data)
            df = update_access_token(df, str(access_token))
            start_media_download(str(project_name), int(nr_ads), data = df)
            img_path = f"output/{project_name}/ads_images"
            images = os.listdir(img_path)

            displayed_images = images[:5] if len(images) > 5 else images

            df = analysis.analyse_image_folder(img_path, project_name, nr_images=int(nr_ads))
            color_long_df = df.melt(value_vars=['dom_color_1', 'dom_color_2', 'dom_color_3'], 
                                    var_name='Color_Type', value_name='Color')
            color_counts = color_long_df['Color'].value_counts().reset_index()
            top_colors = color_counts.head(5)

            fig_colors = px.bar(top_colors, x='Color', y='count', title=f'Top 5 Dominant Colors Across All Images',
                        color='Color', text='count', color_discrete_map={color: color for color in top_colors['Color']})
            fig_colors.update_traces(textposition='inside', marker_line_color='black')

            # image quality, first normalize
            df["brightness_norm"] = (df["brightness"] - df["brightness"].min()) / (df["brightness"].max() - df["brightness"].min())
            df["contrast_norm"] = (df["contrast"] - df["contrast"].min()) / (df["contrast"].max() - df["contrast"].min())
            df["sharpness_norm"] = (df["sharpness"] - df["sharpness"].min()) / (df["sharpness"].max() - df["sharpness"].min())
            df_quality = df.melt(value_vars=['brightness_norm', 'contrast_norm', 'sharpness_norm'], 
                            var_name='Metric', value_name='Value')
            fig_qual = px.box(df_quality, x='Metric', y='Value', color='Metric',
                        title='Distribution of Image Quality Metrics')
            fig_qual.update_layout(xaxis_title="Quality Metric", yaxis_title="Value", legend_title="Metric")            

        except Exception as e:
            return html.Div([html.H5(f"Failed to start media download. Error: {e}")])
        
    img_children = html.Div([
        html.H2('Ad Images Download and Analysis.'),
        html.H4(f"Finished downloading media. Image examples (5 out of {len(images)}):"),
        html.Div(
            children=[html.Img(src=encode_image(os.path.join(img_path, img)), style={'width': '20%', 'display': 'inline-block'}) for img in displayed_images]
        ),
        # basic image analysis
        html.Div([
            html.Div([
                dcc.Graph(id='top-colors', figure=fig_colors),
                html.H6(f'For every downloaded image, three most dominant colors were extracted. Out of this set of {nr_ads}x3 colors, five most frequent colors are shown in this barplot.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='img-quality', figure=fig_qual),
                html.H6('This boxplot displays the distribution of image quality metrics across all images. The values were normalized for easier interpretation.')
            ], className='six columns')
        ], className='row'),
        dcc.Store(id='img-features-data', data=df.to_dict('records')),
        html.H2('Optional - Image Analysis using the BLIP model.'),
        html.Div([
            html.Button(id='image-captioning-button', children="Generate Image Captioning Using BLIP"),
            html.H5('This takes on average 2 minutes for every 50 ads.', style={'marginLeft': '20px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Hr(),
        #TODO: add option to input a question
    ])
    
    return img_children


@app.callback(Output('output-image-analysis', 'children'),
              Input('image-captioning-button', 'n_clicks'),
              Input('nr-ads', 'value'),
              State('stored-project-name', 'data'),
              State('img-features-data', 'data'),
              prevent_initial_call=True)
def start_media_captioning(n, nr_ads, project_name, data):
    if n is None:
        return no_update
    
    else:
        try:
            img_path = f"output/{project_name}/ads_images"
            features_data = pd.DataFrame(data)

            # image captioning with BLIP
            img_captions = analysis.blip_call(img_path, nr_images=int(nr_ads))

            # processing of the captions
            tokens, freq_dist, word_cl, textblb_sent, nltk_sent = analysis.start_text_analysis(img_captions["img_caption"])

            top_10_words = freq_dist.most_common(10)
            fig1=px.bar(x=[word for word, count in top_10_words], y=[count for word, count in top_10_words], 
                        labels={'x': 'Words', 'y': 'Frequency'}, title='Top 10 Most Frequent Words')
            fig1.update_xaxes(tickfont=dict(size=14))
            fig1.update_traces(text=[count for word, count in top_10_words], textposition='outside')

            lda_model, coherence, topics_df = analysis.get_topics(tokens)
            topics = [f"Topic {topic[0]}: {topic[1]}" for topic in lda_model.print_topics(num_words=8)]
            topic_distribution = topics_df['dom_topic'].value_counts().reset_index()

            # gather all img features
            all_features_data = features_data.merge(img_captions, how='inner', on='ad_id')
            all_features_data = pd.concat([all_features_data, textblb_sent, nltk_sent, topics_df], axis=1)

            fig2 = px.bar(topic_distribution, x='dom_topic', y='count', 
                        labels={'dom_topic': 'Dominant Topic', 'count': 'Count'}, 
                        title='Topic Distribution Across all Ads', color='dom_topic')
            fig2.update_layout(xaxis=dict(categoryorder='total descending'))

            if coherence < 0.4:
                qual = 'bad'
                qual_color = 'red'
            elif coherence >= 0.6:
                qual = 'good'
                qual_color = 'green'
            else:
                qual = 'average'
                qual_color = 'orange'

        except Exception as e:
            return html.Div([html.H5(f"Failed to perform ad media content analysis. Error: {e}")])
        
    img_analysis_children = html.Div([
        html.H2('Ad Media Captioning.'),
        #TODO: change column width in the table
        dash_table.DataTable(
            data=img_captions.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in img_captions.columns],
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

            # hover over a cell to see its contents
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()
                } for row in img_captions.to_dict('records')
            ],
            tooltip_duration=None
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='top-10-words-captions', figure=fig1)
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='topic-dist-captions', figure=fig2)
            ], className='six columns')
        ], className='row'),
        html.Div([
            html.H4("Topics:"),
            html.Ul([html.Li(topic) for topic in topics]),
            html.H4("Coherence Score:"),
            html.P([f"{coherence} - ", html.Span(qual, style={'color': qual_color})])
        ]),

        dcc.Store(id='all-img-features-data', data=features_data.to_dict('records')),
        html.Button("Download Image Features Data", id="btn_xlsx_img_ft"),
        dcc.Download(id="download-img-dataframe"),
        html.Hr(),
    ])
        
    return img_analysis_children


@app.callback(Output('output-text-analysis', 'children'),
              Input('text-analysis-button', 'n_clicks'),
              State('stored-data', 'data'))
def make_text_analysis(n, data):
    if n is None:
        return no_update
    else:
        try:
            df = pd.DataFrame(data)
            df = df.dropna(subset = ["ad_creative_bodies"])
            tokens, freq_dist, word_cl, textblb_sent, nltk_sent = analysis.start_text_analysis(df["ad_creative_bodies"])
            # add these to the dataframe and save it
            top_10_words = freq_dist.most_common(10)
            fig1=px.bar(x=[word for word, count in top_10_words], y=[count for word, count in top_10_words], 
                        labels={'x': 'Words', 'y': 'Frequency'}, title='Top 10 Most Frequent Words')
            fig1.update_xaxes(tickfont=dict(size=14))
            fig1.update_traces(text=[count for word, count in top_10_words], textposition='outside')

            sentiment_data = {'sentiment_category': [], 'score': []}
            for entry in nltk_sent:
                for category, score in entry.items():
                    if category != 'compound':
                        sentiment_data['sentiment_category'].append(category)
                        sentiment_data['score'].append(score)

            sent_df = pd.DataFrame(sentiment_data)
            sentiment_labels = {'pos': 'Positive', 'neg': 'Negative', 'neu': 'Neutral'}
            sent_df['sentiment_category'] = sent_df['sentiment_category'].map(sentiment_labels)

            #fig2=px.pie(sent_df, names='sentiment_category', values='score', title='Pie Chart of Total Sentiment Scores')
            fig2 = px.box(sent_df, x='sentiment_category', y='score', 
                          title='Distribution of Scores for Each Sentiment Type (using Vader from NLTK)',
                          labels={'sentiment_category': 'Sentiment', 'score': 'Score'},
                          color='sentiment_category')
            fig2.update_traces(marker_line_color='black')
            fig2.update_layout(barmode='relative', xaxis=dict(categoryorder='total descending'))

            # add the sentiment to the original data (to save it later)
            df["textblb_sent"] = textblb_sent
            df["nltk_sent"] = nltk_sent

        except Exception as e:
            return html.Div([html.H5(f"Failed to perform ad text analysis. Error: {e}")])
        
    text_children = html.Div([
        html.H2('Ad Creative Analysis - Word Count and Sentiment.'),
        html.H6("""This section analyses the text content of the ads, in terms of word count and sentiment.
                For each ad caption, preprocessing was performed, including tokenizing (breaking down the text into words), lowercasing, removing stopwords 
                and numbers (common words, e.g. "and", "the", etc.), and lemmatizing (reducing words to their root form, e.g. "running" -> "run")."""),
        html.Div([
            html.Div([
                dcc.Graph(id='top-10-words', figure=fig1),
                html.H6('After ad captions have been processed, the words frequency distribution is computed. This bar plot displays the Top 10 most frequent words across all ads.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='sentiment', figure=fig2),
                html.H6('This boxplot represents the distribution of the three sentiment types across all ad captions. The sentiment analysis was performed using Vader from the NLTK library, on the preprocessed ad captions.')
            ], className='six columns')
        ], className='row'),
        dcc.Store(id='sentiment-data', data=df.to_dict('records')),
        html.Button("Download Sentiment Data", id="btn_xlsx_sent"),
        dcc.Download(id="download-sent-dataframe"),
        html.Hr(),
    ])
        
    return text_children


@app.callback(Output('output-topic-analysis', 'children'),
              Input('topic-button', 'n_clicks'),
              State('stored-data', 'data'))
def make_topic_analysis(n, data):
    if n is None:
        return no_update
    else:
        try:
            df = pd.DataFrame(data)
            captions = df["ad_creative_bodies"].dropna()
            tokens = captions.apply(analysis.preprocess)
            lda_model, coherence, topics_df = analysis.get_topics(tokens)
            topics = [f"Topic {topic[0]}: {topic[1]}" for topic in lda_model.print_topics(num_words=8)]
            topic_distribution = topics_df['dom_topic'].value_counts().reset_index()

            fig1 = px.bar(topic_distribution, x='dom_topic', y='count', 
                        labels={'dom_topic': 'Dominant Topic', 'count': 'Count'}, 
                        title='Topic Distribution Across all Ads', color='dom_topic')
            fig1.update_layout(xaxis=dict(categoryorder='total descending'))

            topics_df = pd.concat([df, topics_df], axis=1)
            if coherence < 0.4:
                qual = 'bad'
                qual_color = 'red'
            elif coherence >= 0.6:
                qual = 'good'
                qual_color = 'green'
            else:
                qual = 'average'
                qual_color = 'orange'

            fig2 = analysis.show_topics_top_pages(topics_df)

        except Exception as e:
            return html.Div([html.H5(f"Failed to perform ad topic analysis. Error: {e}")])
        
    topic_children = html.Div([
        html.H2('Ad Creative Analysis - Topic Modeling.'),
        html.H6("""This section analyses the text content of the ads in terms of topics. 
                To peform a topic analysis, the ad captions must be preprocessed and passed as tokens. Tokens that occured in less than 5 ad captions, or tokens that occured in more than 90% of the captions were removed.
                A dictionary is created out of the processed tokens, and then a corpus is created from the dictionary. Next, using the Latent Dirichlet Allocation model, 
                three topics are found across all ads. Finally, a coherence score is computed for the discovered topics, that represents the semantic similarity and 
                co-occurrence patterns of high scoring words within each topic."""),
        html.Div([
            html.Div([
                dcc.Graph(id='topic-dist', figure=fig1),
                html.H6('This histogram shows the distribution of the 3 discovered topics across all ads.')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='topic-dist-top-pages', figure=fig2),
                html.H6('This bar plot includes only the Top 20 pages in the dataset by their total number of published ads. For each of these pages, the distribution of the three topics is shown.')
            ], className='six columns'),
        ], className='row'),
        dcc.Store(id='topic-data', data=topics_df.to_dict('records')),
        html.Div([
            html.H4("These are the three discovered topics and their first 8 words:"),
            html.Ul([html.Li(topic) for topic in topics]),
            html.H4("Coherence Score (how meaningful the topics are):"),
            html.P([f"{coherence} - ", html.Span(qual, style={'color': qual_color})])
        ]),        

        html.Button("Download Topic Data", id="btn_xlsx_topic"),
        dcc.Download(id="download-topic-dataframe"),
        html.Hr(),
    ])
        
    return topic_children


if __name__ == '__main__':
    app.run_server(debug=True)
