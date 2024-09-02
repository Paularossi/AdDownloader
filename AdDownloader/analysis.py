"""This module provides different text and image analysis, and visualization functions for the AdDownloader."""

from math import sqrt, ceil
import numpy as np
import pandas as pd
import os
import re
import random
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from textblob import TextBlob
import plotly.express as px
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color as sk_color
from skimage.feature import canny, corner_harris, corner_peaks
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import ast
from itertools import combinations
import logging

#nltk.download('omw-1.4')s
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
# disable gensim logging
logging.getLogger('gensim').setLevel(logging.WARNING)

DATE_MIN = 'ad_delivery_start_time'
DATE_MAX = 'ad_delivery_stop_time'

GENDERS = ['female', 'male', 'unknown']
AGE_RANGES = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

AD_PATTERN = re.compile(r"ad_([0-9]+)_(img|frame[0-9]+)\.(png|jpeg|jpg)")

stop_words = set(stopwords.words('english')) # need to add more languages

def load_data(data_path):
    """
    Load ads data from an Excel file into a dataframe given a valid path.

    :param data_path: A valid path to an Excel file containing ad data.
    :type data_path: str
    :returns: A dataframe containing ads data and an additional campaign duration column.
    :rtype: pandas.DataFrame
    """
    try:
        data = pd.read_excel(data_path)
        data[DATE_MIN] = pd.to_datetime(data[DATE_MIN])
        data[DATE_MAX] = pd.to_datetime(data[DATE_MAX])

        data['campaign_duration'] = np.where(
            data['ad_delivery_stop_time'].isna(),
            (pd.Timestamp('today') - data['ad_delivery_start_time']).dt.days,
            (data['ad_delivery_stop_time'] - data['ad_delivery_start_time']).dt.days
        )
        return data
    except Exception as e:
        print(f"Unable to load data. Error: {e}")
        return None


def preprocess(text):
    """
    Preprocesses the input text by tokenizing, lowercasing, removing stopwords, and lemmatizing.

    :param text: The input text to be preprocessed.
    :type text: str
    :return: A list of preprocessed words.
    :rtype: list
    """
    try:
        text = ast.literal_eval(text)[0]
    except:
        pass # don't need to remove the square brackets
    
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language='english')
    try:
        words = word_tokenize(text)
        # first lemmatize then stem the words to create tokens
        tokens = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word.lower() not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        processed_text = " ".join(tokens)
        return processed_text
    except Exception as e:
        print(f"exception {e} occured for text {text}.")
        return ""


def get_word_freq(tokens):
    """
    Calculate word frequencies from a list of tokenized words using `CountVectorizer`.

    :param tokens: A list of tokenized words, created using the `preprocess` function from this module.
    :type tokens: list
    :return: A list of tuples, where each tuple contains a word and its corresponding frequency, sorted by frequency in descending order.
    :rtype: list of tuple
    """
    try:
        vectorizer = CountVectorizer(stop_words = list(stop_words), max_features = 1000, min_df = 5, max_df = 0.98)
        vect_text = vectorizer.fit_transform(tokens)
    except Exception as e: # too few tokens to prune, will consider all of them
        vectorizer = CountVectorizer(stop_words = list(stop_words), max_features = 1000)
        vect_text = vectorizer.fit_transform(tokens)
    
    tf_feature_names = vectorizer.get_feature_names_out()
    
    # sum the occurrences of each word across all documents
    freq = np.asarray(vect_text.sum(axis=0)).flatten()
    # create a dictionary mapping words to their frequencies
    word_freq = dict(zip(tf_feature_names, freq))
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    return sorted_word_freq


def get_sentiment(captions):
    """
    Retrieve the sentiment of the ad captions using two libraries: Vader from NLTK and TextBlob.

    :param captions: A pandas Series containing ad captions.
    :type captions: pandas.Series
    :return: A tuple containing sentiment scores calculated using TextBlob and Vader.
    :rtype: tuple
    """
    # sentiment analysis using textblob
    textblb_sent = captions.apply(lambda v: TextBlob(v).sentiment.polarity)

    # sentiment analysis using Vader from nltk
    sia = SentimentIntensityAnalyzer()
    nltk_sent = captions.apply(lambda v: sia.polarity_scores(v))

    return textblb_sent, nltk_sent


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float (intersection / union)


def get_topics(tokens, nr_topics = 3):
    """
    Perform topic modeling on a given set of tokens using Latent Dirichlet Allocation (LDA).
    The coherence score of the model can be improved by adjusting the number of topics or the hyperparameters of LDA.

    :param tokens: A list of tokenized words, created using the `preprocess` function from this module.
    :type tokens: list
    :param nr_topics: The number of topics to extract (default is 3).
    :type nr_topics: int, optional
    :return: A tuple containing the trained LDA model, the topics, the coherence score, perplexity, log-likelihood, similarity and a dataframe with a topic assigned to each ad.
    :rtype: tuple
    """    
    tokenized_docs = [doc.split() for doc in tokens]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # the dictionary represents the index of the word and its frequency
    print(f'Number of unique tokens: {len(dictionary)}')
    print(f'Number of documents: {len(corpus)}')
    
    try:
        vectorizer = CountVectorizer(stop_words = list(stop_words), max_features = 1000, min_df = 5, max_df = 0.95)
        vect_text = vectorizer.fit_transform(tokens)
    except Exception as e:
        vectorizer = CountVectorizer(stop_words = list(stop_words), max_features = 1000) # consider all tokens
        vect_text = vectorizer.fit_transform(tokens)
        
    tf_feature_names = vectorizer.get_feature_names_out()
    
    # create a gensim lda model
    lda_model = LatentDirichletAllocation(n_components=nr_topics, learning_method='online', max_iter=10, learning_decay=0.7, learning_offset=10).fit(vect_text) 
    
    # extract topics and words from the scikit-learn LDA model to calculate coherence and jaccard similarity
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [tf_feature_names[i] for i in topic.argsort()[:-8 -1:-1]]
        topics.append(topic_words)
    
    jaccard_sims = []
    for (topic1, topic2) in combinations(topics, 2):
        sim = jaccard_similarity(set(topic1), set(topic2))
        jaccard_sims.append(sim)
    
    avg_similarity = np.mean(jaccard_sims)
    
    # evaluate model coherence - the degree of semantic similarity between high scoring words in each topic
    # c_v - frequency of the top words and their degree of co-occurrence
    coherence_model = CoherenceModel(topics=topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    # to improve this score, we can adjust the number of topics, the hyperparameters of the lda model (alpha and beta), or experiment with preprocessing 

    # compute other metrics
    perplexity = lda_model.perplexity(vect_text) # how well the model generalizes to unseen data, the lower the better
    log_likelihood = lda_model.score(vect_text) # how well the model fits the data, the higher the better
    
    print(f"Finished topic modeling for {nr_topics} topics.\n" 
          f"Coherence: {round(coherence_lda, 2)}; Perplexity: {round(perplexity, 2)}; Log-Likelihood: {round(log_likelihood, 2)}; Similarity: {round(avg_similarity, 2)}\n")
    
    # print the topics
    for idx, topic in enumerate(lda_model.components_):
        # get the indices of the top words for this topic
        top_word_indices = topic.argsort()[:-8 -1:-1]
        top_words = [tf_feature_names[i] for i in top_word_indices]
        print(f"Topic {idx}: {top_words}")
    
    # associate each caption with a topic
    topics_df = get_topic_per_caption(lda_model, vect_text, tf_feature_names)

    return lda_model, topics, coherence_lda, perplexity, log_likelihood, avg_similarity, topics_df


def get_topic_per_caption(lda_model, vect_text, tf_feature_names, captions = None):
    """
    Extract the main topic per caption using a trained LDA model from scikit-learn.

    :param lda_model: A trained LDA model from scikit-learn.
    :type lda_model: sklearn.decomposition.LatentDirichletAllocation
    :param vect_text: The document-term matrix (output of the vectorizer).
    :type vect_text: scipy.sparse.csr_matrix
    :param tf_feature_names: The feature names (words) from the vectorizer.
    :type tf_feature_names: numpy.ndarray or list of str
    :param captions: A Series containing the original captions (optional).
    :type captions: pandas.Series
    :return: A DataFrame containing the dominant topic, percentage contribution, topic keywords, and the original caption for each document.
    :rtype: pandas.DataFrame
    """
    doc_topic_dist = lda_model.transform(vect_text)
    
    rows = []
    
    # iterate over each document's topic distribution
    for i, topic_dist in enumerate(doc_topic_dist):
        # get the dominant topic and its percentage contribution
        dom_topic_idx = np.argmax(topic_dist)
        perc_contr = topic_dist[dom_topic_idx]
        
        # get the keywords for the dominant topic
        topic_keywords = [tf_feature_names[i] for i in lda_model.components_[dom_topic_idx].argsort()[:-11:-1]]
        topic_keywords_str = ", ".join(topic_keywords)
        
        rows.append({
            'dom_topic': dom_topic_idx,
            'perc_contr': round(perc_contr, 4),
            'topic_keywords': topic_keywords_str
        })
    
    topics_df = pd.DataFrame(rows)
    if captions is not None:
        # add original text to the end of the output
        captions = captions.reset_index(drop=True)
        topics_df = pd.concat([topics_df, captions], axis=1)

    return(topics_df)


# main function
def start_text_analysis(text_data, column_name = "ad_creative_bodies", topics = False):
    """
    Perform text analysis including preprocessing, word frequency calculation, sentiment analysis, and topic modeling.
    If `topics = False`, the function will only return the tokens, frequency distribution, word cloud, and text sentiment, otherwise it will additionally return the LDA model, coherence and a dataframe with assigned topics to each ad.

    :param data: A pandas DataFrame containing an `ad_creative_bodies` column with ad captions.
    :type data: pandas.DataFrame
    :param column_name: The name of the column in the Data Frame that contains the text data (default is "ad_creative_bodies").
    :type column_name: str
    :param topics: If True, topic modelling will be performed in addition to the text and sentiment analysis.
    :type topics: bool
    :return: A tuple containing the tokens, word frequency distribution, sentiment scores using TextBlob and Vader, the LDA model and its metrics.
    :rtype: tuple
    """
    try:
        text_data = text_data.dropna(subset = column_name).copy()
    except Exception as e:
        print(f"Error occured when processing the text: {e}.")
    
    try:
        text_data.loc[:, column_name] = text_data[column_name].apply(lambda x: ast.literal_eval(x)[0])
    except:
        pass # don't need to remove the square brackets
    
    tokens = text_data[column_name].apply(preprocess)
    freq_dist = get_word_freq(tokens)

    textblb_sent, nltk_sent = get_sentiment(text_data[column_name])

    if topics:
        lda_model, topics, coherence_lda, perplexity, log_likelihood, avg_similarity, topics_df = get_topics(tokens)
        return tokens, freq_dist, textblb_sent, nltk_sent, lda_model, topics, coherence_lda, perplexity, log_likelihood, avg_similarity, topics_df
    else: 
        return tokens, freq_dist, textblb_sent, nltk_sent


def transform_data_by_age(data):
    """
    Transform demographic data into long format, separating data by age ranges.

    :param data: A pandas DataFrame containing demographic data.
    :type data: pandas.DataFrame
    :return: A pandas DataFrame with columns 'Reach' and 'Age Range' in long format.
    :rtype: pandas.DataFrame
    """

    # separate demographic columns into age ranges
    age_13_17_columns = [col for col in data.columns if '13-17' in col]
    age_18_24_columns = [col for col in data.columns if '18-24' in col]
    age_25_34_columns = [col for col in data.columns if '25-34' in col]
    age_35_44_columns = [col for col in data.columns if '35-44' in col]
    age_45_54_columns = [col for col in data.columns if '45-54' in col]
    age_55_64_columns = [col for col in data.columns if '55-64' in col]
    age_65_columns = [col for col in data.columns if '65+' in col]
    
    age_columns = []

    # check if 'age_13_17_columns' exist before including it
    if any('13-17' in col for col in data.columns):
        age_columns.append(data[age_13_17_columns].values.flatten())

    age_columns.extend([
        data[age_18_24_columns].values.flatten(),
        data[age_25_34_columns].values.flatten(),
        data[age_35_44_columns].values.flatten(),
        data[age_45_54_columns].values.flatten(),
        data[age_55_64_columns].values.flatten(),
        data[age_65_columns].values.flatten()
    ])

    long_data_age = pd.DataFrame({
        'Reach': [value for sublist in age_columns for value in sublist],  # flatten the list
        'Age Range': [label for label, sublist in zip(AGE_RANGES, age_columns) for _ in sublist]  # repeat labels accordingly
    })

    return long_data_age


def transform_data_by_gender(data):
    """
    Transform demographic data into long format, separating data by gender.

    :param data: A pandas DataFrame containing demographic data.
    :type data: pandas.DataFrame
    :return: A pandas DataFrame with columns 'Reach' and 'Gender' in long format.
    :rtype: pandas.DataFrame
    """

    # separate demographic columns into genders
    female_columns = [col for col in data.columns if 'female' in col]
    male_columns = [col for col in data.columns if 'male' in col and not 'female' in col]
    unknown_columns = [col for col in data.columns if 'unknown' in col]
    
    data_by_gender = [data[female_columns].values.flatten(), data[male_columns].values.flatten(), data[unknown_columns].values.flatten()]
    # transpose the data to have genders on the x-axis
    long_data_gender = pd.DataFrame({
        'Reach': [value for sublist in data_by_gender for value in sublist],  
        'Gender': [label for label, sublist in zip(GENDERS, data_by_gender) for _ in sublist]
    })

    return long_data_gender


def get_graphs(data):
    """
    Generate various graphs based on ad data. These include: 
    * Total reach by `ad_delivery_start_time`
    * Total reach distribution (overall)
    * Number of ads per page
    * Top 20 pages with most ads
    * Total reach by page
    * Top 20 pages by reach
    * Ad campaign duration distribution
    * Ad campaign duration vs. total reachs
    * Reach across age ranges
    * Reach across genders

    :param data: A pandas DataFrame containing ad data.
    :type data: pandas.DataFrame
    :return: A tuple containing multiple Plotly figures representing different visualizations.
    :rtype: tuple
    """
    
    pol_ads = True if "impressions" in data.columns else False
    # total reach/impressions by ad_delivery_start_time (cohort)
    if pol_ads: # political ads
        ad_start_cohort = data.groupby("ad_delivery_start_time")['impressions_avg'].sum().reset_index()
        fig1 = px.line(ad_start_cohort, x='ad_delivery_start_time', y='impressions_avg', 
                    title='Total Average Impressions by Ad Delivery Start Date',
                    labels={'ad_delivery_start_time': 'Ad Campaign Start', 'impressions_avg': 'Total Average Impressions'})

    else: # all ads
        ad_start_cohort = data.groupby("ad_delivery_start_time")['eu_total_reach'].sum().reset_index()
        fig1 = px.line(ad_start_cohort, x='ad_delivery_start_time', y='eu_total_reach', 
                   title='EU Total Reach by Ad Delivery Start Date',
                   labels={'ad_delivery_start_time': 'Ad Campaign Start', 'eu_total_reach': 'EU Total Reach'})

    # total reach/impressions distribution (overall) - VERY skewed
    if pol_ads:
        fig2 = px.histogram(data, x='impressions_avg',  title='Distribution of Total Average Impressions',
                            labels={'impressions_avg': 'Total Average Impressions', 'count': 'Number of Pages'})
        fig2.update_traces(marker_color='darkgreen', marker_line_color='black')
        fig2.update_layout(bargap=0.1, bargroupgap=0.05)
    
    else:
        fig2 = px.histogram(data, x='eu_total_reach',  title='Distribution of EU Total Reach',
                            labels={'eu_total_reach': 'EU Total Reach', 'count': 'Number of Pages'})
        fig2.update_traces(marker_color='darkgreen', marker_line_color='black')
        fig2.update_layout(bargap=0.1, bargroupgap=0.05)
    
    # number of ads per page - very skewed
    nr_ads_per_page = data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")
    fig3 = px.histogram(nr_ads_per_page, x='nr_ads', title='Distribution of Ads per Page',
                        labels={'nr_ads': 'Number of Ads', 'count': 'Number of Pages'})
    fig3.update_traces(marker_color='darkmagenta', marker_line_color='black')
    fig3.update_layout(bargap=0.1, bargroupgap=0.05)
    
    # top X pages with most ads
    nr_ads_per_page_sorted = nr_ads_per_page.sort_values(by="nr_ads", ascending=False).head(20) 

    fig4 = px.bar(nr_ads_per_page_sorted, x='page_name', y='nr_ads', 
                title=f'Top {len(nr_ads_per_page_sorted)} pages by number of ads',
                labels={'nr_ads': 'Number of Ads', 'page_name': 'Page Name'})
    fig4.update_xaxes(tickangle=45, tickmode='linear')
    fig4.update_traces(marker_color='darkorange', marker_line_color='black')
    fig4.update_layout(xaxis=dict(categoryorder='total descending'))

    # total reach/impressions per page - very skewed 
    if pol_ads:
        reach_by_page = data.groupby(["page_id", "page_name"])["impressions_avg"].sum().reset_index(name="impressions_avg")
        nbins = ceil(sqrt(len(reach_by_page)))

        fig5 = px.histogram(reach_by_page, x='impressions_avg', 
                        title='Distribution of Total Average Impressions per Page',
                        labels={'impressions_avg': 'Total Average Impressions', 'page_name': 'Page Name'}, nbins=nbins)
        fig5.update_traces(marker_color='yellowgreen', marker_line_color='black')
        fig5.update_layout(bargap=0.1, bargroupgap=0.05)

    else:
        reach_by_page = data.groupby(["page_id", "page_name"])["eu_total_reach"].sum().reset_index(name="eu_total_reach")

        fig5 = px.histogram(reach_by_page, x='eu_total_reach', 
                        title='Distribution of EU total reach per Page',
                        labels={'eu_total_reach': 'Total EU reach', 'page_name': 'Page Name'})
        fig5.update_traces(marker_color='yellowgreen', marker_line_color='black')
        fig5.update_layout(bargap=0.1, bargroupgap=0.05)

    # top 20 pages with highest total reach/impressions
    if pol_ads:
        reach_by_page_sorted = reach_by_page.sort_values(by="impressions_avg", ascending=False).head(20)

        fig6 = px.bar(reach_by_page_sorted, x='page_name', y='impressions_avg', 
                    title=f'Top {len(reach_by_page_sorted)} Pages by Total Average Impressions',
                    labels={'impressions_avg': 'Total Average Impressions', 'page_name': 'Page Name'})
        fig6.update_xaxes(tickangle=45, tickmode='linear')
        fig6.update_traces(marker_color='burlywood', marker_line_color='black')
        fig6.update_layout(xaxis=dict(categoryorder='total descending'))

    else:
        reach_by_page_sorted = reach_by_page.sort_values(by="eu_total_reach", ascending=False).head(20)

        fig6 = px.bar(reach_by_page_sorted, x='page_name', y='eu_total_reach', 
                    title=f'Top {len(reach_by_page_sorted)} Pages by EU Total Reach',
                    labels={'eu_total_reach': 'EU Total reach', 'page_name': 'Page Name'})
        fig6.update_xaxes(tickangle=45, tickmode='linear')
        fig6.update_traces(marker_color='burlywood', marker_line_color='black')
        fig6.update_layout(xaxis=dict(categoryorder='total descending'))

    # ad campain duration distribution
    fig7 = px.histogram(data, x='campaign_duration', 
                    title='Distribution of ad campaign duration',
                    labels={'campaign_duration': 'Duration of the ad campaign (days)'})
    fig7.update_traces(marker_color='turquoise', marker_line_color='black')
    fig7.update_layout(bargap=0.1, bargroupgap=0.05)

    campaign_duration_by_page = data.groupby(["page_id", "page_name"])["campaign_duration"].mean().reset_index(name="avg_campaign_duration")

    campaign_duration_by_page_sorted = campaign_duration_by_page.sort_values(by="avg_campaign_duration", ascending=False).head(20) 

    # relationship between campaign duration and total reach/impressions
    if pol_ads:
        fig8 = px.scatter(data, x='campaign_duration', y='impressions_avg', 
                    title='Campaign Duration vs. Total Average Impressions',
                    labels={'campaign_duration': 'Campaign Duration (Days)', 'impressions_avg': 'Total Average Impressions'})
        fig8.update_traces(marker_color='darkorchid', marker_line_color='black')

    else:
        fig8 = px.scatter(data, x='campaign_duration', y='eu_total_reach', 
                    title='Campaign Duration vs. EU Total Reach',
                    labels={'campaign_duration': 'Campaign Duration (Days)', 'eu_total_reach': 'EU Total Reach'})
        fig8.update_traces(marker_color='darkorchid', marker_line_color='black')
    
    # reach data by age and gender
    data_by_age = transform_data_by_age(data)
    data_by_gender = transform_data_by_gender(data)

    # reach across age ranges (all ads)
    fig9 = px.violin(data_by_age, y='Reach', x='Age Range', color='Age Range', title="Reach Across Age Ranges for all ads")
    
    # reach across genders (all ads)
    fig10 = px.violin(data_by_gender, y ='Reach', x='Gender', color='Gender', title="Reach Across Genders for all ads")
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10


def show_topics_top_pages(topics_df, original_data, n = 20):
    """
    Visualize the distribution of dominant topics for the top 20 pages by the number of ads.

    :param topics_df: A pandas DataFrame containing data with dominant topics.
    :type topics_df: pandas.DataFrame
    :param original_data: A pandas DataFrame containing the original ad data.
    :type original_data: pandas.DataFrame
    :param n: The number of top pages to show topics for (default is 20).
    :type n: int, optional
    :return: A Plotly figure showing the distribution of dominant topics for the top 20 pages.
    :rtype: plotly.graph_objs._figure.Figure
    """
    original_data = original_data.dropna(subset = ["ad_creative_bodies"])
    original_data = original_data.reset_index(drop=True)
    topic_data = pd.concat([original_data, topics_df], axis=1)
    nr_ads_per_page = topic_data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")
    top_n_pages = nr_ads_per_page.sort_values(by="nr_ads", ascending=False)['page_id'].head(n) 

    filtered_data = topic_data[topic_data['page_id'].isin(top_n_pages)].dropna(subset = "dom_topic")

    # aggregate to count the number of ads per dominant topic for each page
    dom_topic_distribution = filtered_data.groupby(['page_name', 'dom_topic']).size().reset_index(name='nr_ads')

    fig = px.bar(dom_topic_distribution,
                x='page_name', y='nr_ads', color='dom_topic',
                title=f'Distribution of Dominant Topics for the Top {len(top_n_pages)} Pages by Number of Ads',
                labels={'nr_ads': 'Number of Ads', 'page_name': 'Page Name', 'dom_topic': 'Dominant Topic'})

    fig.update_xaxes(tickangle=45, tickmode='linear', categoryorder='total descending')
    fig.update_traces(marker_line_color='black')
    fig.update_layout(legend_title_text='Dominant Topic')

    return fig


# image captioning
def blip_call(images_path, task = "image_captioning", nr_images = None, questions = None):
    """
    Perform image captioning or visual question answering using the BLIP model on a set of images.

    :param images_path: Path to the directory containing images.
    :type images_path: str
    :param task: The task to perform ("image_captioning" is default or "visual_question_answering").
    :type task: str, optional
    :param nr_images: The number of images to process (default is None, which processes all images in the directory).
    :type nr_images: int, optional
    :param questions: A string containing one or more questions separated by question marks, used for visual question answering (default is None).
    :type questions: str, optional
    :return: A pandas DataFrame containing image captions or answers to the provided questions.
    :rtype: pandas.DataFrame
    """
    # processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_captioning = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model_answering = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    if task == "visual_question_answering": 
        if questions is None:
            print("No question provided.")
            return
        else:
            # extract the questions
            questions_all = [question.strip() + "?" for question in questions.split('?')[:-1]]
    
    image_files = [f for f in os.listdir(images_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    if nr_images is None or nr_images > len(image_files):
        nr_images = len(image_files)
    else:
        image_files = random.sample(image_files, nr_images)
    #tasks = ["image_captioning", "visual_question_answering"]
    rows_list = []
    print(f"nr_images: {nr_images}, questions: {questions}")

    for image_file in image_files:
        raw_image = Image.open(os.path.join(images_path, image_file)).convert('RGB')
        numbers = AD_PATTERN.findall(image_file)
        ad_ids = [match[0] for match in numbers]
        ad_id = '_'.join(ad_ids) # extract ad id

        # question answering
        if task == "visual_question_answering":
        
            dict = {'ad_id': ad_id}
            for question in questions_all:
                inputs_quest = processor(raw_image, question, return_tensors="pt")

                out_quest = model_answering.generate(**inputs_quest, max_length = 30)
                #print(processor.decode(out_quest[0], skip_special_tokens=True))
                dict[question] = processor.decode(out_quest[0], skip_special_tokens=True)

        # image_captioning
        else:
            inputs_cpt = processor(raw_image, return_tensors="pt")
            out_cpt = model_captioning.generate(**inputs_cpt, max_length = 20)
            #print(processor.decode(out_cpt[0], skip_special_tokens=True))
            dict = {'ad_id': ad_id, 'img_caption': processor.decode(out_cpt[0], skip_special_tokens=True)}

        rows_list.append(dict)

        print(f'Done with {task} of ad with id {ad_id}') 
        

    img_content = pd.DataFrame(rows_list)
    return(img_content)


def extract_dominant_colors(image_path, num_colors = 3):
    """
    Extracts the dominant colors from an image using KMeans clustering.

    This function processes the image by resizing it for efficiency and reshaping it into a list of pixels. It then uses KMeans clustering to find the most common colors in the image. The dominant colors are returned as HEX codes along with their percentages in the image.

    :param image_path: The file path of the image to be analyzed.
    :type image_path: str
    :param num_colors: The number of dominant colors to extract from the image. Defaults to 3.
    :type num_colors: int, optional
    :return: A tuple of two lists: the first list contains the HEX codes of the dominant colors, and the second list contains the percentages of these colors within the image.
    :rtype: tuple(list, list)
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image to speed up processing
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    reshaped_image = resized_image.reshape((resized_image.shape[0] * resized_image.shape[1], 3))

    # find and display the most common colors
    clf = KMeans(n_clusters=num_colors, n_init=10)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_

    # calculate percentages
    total_pixels = len(reshaped_image)
    color_percentages = [(counts[i] / total_pixels) * 100 for i in counts.keys()]

    colors_and_percentages = list(zip(center_colors, color_percentages))

    # Sort the colors by percentage in descending order
    sorted_colors_and_percentages = sorted(colors_and_percentages, key=lambda cp: cp[1], reverse=True)
    ordered_colors = [cp[0] for cp in sorted_colors_and_percentages]
    sorted_percentages = [cp[1] for cp in sorted_colors_and_percentages]

    hex_colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in ordered_colors]

    #plt.figure(figsize=(8, 6))
    #plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    #plt.show()

    return hex_colors, sorted_percentages


def assess_image_quality(image_path):
    """
    Assesses the quality of an image based on its resolution, brightness, contrast, and sharpness.

    This function calculates the resolution as the product of image width and height, evaluates brightness as the average value of the pixel intensity, measures contrast as the standard deviation of grayscale pixel intensities, and assesses sharpness by the variance of the Laplacian of the grayscale image. High values of sharpness and contrast generally indicate a higher-quality image.

    :param image_path: The file path of the image to assess.
    :type image_path: str
    :return: A tuple containing the resolution (as a single integer value of width multiplied by height), brightness (average pixel intensity), contrast (standard deviation of pixel intensities), and sharpness (variance of the Laplacian) of the image.
    :rtype: tuple(int, float, float, float)
    """
    with Image.open(image_path) as img:
        # resolution
        width, height = img.size
        resolution = width * height
        gray_img = img.convert('L')

        # brightness
        brightness = sum(img.getpixel((x, y))[0] for x in range(width) for y in range(height)) / (width * height)

        # sharpness and contrast
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        # a higher variance indicates that the image is sharp, with clear edges and details
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # contrast indicates the spread of brightness levels across the image
        # a higher std suggests a higher contrast, the image has a wide range of tones from dark to light
        contrast = gray.std()

        return resolution, brightness, contrast, sharpness
    

# analyse an image including all features of interest
def analyse_image(image_path):
    """
    Analyzes an image by extracting its dominant colors, assessing image quality (resolution, brightness, contrast, sharpness), and detecting edges and corners.

    This comprehensive analysis function performs multiple assessments on an image. It extracts the dominant colors and their proportions, calculates quality metrics such as resolution, brightness, contrast, and sharpness, and performs edge detection and corner detection to analyze the image's structure. Additionally, it extracts an advertisement ID from the image file path using a predefined pattern.

    :param image_path: The file path of the image to be analyzed.
    :type image_path: str
    :return: A dictionary containing the analysis results, including ad ID, dominant colors and their proportions, resolution, brightness, contrast, sharpness, and the number of corners detected.
    :rtype: dict
    """

    # extract dominant colors
    dominant_colors, percentages = extract_dominant_colors(image_path)

    resolution, brightness, contrast, sharpness = assess_image_quality(image_path)
    img = Image.open(image_path)
    img = sk_color.rgb2gray(img)

    # edge detection
    # higher the sigma smoother the image (focus on the outside shape)
    canny_edges = canny(img, sigma=1.8)

    # corners - meh
    measure_image = corner_harris(img)
    coords = corner_peaks(measure_image, min_distance=32, threshold_rel=0.02)
    ncorners = len(coords)
    #print("With a min_distance set to 32, we detect a total", len(coords), "corners in the image.")

    numbers = AD_PATTERN.findall(image_path)
    ad_ids = [match[0] for match in numbers]
    ad_id = '_'.join(ad_ids) # extract ad id

    dict = {'ad_id': ad_id, 'resolution': resolution, 'brightness': brightness, 'contrast': contrast, 
            'sharpness': sharpness, 'ncorners': ncorners}
    
    for i, (color, percentage) in enumerate(zip(dominant_colors, percentages), start=1):
        dict[f'dom_color_{i}'] = color
        dict[f'dom_color_{i}_prop'] = percentage
    return dict


def analyse_image_folder(folder_path, nr_images = None):
    """
    Analyzes a set of images in a specified folder and exports the results to an Excel file.

    This function iterates over image files in the specified folder, performing an analysis on each image using the `analyse_image` function. The analysis covers extracting dominant colors, assessing image quality (resolution, brightness, contrast, sharpness), and detecting edges and corners. The results are compiled into a pandas DataFrame and then exported to an Excel file.

    :param folder_path: The path to the folder containing the image files to be analyzed. The folder can contain images in jpg, png, and jpeg formats.
    :type folder_path: str
    :param nr_images: The number of images to analyze from the folder. If None, all images in the folder are analyzed. This parameter allows for limiting the analysis to a subset of images.
    :type nr_images: int, optional
    :return: A pandas DataFrame containing the analysis results for each image.
    :rtype: pandas.DataFrame
    """

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
    if nr_images is None:
        nr_images = len(image_files)

    analysis_results = []
    i = 0

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        analysis_result = analyse_image(image_path)
        analysis_results.append(analysis_result)
        i += 1
        if i == nr_images:
            break

    df = pd.DataFrame(analysis_results)

    return df
