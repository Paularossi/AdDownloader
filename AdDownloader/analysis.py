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
from nltk import FreqDist
import nltk
#import spacy # use to lemmatize other languages than english
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
from textblob import TextBlob
from gensim.parsing.preprocessing import remove_stopwords
import plotly.express as px
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color as sk_color
from skimage.feature import canny, corner_harris, corner_peaks
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

# add to the dependencies: spacy==3.7.4

#nltk.download('omw-1.4')


DATE_MIN = 'ad_delivery_start_time'
DATE_MAX = 'ad_delivery_stop_time'

GENDERS = ['female', 'male', 'unknown']
AGE_RANGES = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

AD_PATTERN = re.compile(r"ad_([0-9]+)_(img|frame[0-9]+)\.(png|jpeg|jpg)")



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
    processed_text = []
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language='english')
    try: # otherwise throws an error if Nan
        #text = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in all_stopwords]
        # check if the stopwords are reliable
        all_stopwords = set(stopwords.words('english')) | \
                        set(stopwords.words('french')) | \
                        set(stopwords.words('dutch'))
        words = word_tokenize(text)
        for word in words:
            lower_word = word.lower()
            if lower_word.isalnum() and lower_word not in all_stopwords and not lower_word.isdigit():
                # lemmatization
                lemmatized_word = lemmatizer.lemmatize(lower_word)
                stemmed_word = stemmer.stem(lemmatized_word)
                processed_text.append(stemmed_word)
    except Exception as e:
        # text analysis
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('wordnet')
        pass
        #print(f"exception {e} occured for text {text}.")
    return processed_text


def get_word_freq(tokens):
    """
    Calculate word frequencies and generate a word cloud.

    :param tokens: A list of tokenized words, created using the `preprocess` function from this module.
    :type tokens: list
    :return: A tuple containing the word frequency distribution (FreqDist) and the generated word cloud (WordCloud).
    :rtype: tuple
    """
    # calculate word frequencies
    merged_tkn = []
    for lst in tokens:
        merged_tkn += lst
    fd = FreqDist(merged_tkn) # takes a list of strings as input
    # fd.tabulate()

    wc_tokens = ' '.join(merged_tkn)
    wc = WordCloud(collocations = False, background_color="white").generate(wc_tokens) # only takes a string as input

    return fd, wc


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


def get_topics(tokens, nr_topics = 3):
    """
    Perform topic modeling on a given set of tokens using Latent Dirichlet Allocation (LDA).
    The coherence score of the model can be improved by adjusting the number of topics or the hyperparameters of LDA.

    :param tokens: A list of tokenized words, created using the `preprocess` function from this module.
    :type tokens: list
    :param nr_topics: The number of topics to extract (default is 3).
    :type nr_topics: int, optional
    :return: A tuple containing the trained LDA model and the coherence score.
    :rtype: tuple
    """
    
    # create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokens) # only accepts an array of unicode tokens on input
    # the dictionary represents the index of the word and its frequency

    # filter out words that occur less than 5/20 documents, or more than 90% of the documents.
    if len(tokens) < 50:
        pass
    elif len(tokens) < 100:
        dictionary.filter_extremes(no_below = 5, no_above=0.9)
    else:
        dictionary.filter_extremes(no_below = 20, no_above=0.9)

    corpus = [dictionary.doc2bow(text) for text in tokens]
    print(f'Number of unique tokens: {len(dictionary)}')
    print(f'Number of documents: {len(corpus)}')

    # create a gensim lda model
    lda_model = LdaModel(corpus, id2word = dictionary, num_topics = nr_topics, update_every = 1, passes = 20, alpha='auto', eval_every = None)

    # evaluate model coherence - the degree of semantic similarity between high scoring words in each topic
    # c_v - frequency of the top words and their degree of co-occurrence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # to improve this score, we can adjust the number of topics, the hyperparameters of the lda model (alpha and beta), or experiment with preprocessing 

    # associate each caption with a topic
    topics_df = get_topic_per_caption(lda_model, corpus)
    #TODO: need to adapt for different languages?

    return lda_model, coherence_lda, topics_df


def get_topic_per_caption(lda_model, corpus, captions = None):
    """
    Extract the main topic per caption using a trained LDA model.

    :param lda_model: A trained LDA model.
    :type lda_model: gensim.models.LdaModel
    :param corpus: The corpus of documents (in Bag of Words format) used to train the LDA model.
    :type corpus: list of list of strings
    :param captions: A Series containing the original captions (optional).
    :type captions: pandas.Series
    :return: A DataFrame containing the dominant topic, percentage contribution, topic keywords, and the original caption for each document.
    :rtype: pandas.DataFrame
    """
    topics_df = pd.DataFrame(columns = ['dom_topic', 'perc_contr', 'topic_keywords'])
    
    # get main topic in each document
    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # get the dominant topic, percentage contribution and keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = pd.concat([topics_df, 
                                        pd.DataFrame({'dom_topic': [int(topic_num)], 'perc_contr': [round(prop_topic,4)], 'topic_keywords': [topic_keywords]})], 
                                        ignore_index=True)
            else:
                break

    if captions is not None:
        # add original text to the end of the output
        captions = captions.reset_index(drop = True)
        topics_df = pd.concat([topics_df, captions], axis=1)
    return(topics_df)
 

# main function
def start_text_analysis(text_data, topics = False):
    """
    Perform text analysis including preprocessing, word frequency calculation, sentiment analysis, and topic modeling.

    :param data: A pandas DataFrame containing an `ad_creative_bodies` column with ad captions.
    :type data: pandas.DataFrame
    :param topics: 
    :type topics: bool
    :return: A tuple containing the word frequency distribution, word cloud, sentiment scores using TextBlob and Vader, LDA model, and coherence score.
    :rtype: tuple
    """
    
    captions = text_data.dropna()
    tokens = captions.apply(preprocess)

    freq_dist, word_cl = get_word_freq(tokens)

    textblb_sent, nltk_sent = get_sentiment(captions)

    if topics:
        lda_model, coherence, sent_topics_df = get_topics(tokens, captions)
        return tokens, freq_dist, word_cl, textblb_sent, nltk_sent, lda_model, coherence, sent_topics_df
    else: 
        return tokens, freq_dist, word_cl, textblb_sent, nltk_sent


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


def show_topics_top_pages(topic_data, original_data):
    """
    Visualize the distribution of dominant topics for the top 20 pages by the number of ads.

    :param topic_data: A pandas DataFrame containing data with dominant topics.
    :type topic_data: pandas.DataFrame
    :param original_data: A pandas DataFrame containing the original ad data.
    :type original_data: pandas.DataFrame
    :return: A Plotly figure showing the distribution of dominant topics for the top 20 pages.
    :rtype: plotly.graph_objs._figure.Figure
    """
    original_data = original_data.dropna(subset = ["ad_creative_bodies"])
    original_data = original_data.reset_index(drop=True)
    topic_data = pd.concat([original_data, topic_data], axis=1)
    nr_ads_per_page = topic_data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")
    top_20_pages = nr_ads_per_page.sort_values(by="nr_ads", ascending=False)['page_id'].head(20) 

    filtered_data = topic_data[topic_data['page_id'].isin(top_20_pages)].dropna(subset = "dom_topic")

    # aggregate to count the number of ads per dominant topic for each page
    dom_topic_distribution = filtered_data.groupby(['page_name', 'dom_topic']).size().reset_index(name='nr_ads')

    fig = px.bar(dom_topic_distribution,
                x='page_name', y='nr_ads', color='dom_topic',
                title='Distribution of Dominant Topics for the Top 20 Pages by Number of Ads',
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
