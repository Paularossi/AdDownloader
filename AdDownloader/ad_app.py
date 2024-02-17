import streamlit as st
import pandas as pd
import plotly.express as px
# from plotly.subplots import make_subplots
import plotly.graph_objects as go

# text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import nltk
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
from textblob import TextBlob


DATE_MIN = 'ad_delivery_start_time'
DATE_MAX = 'ad_delivery_stop_time'

# list of demographic columns
DEMOGRAPHIC_COLS = ['female_13-17', 'female_18-24', 'female_25-34', 'female_35-44', 'female_45-54', 'female_55-64', 'female_65+',
                        'male_13-17', 'male_18-24', 'male_25-34', 'male_35-44', 'male_45-54', 'male_55-64', 'male_65+',
                        'unknown_13-17', 'unknown_18-24', 'unknown_25-34', 'unknown_35-44', 'unknown_45-54', 'unknown_55-64', 'unknown_65+']
GENDERS = ['female', 'male', 'unknown']
AGE_RANGES = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

# separate demographic columns into genders
female_columns = [col for col in DEMOGRAPHIC_COLS if 'female' in col]
male_columns = [col for col in DEMOGRAPHIC_COLS if 'male' in col and not 'female' in col]
unknown_columns = [col for col in DEMOGRAPHIC_COLS if 'unknown' in col]

# separate demographic columns into age ranges
age_13_17_columns = [col for col in DEMOGRAPHIC_COLS if '13-17' in col]
age_18_24_columns = [col for col in DEMOGRAPHIC_COLS if '18-24' in col]
age_25_34_columns = [col for col in DEMOGRAPHIC_COLS if '25-34' in col]
age_35_44_columns = [col for col in DEMOGRAPHIC_COLS if '35-44' in col]
age_45_54_columns = [col for col in DEMOGRAPHIC_COLS if '45-54' in col]
age_55_64_columns = [col for col in DEMOGRAPHIC_COLS if '55-64' in col]
age_65_columns = [col for col in DEMOGRAPHIC_COLS if '65+' in col]


# check if the stopwords are reliable
def get_all_stopwords():
    try:
        all_stopwords = set(stopwords.words('english')) | \
                        set(stopwords.words('french')) | \
                        set(stopwords.words('dutch')) | \
                        set(stopwords.words('german'))
    except AttributeError:
        nltk.download('stopwords')
        all_stopwords = set(stopwords.words('english')) | \
                        set(stopwords.words('french')) | \
                        set(stopwords.words('dutch')) | \
                        set(stopwords.words('german'))
    return all_stopwords


def preprocess(text):
    processed_text = []
    lemmatizer = WordNetLemmatizer()
    all_stopwords = get_all_stopwords()

    try: # otherwise throws an error if Nan
        words = word_tokenize(text)
        for word in words:
            lower_word = word.lower()
            if lower_word.isalnum() and lower_word not in all_stopwords:
                # lemmatization
                lemmatized_word = lemmatizer.lemmatize(lower_word)
                processed_text.append(lemmatized_word)
    except Exception as e:
        pass
        #print(f"exception {e} occured for text {text}.")
    return processed_text


def get_word_freq(tokens):
    # calculate word frequencies
    merged_tkn = []
    for lst in tokens:
        merged_tkn += lst
    fd = FreqDist(merged_tkn) # takes a list of strings as input
    #print(f"Most common 10 keywords: {fd.most_common(10)}")

    wc_tokens = ' '.join(merged_tkn)
    wc = WordCloud(background_color="white").generate(wc_tokens) # only takes a string as input

    return fd, wc


@st.cache_data
def get_sentiment(captions):
    # sentiment analysis using textblob
    textblb_sent = captions.apply(lambda v: TextBlob(v).sentiment.polarity)

    # sentiment analysis using Vader from nltk
    sia = SentimentIntensityAnalyzer()
    nltk_sent = captions.apply(lambda v: sia.polarity_scores(v))

    return textblb_sent, nltk_sent


@st.cache_data
def get_topics(tokens):
    # create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokens) # only accepts an array of unicode tokens on input
    corpus = [dictionary.doc2bow(text) for text in tokens]

    # create a gensim lda models
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    # evaluate model coherence - the degree of semantic similarity between high scoring words in each topic
    # c_v - frequency of the top words and their degree of co-occurrence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # to improve this score, we can adjust the number of topics, the hyperparameters of the lda model (alpha and beta), or experiment with preprocessing 

    #TODO: need to adapt for different languages?

    return lda_model, coherence_lda


#@st.cache_data
def load_data(project_name):
    data_path = f"output/{project_name}/ads_data/processed_data.xlsx"
    data = pd.read_excel(data_path)
    data[DATE_MIN] = pd.to_datetime(data[DATE_MIN])
    data[DATE_MAX] = pd.to_datetime(data[DATE_MAX])
    data["campaign_duration"] = (data["ad_delivery_stop_time"] - data["ad_delivery_start_time"]).dt.days

    # check if the age_country_gender_reach_breakdown column has been processed
    if not (any(col in data.columns for col in DEMOGRAPHIC_COLS)):
        #data['flattened_data'] = data['age_country_gender_reach_breakdown'].apply(flatten_age_country_gender, target_country=country)
        # create a new DataFrame from the flattened data
        #flattened_df = pd.DataFrame(data['flattened_data'].sum()) # DONT FORGET TO CHANGE HERE
        print("`age_country_gender_reach_breakdown` column needs to be processed.")
    return data


# transpose the data to have age ranges on the x-axis
def transform_data_by_age(data):
    age_columns = []

    # check if 'age_13_17_columns' exist before including them
    if 'male_13-17' in data.columns:
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
        'Reach': [value for sublist in age_columns for value in sublist],  # Flatten the list
        'Age Range': [label for label, sublist in zip(AGE_RANGES, age_columns) for _ in sublist]  # Repeat labels accordingly
    })

    return long_data_age


def transform_data_by_gender(data):
    data_by_gender = [data[female_columns].values.flatten(), data[male_columns].values.flatten(), data[unknown_columns].values.flatten()]
    # transpose the data to have genders on the x-axis
    long_data_gender = pd.DataFrame({
        'Reach': [value for sublist in data_by_gender for value in sublist],  
        'Gender': [label for label, sublist in zip(GENDERS, data_by_gender) for _ in sublist]
    })

    return long_data_gender



st.set_page_config(page_title='AdDownloader Analytics', page_icon=':bar_chart:', layout='wide')

st.title(' :bar_chart: Sample AdDownloader Analytics')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# do this when integrated into the CLI:
# if len(sys.argv) > 1:
#     project_name = sys.argv[-1] # last arg
# else:
#     project_name = 'temp'
#     st.write("No data file path provided.")

project_name = "teststream"
data = load_data(project_name)

# START HERE
# just show the first 10 rows of the data
st.subheader('First glance on your data')
st.dataframe(data.head(20))

st.subheader('Filter your results by date:')
col1, col2 = st.columns((2))

# getting the min and max date 
startDate = pd.to_datetime(data[DATE_MIN].min())
endDate = pd.to_datetime(data[DATE_MAX].max())

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

data = data[(data[DATE_MIN] >= date1) & (data[DATE_MAX] <= date2)].copy()
#TODO: the filter is not working for the text analysis


# quick stats
st.subheader('Quick stats for your data')
unique_pages = data["page_id"].nunique()
col1, col2, col3 = st.columns(3)
col1.metric(":green[Total ads]", len(data))
col2.metric(":green[Unique pages]", unique_pages)
col3.metric("Think of something else", "-----86%")


# next, visualize the reach data:
col4, col5 = st.columns(2)
# transform the age data
data_by_age = transform_data_by_age(data)
data_by_gender = transform_data_by_gender(data)

#st.subheader('Reach, Page and Campaign analysis')

with col4:
    # EU total reach by ad_delivery_start_time (cohort)
    ad_start_cohort = data.groupby("ad_delivery_start_time")['eu_total_reach'].sum().reset_index()
    fig = px.line(ad_start_cohort, x='ad_delivery_start_time', y='eu_total_reach', 
                  title='EU Total Reach by Ad Delivery Start Date')
    st.plotly_chart(fig, use_container_width=True)

    # distribution of ads per page
    nr_ads_per_page = data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")
    fig = px.histogram(nr_ads_per_page, x='nr_ads', title='Distribution of Ads per Page')
    fig.update_traces(marker_color='gold')
    st.plotly_chart(fig, use_container_width=True)

    # EU reach per page
    reach_by_page = data.groupby(["page_id", "page_name"])["eu_total_reach"].sum().reset_index(name="eu_total_reach")
    fig = px.histogram(reach_by_page, x='eu_total_reach', title='Distribution of EU Total Reach per Page')
    fig.update_traces(marker_color='#F08080')
    st.plotly_chart(fig, use_container_width=True)

    # reach across age ranges (all ads)
    fig = px.box(data_by_age, y = 'Reach', x = 'Age Range', color='Age Range', template = "seaborn", title="Reach Across Age Ranges for all ads")
    st.plotly_chart(fig, use_container_width=True, height = 200)

    # distribution of ad campaign duration
    fig = px.histogram(data, x='campaign_duration', title='Distribution of Ad Campaign Duration')
    fig.update_traces(marker_color='green')
    st.plotly_chart(fig, use_container_width=True)
    

with col5:
    # total reach distribution (overall) - VERY skewed
    fig = px.histogram(data, x='eu_total_reach',  title='Distribution of EU Total Reach')
    fig.update_traces(marker_color='orange')
    st.plotly_chart(fig, use_container_width=True)

    # top 20 pages by number of ads
    top_pages_by_ads = nr_ads_per_page.sort_values(by="nr_ads", ascending=False).head(20)
    fig = px.bar(top_pages_by_ads, x='page_name', y='nr_ads', title='Top 20 Pages by Number of Ads')
    fig.update_traces(marker_color='purple')
    st.plotly_chart(fig, use_container_width=True)

    # top 20 pages with highest total reach
    top_pages_by_reach = reach_by_page.sort_values(by="eu_total_reach", ascending=False).head(20) 
    fig = px.bar(top_pages_by_reach, x='page_name', y='eu_total_reach', title='Top 20 Pages by EU Total Reach')
    fig.update_traces(marker_color='#48D1CC')
    st.plotly_chart(fig, use_container_width=True)

    # reach across genders (all ads)
    fig = px.box(data_by_gender, y = 'Reach', x = 'Gender', color='Gender', title="Reach Across Genders for all ads")
    st.plotly_chart(fig, use_container_width=True, height = 200)

    # campaign duration vs. EU total reach
    fig = px.scatter(data, x='campaign_duration', y='eu_total_reach', title='Campaign Duration vs. EU Total Reach')
    fig.update_traces(marker_color='#DDA0DD')
    st.plotly_chart(fig, use_container_width=True)


# finally, add the analysis of the text content:
st.subheader('Text Analysis Results')
col6, col7 = st.columns(2)

captions = data["ad_creative_bodies"].dropna()
tokens = captions.apply(preprocess)

# wordcloud
with col6:
    st.subheader('Word Cloud')
    _, wc = get_word_freq(tokens) 
    wc_image = wc.to_array()
    st.image(wc_image, use_column_width=True)

with col7:
    # sentiment
    st.subheader('Sentiment Analysis')
    textblb_sent, nltk_sent = get_sentiment(captions) 
    avg_polarity = textblb_sent.mean()
    st.write(f"Average Sentiment Polarity: {avg_polarity}")

    # topic modeling
    #st.subheader('Topics from LDA Model')
    #lda_model, coherence = get_topics(tokens)
    # for idx, topic in lda_model.print_topics(num_words=5):
    #     #st.write("Topic: {} \nWords: {}".format(idx + 1, topic))
    #     st.write(f"Topic: {topic[0]} Words: {topic[1]}")
    # #BUG: nothing is being shown here
    # st.write("Coherence Score: ", coherence)

    # if lda_model:
    #     topics = lda_model.show_topics(formatted=True, num_topics=3, num_words=5)
    #     for topic in topics:
    #         st.write(f"Topic: {topic[0]} Words: {topic[1]}")
    #     st.write("Coherence Score: ", coherence)
    # else:
    #     st.write("No topics generated.")

#TODO: add blip captioning from transformers (HF) + other image analysis



# PLOT IDEAS FOR THE DASHBOARD:
# 1. Gender and Age Group Reach - barcharts, heatmap (for showing all together)
# 2. Reach by Location - map? or barchart
# 3. Ad Creative Analysis - word cloud, most common words/topics
# 4. Time Series Analysis - line chart over time using ad_delivery_start_time and ad_delivery_stop_time
# 5. Ads Activity Duration - histogram, show the distribution
# 6. Page Activity - barchart, could also show like a number of unique page ids
# 7. Ad Engagement by Target Gender/Age - compare with the original target gender and age 