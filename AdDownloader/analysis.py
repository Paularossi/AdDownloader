"""This module provides different text and image analysis functions for the AdDownloader."""

#import cv2
import numpy as np
#from PIL import Image, ImageEnhance, ImageStat
import pandas as pd
import os
#from sklearn.cluster import KMeans
#from skimage import color, data
#from skimage.feature import canny, corner_harris, corner_peaks
#from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
#import nltk
#import spacy # use to lemmatize other languages than english
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
from textblob import TextBlob
from gensim.parsing.preprocessing import remove_stopwords
import plotly.express as px



# add to the dependencies: transformers==4.37.2, torch==2.2.0, torchvision==0.17.0 (check if needed?), 
# wordcloud==1.9.3, textblob==0.17.1, gensim==4.3.2, spacy==3.7.4

# text analysis
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('vader_lexicon')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# tokenize and remove stopwords
all_stopwords = set(stopwords.words('english')) | \
                set(stopwords.words('french')) | \
                set(stopwords.words('dutch'))
# check if the stopwords are reliable

DATE_MIN = 'ad_delivery_start_time'
DATE_MAX = 'ad_delivery_stop_time'

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


def load_data(data_path):
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
        #print(f"Error while trying to load the data. Check if there exists a file output/{project_name}/ads_data/processed_data.xlsx")
        print("Unable to load data.")
        return None


def preprocess(text):
    processed_text = []
    lemmatizer = WordNetLemmatizer()
    try: # otherwise throws an error if Nan
        #text = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in all_stopwords]
        words = word_tokenize(text)
        for word in words:
            lower_word = word.lower()
            if lower_word.isalnum() and lower_word not in all_stopwords and not lower_word.isdigit():
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
    #fd.tabulate()
    #print(f"Most common 10 keywords: {fd.most_common(10)}")

    wc_tokens = ' '.join(merged_tkn)
    wc = WordCloud(background_color="white").generate(wc_tokens) # only takes a string as input
    #plt.imshow(wc, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

    return fd, wc


def get_sentiment(captions):
    # sentiment analysis using textblob
    textblb_sent = captions.apply(lambda v: TextBlob(v).sentiment.polarity)

    # sentiment analysis using Vader from nltk
    sia = SentimentIntensityAnalyzer()
    nltk_sent = captions.apply(lambda v: sia.polarity_scores(v))

    return textblb_sent, nltk_sent


def get_topics(tokens):
    # TOPIC MODELING
    # create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokens) # only accepts an array of unicode tokens on input
    corpus = [dictionary.doc2bow(text) for text in tokens]

    # create a gensim lda models
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    #for idx, topic in lda_model.print_topics(num_words=5):
    #    print("Topic: {} \nWords: {}".format(idx + 1, topic))
    #    print("\n")

    # evaluate model coherence - the degree of semantic similarity between high scoring words in each topic
    # c_v - frequency of the top words and their degree of co-occurrence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score:', coherence_lda)
    # to improve this score, we can adjust the number of topics, the hyperparameters of the lda model (alpha and beta), or experiment with preprocessing 

    #TODO: need to adapt for different languages?

    return lda_model, coherence_lda 


# main function
def start_analysis(data):
    captions = data["ad_creative_bodies"].dropna()
    tokens = captions.apply(preprocess)

    fd, wc = get_word_freq(tokens)

    get_sentiment(captions)

    lda_model, coherence = get_topics(tokens)


def get_graphs(data):
    
    # total reach by ad_delivery_start_time (cohort)
    ad_start_cohort = data.groupby("ad_delivery_start_time")['eu_total_reach'].sum().reset_index()
    fig1 = px.line(ad_start_cohort, x='ad_delivery_start_time', y='eu_total_reach', 
                   title='EU Total Reach by Ad Delivery Start Date',
                   labels={'ad_delivery_start_time': 'Ad Campaign Start', 'eu_total_reach': 'EU Total Reach'})

    # total reach distribution (overall) - VERY skewed
    fig2 = px.histogram(data, x='eu_total_reach',  title='Distribution of EU Total Reach',
                        labels={'eu_total_reach': 'EU Total Reach', 'count': 'Number of Pages'})
    fig2.update_traces(marker_color='darkgreen', marker_line_color='black')
    fig2.update_layout(bargap=0.1, bargroupgap=0.05)
    
    # number of ads per page - very skewed
    nr_ads_per_page = data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")
    fig3 = px.histogram(nr_ads_per_page, x='nr_ads', title='Distribution of Ads per Page',
                        labels={'nr_ads': 'Number of Ads', 'count': 'Number of Pages'})
    fig3.update_traces(marker_color='gold', marker_line_color='black')
    fig3.update_layout(bargap=0.1, bargroupgap=0.05)
    
    # top 20 pages with most ads
    nr_ads_per_page_sorted = nr_ads_per_page.sort_values(by="nr_ads", ascending=False).head(20) 

    fig4 = px.bar(nr_ads_per_page_sorted, x='page_name', y='nr_ads', 
                title='Top 20 pages by number of ads',
                labels={'nr_ads': 'Number of Ads', 'page_name': 'Page Name'})
    fig4.update_xaxes(tickangle=45, tickmode='linear')
    fig3.update_traces(marker_color='darkorchid', marker_line_color='black')
    fig4.update_layout(xaxis=dict(categoryorder='total descending'))

    # total reach per page - very skewed 
    reach_by_page = data.groupby(["page_id", "page_name"])["eu_total_reach"].sum().reset_index(name="eu_total_reach")

    fig5 = px.histogram(reach_by_page, x='eu_total_reach', 
                    title='Distribution of EU total reach per Page',
                    labels={'eu_total_reach': 'Total EU reach', 'page_name': 'Page Name'})
    fig5.update_traces(marker_color='darkmagenta', marker_line_color='black')
    fig5.update_layout(bargap=0.1, bargroupgap=0.05)

    # top 20 pages with highest total reach
    reach_by_page_sorted = reach_by_page.sort_values(by="eu_total_reach", ascending=False).head(20)

    fig6 = px.bar(reach_by_page_sorted, x='page_name', y='eu_total_reach', 
                title='Top 20 pages by EU total reach',
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

    # relationship between campaign duration and total reach
    fig8 = px.scatter(data, x='campaign_duration', y='eu_total_reach', 
                 title='Campaign Duration vs. EU Total Reach',
                 labels={'campaign_duration': 'Campaign Duration (Days)', 'eu_total_reach': 'EU Total Reach'})


    """
    # total reach distribution (overall) - VERY skewed
    plt.figure(figsize=(10, 6))
    plt.hist(data['eu_total_reach'], bins=20, color='skyblue', edgecolor='black') 
    plt.title('Distribution of EU total reach ')
    plt.xlabel('EU total reach')
    plt.ylabel('Number of Pages')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # unique page_ids
    #unique_pages = data["page_id"].nunique()
    #print(f"Number of unique page_ids: {unique_pages}.")

    # nr of ads per page - very skewed
    nr_ads_per_page = data.groupby(["page_id", "page_name"])["id"].count().reset_index(name="nr_ads")

    plt.figure(figsize=(10, 6))
    plt.hist(nr_ads_per_page['nr_ads'], bins=20, color='skyblue', edgecolor='black') 
    plt.title('Distribution of Ads per Page')
    plt.xlabel('Number of Ads')
    plt.ylabel('Number of Pages')
    plt.grid(axis='y', alpha=0.75)
    plt.show() 

    # top 20 pages with most ads
    nr_ads_per_page_sorted = nr_ads_per_page.sort_values(by="nr_ads", ascending=False).head(20) 

    plt.figure(figsize=(10, 8))
    plt.bar(nr_ads_per_page_sorted['page_name'], nr_ads_per_page_sorted['nr_ads'], color='skyblue')
    plt.xlabel('Page Name')
    plt.ylabel('Number of Ads')
    plt.xticks(rotation=45, ha="right")
    plt.title('Top 20 pages by number of ads')
    plt.tight_layout() 
    plt.show()

    # total reach per page - very skewed 
    reach_by_page = data.groupby(["page_id", "page_name"])["eu_total_reach"].sum().reset_index(name="eu_total_reach")

    plt.figure(figsize=(10, 6))
    plt.hist(reach_by_page['eu_total_reach'], bins=20, color='skyblue', edgecolor='black') 
    plt.title('Distribution of EU total reach per Page')
    plt.xlabel('Total EU reach')
    plt.ylabel('Number of Pages')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # top 20 pages with highest total reach
    reach_by_page_sorted = reach_by_page.sort_values(by="eu_total_reach", ascending=False).head(20) 

    plt.figure(figsize=(10, 8))
    plt.bar(reach_by_page_sorted['page_name'], reach_by_page_sorted['eu_total_reach'], color='skyblue')
    plt.xlabel('Page Name')
    plt.ylabel('EU total reach')
    plt.xticks(rotation=45, ha="right")
    plt.title('Top 20 pages by EU total reach (for all ads)')
    plt.tight_layout() 
    plt.show()

    # ad campain duration distribution
    data["campaign_duration"] = (data["ad_delivery_stop_time"] - data["ad_delivery_start_time"]).dt.days

    plt.figure(figsize=(10, 6))
    plt.hist(data["campaign_duration"], bins=20, color='skyblue', edgecolor='black') 
    plt.title('Distribution of ad campaign duration')
    plt.xlabel('Duration of the ad campaign (days)')
    plt.ylabel('Number of ads')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    campaign_duration_by_page = data.groupby(["page_id", "page_name"])["campaign_duration"].mean().reset_index(name="avg_campaign_duration")

    campaign_duration_by_page_sorted = campaign_duration_by_page.sort_values(by="avg_campaign_duration", ascending=False).head(20) 

    # relationship between campaign duration and total reach
    plt.figure(figsize=(10, 6))
    plt.scatter(data['campaign_duration'], data['eu_total_reach'], alpha=0.5)
    plt.title('Campaign Duration vs. EU Total Reach')
    plt.xlabel('Campaign Duration (Days)')
    plt.ylabel('EU Total Reach')
    plt.show()

    reach_data = data[DEMOGRAPHIC_COLS]
    reach_data.plot(kind='bar', stacked=True, figsize=(12, 7))
    plt.title('Reach by Age and Gender')
    plt.xlabel('Age Ranges')
    plt.ylabel('Reach')
    plt.show()

    data["languages"].nunique()
    """
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

#project_name = "testmeet"
#run_analysis(project_name)


""" 
# IMAGE CAPTIONING

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_captioning = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model_answering = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

questions = ["Does the ad contain food?", "Is the ad targeted at children?", "Does the ad contain toys/cartoons?",
            "Is there a special offer or promotion mentioned?", "Are there any logos or brand names visible?",
            "What meal of the day is likely being shown?", "What food items are shown in the ad?",
            "What mood does the ad suggest?", "Does the ad contain people?"]

folder = f"output/{project_name}/ads_images"
imgs = os.listdir(folder)
raw_image = Image.open(os.path.join(folder, imgs[2])).convert('RGB')

question = "what objects do you see in this picture?"
inputs_cpt = processor(raw_image, return_tensors="pt")
inputs_quest = processor(raw_image, question, return_tensors="pt")

out_cpt = model_captioning.generate(**inputs_cpt, max_length = 30)
print(processor.decode(out_cpt[0], skip_special_tokens=True))

out_quest = model_answering.generate(**inputs_quest, max_length = 20)
print(processor.decode(out_quest[0], skip_special_tokens=True))

 """

# IMAGE PROCESSING:

"""
# extract 3 dominant colors of the ad image
def extract_dominant_color(image, num_colors = 3):
    # Resize image to speed up processing
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # Reshape the image to be a list of pixels
    reshaped_image = resized_image.reshape((resized_image.shape[0] * resized_image.shape[1], 3))

    # Find and display the most common colors
    clf = KMeans(n_clusters=num_colors, n_init=10)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in ordered_colors]
    rgb_colors = [tuple(c) for c in ordered_colors]

    # Calculate percentages
    total_pixels = len(reshaped_image)
    color_percentages = [(counts[i] / total_pixels) * 100 for i in counts.keys()]

    #plt.figure(figsize=(8, 6))
    #plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    #plt.show()

    return hex_colors, color_percentages

# assess image quality - resolution, brightness, sharpness and contrast
def assess_image_quality(image_path):
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
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract dominant colors
    dominant_colors, percentages = extract_dominant_color(image, show_chart=True)
    for col, percentage in zip(dominant_colors, percentages):
        print(f"Color: {col}, Percentage: {percentage:.2f}%")

    resolution, brightness, contrast, sharpness = assess_image_quality(image_path)
    print(f"Resolution: {resolution} pixels, Brightness: {brightness}, Contrast: {contrast}, Sharpness: {sharpness}")

    img = Image.open(image_path)
    img = color.rgb2gray(img)

    # edge detection
    # higher the sigma smoother the image (focus on the outside shape)
    canny_edges = canny(img, sigma=1.8)

    # corners - meh
    measure_image = corner_harris(img)
    coords = corner_peaks(measure_image, min_distance=32, threshold_rel=0.02)
    ncorners = len(coords)
    print("With a min_distance set to 32, we detect a total", len(coords), "corners in the image.")

    numbers = pattern.findall(image_path)
    ad_id = '_'.join(numbers) # extract ad id

    dict = {'ad_id': ad_id, 'dom_colors': dominant_colors, 'dom_colors_prop': percentages, 'resolution': resolution,
            'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness, 'ncorners': ncorners}
    return dict
    

# TODO: add a for loop to analyse a set of images

images_path = f"output\\{project_name}\\ads_images"
images = os.listdir(images_path)
os.path.exists(data_path)
row_list2 = []
#row = analyse_image(image_path)
#row_list2.append(row)

img_results = pd.DataFrame(row_list2)
img_results.to_excel(f'data\\img_features.xlsx', index=False)
"""