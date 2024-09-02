from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download

# ============================================================
# 
# Download ad data and media content using the AdLibAPI class.
#
# ============================================================
access_token = input() # your fb-access-token-here
ads_api = adlib_api.AdLibAPI(access_token, project_name = "test1")

# add parameters for your search
# for available parameters, visit https://developers.facebook.com/docs/graph-api/reference/ads_archive/

# either search_terms OR search_pages_ids
ads_api.add_parameters(ad_reached_countries = 'BE', ad_delivery_date_min = "2024-08-01", ad_delivery_date_max = "2024-08-05",
                       search_terms = "pizza", ad_type = 'ALL')

# check the parameters
ads_api.get_parameters()

# start the ad data download
data = ads_api.start_download()

# if you want to download media right away
start_media_download(project_name = "test1", nr_ads = 20, data = data)

# if you want to download media from an earlier project
from AdDownloader.helpers import update_access_token
import pandas as pd

data_path = 'output/<project_name>/ads_data/<project_name>_processed_data.xlsx'
new_data = pd.read_excel(data_path)
new_data = update_access_token(data = new_data, new_access_token = access_token)

start_media_download(project_name = "test1", nr_ads = 30, data = new_data)

# you can find all the output in the 'output/<your-project-name>' folder


#### Example with Political ads:
plt_ads_api = adlib_api.AdLibAPI(access_token, project_name = "test2")

plt_ads_api.add_parameters(ad_reached_countries = 'US', ad_delivery_date_min = "2020-11-01", ad_delivery_date_max = "2020-11-03", 
                           ad_type = "POLITICAL_AND_ISSUE_ADS", search_page_ids = "us_parties.xlsx")

plt_ads_api.clear_parameters()

# check the parameters
plt_ads_api.get_parameters()

# start the ad data download
plt_data = plt_ads_api.start_download()

# start the media download
start_media_download(project_name = "test2", nr_ads = 20, data = plt_data)



# ===========================================================
# 
# Download ad data and media content using the automated CLI.
#
# ===========================================================

from AdDownloader.cli import run_analysis
run_analysis()



# ======================================================
# 
# Create a dashboard with various graphs and statistics.
# Access http://127.0.0.1:8050/ once Dash is running.
#
# ======================================================
from AdDownloader.start_app import start_gui # takes some time to load...
start_gui()




# ==================================================
# 
# Analyze data locally without creating a dashboard.
#
# ==================================================
from AdDownloader.analysis import *

data_path = "output/<project_name>/ads_data/<project_name>_processed_data.xlsx" # need to use the processed data
data = load_data(data_path)
data.head(20)

##### EDA GRAPHS
fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = get_graphs(data)
fig1.show() # will open a webpage with the graph, which can also be saved locally

##### TEXT AND TOPIC ANALYSIS
tokens, freq_dist, textblb_sent, nltk_sent = start_text_analysis(data)
print(f"Most common 10 keywords: {freq_dist[0:10]}")

# check the sentiment
nltk_sent.head(20) # or textblb_sent

lda_model, topics, coherence, perplexity, log_likelihood, avg_similarity, topics_df = get_topics(tokens, nr_topics=10) # change nr of topics
topics_df.head(20)

fig = show_topics_top_pages(topics_df, data)
fig.show()


##### IMAGE ANALYSIS
images_path = f"output/<project_name>/ads_images"
image_files = [f for f in os.listdir(images_path) if f.endswith(('jpg', 'png', 'jpeg'))]

# for an individual image:
dominant_colors, percentages = extract_dominant_colors(os.path.join(images_path, image_files[2]))
for col, percentage in zip(dominant_colors, percentages):
    print(f"Color: {col}, Percentage: {percentage:.2f}%")
resolution, brightness, contrast, sharpness = assess_image_quality(os.path.join(images_path, image_files[2]))
print(f"Resolution: {resolution} pixels, Brightness: {brightness}, Contrast: {contrast}, Sharpness: {sharpness}")
# OR
analysis_result = analyse_image(os.path.join(images_path, image_files[2]))
print(analysis_result)

# for a defined number of images
df = analyse_image_folder(images_path, nr_images = 100)
df.head(5)


##### CAPTIONING AND QUESTION ANSWERING WITH BLIP
img_captions = blip_call(images_path, nr_images=20)
img_captions.head(5)

img_content = blip_call(images_path, task="visual_question_answering", nr_images=20, questions="Are there people in this ad?")
img_content.head(5)

# then preprocess the captions and analyze the text
tokens, freq_dist, textblb_sent, nltk_sent = start_text_analysis(img_captions, column_name = "img_caption")

lda_model, topics, coherence, perplexity, log_likelihood, avg_similarity, topics_df = get_topics(tokens, nr_topics = 2)
topics_df.head(5)



# ==================================================
# 
# Download branded content from the Meta Ad Library.
#
# ==================================================
import requests
url = "https://graph.facebook.com/v20.0/branded_content_search"
access_token = input()
# fields from https://developers.facebook.com/docs/graph-api/reference/branded-content-search/
params = {
    "fields": "type,creation_date,creator,partners,url",
    "ig_username": "mcdonaldsnl", # either ig or fb
    "page_url": None,
    "creation_date_min": "2023-09-01",
    "creation_date_max": "2023-12-12",
    "access_token": access_token
}
response = requests.get(url, params = params)
data = response.json()
data_list = data.get('data', [])
df = pd.DataFrame(data_list)
df.to_excel(f'data/branded.xlsx', index=False)