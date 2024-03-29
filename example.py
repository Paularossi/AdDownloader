from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
import pandas as pd

# ============================================================
# 
# Download ad data and media content using the AdLibAPI class.
#
# ============================================================

access_token = input() # your fb-access-token-here
ads_api = adlib_api.AdLibAPI(access_token, project_name = "testbig3")

# add parameters for your search
# for available parameters, visit https://developers.facebook.com/docs/graph-api/reference/ads_archive/

# either search_terms OR search_pages_ids
ads_api.add_parameters(countries = 'BE, NL', start_date = "2024-01-01", end_date = "2024-03-10", search_terms = "pizza", languages = 'en')

# check the parameters
ads_api.get_parameters()

# start the ad data download
data = ads_api.start_download()

# if you want to download media right away
start_media_download(project_name = "testbig", nr_ads = 50, data = data)

# if you want to download media from an earlier project
from AdDownloader.helpers import update_access_token
data_path = 'path/to/your/data.xlsx'
new_data = pd.read_excel(data_path)
new_data = update_access_token(data = new_data, new_access_token = access_token)

start_media_download(project_name = "test", nr_ads = 20, data = new_data)

# you can find all the output in the 'output/<your-project-name>' folder


#### Example with Political ads:
plt_ads_api = adlib_api.AdLibAPI(access_token, project_name = "testbig2")

plt_ads_api.add_parameters(countries = 'US', start_date = "2023-12-01", end_date = "2024-03-10", ad_type = "POLITICAL_AND_ISSUE_ADS",
                   ad_active_status = "ALL", estimated_audience_size_max = 10000, languages = 'es', search_terms = "Biden")

plt_ads_api.clear_parameters()

# check the parameters
plt_ads_api.get_parameters()

# start the ad data download
plt_data = plt_ads_api.start_download()

# start the media download
start_media_download(project_name = "test2", nr_ads = 20, data = plt_data)



# branded content
import requests
url = "https://graph.facebook.com/v19.0/branded_content_search"
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
import matplotlib.pyplot as plt

##### TEXT AND TOPIC ANALYSIS
data_path = "output/testbig/ads_data/testbig_processed_data.xlsx"
data = load_data(data_path)
data.head(20)


# create graphs with EDA
fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = get_graphs(data)
fig1.show() # will open a webpage with the graph, which can also be saved locally

# perform text analysis of the ad captions
data = data.dropna(subset = ["ad_creative_bodies"])
tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(data['ad_creative_bodies'])
print(f"Most common 10 keywords: {freq_dist.most_common(10)}")

# show the word cloud
plt.imshow(word_cl, interpolation='bilinear')
plt.axis("off")
plt.show()

# check the sentiment
textblb_sent.head(20) # or textblb_sent

# topics - optional
lda_model, coherence, topics_df = get_topics(tokens)
topics_df.head(20)

# print the topics and the coherence score
for idx, topic in lda_model.print_topics(num_words=8):
    print("Topic: {} \nWords: {}".format(idx, topic))

print('Coherence Score:', coherence)


##### IMAGE ANALYSIS
images_path = f"output/test77/ads_images"
image_files = [f for f in os.listdir(images_path) if f.endswith(('jpg', 'png', 'jpeg'))]

# for an individual image:
dominant_colors, percentages = extract_dominant_colors(os.path.join(images_path, image_files[2]))
resolution, brightness, contrast, sharpness = assess_image_quality(os.path.join(images_path, image_files[2]))
# OR
analysis_result = analyse_image(os.path.join(images_path, image_files[2]))

# for a defined number of images
df = analyse_image_folder(images_path, project_name="test77", nr_images=50)


### CAPTIONING WITH BLIP

img_caption = blip_call(images_path, nr_images=50)
img_content = blip_call(images_path, task="visual_question_answering", nr_images=20, question="What items do you see in this image?")

# then preprocess the captions and analyze the text
tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(img_caption["img_caption"])

lda_model, coherence, topics_df = get_topics(tokens)
topics_df.head(20)




# distribution of brightness
fig_brightness = px.histogram(df, x='brightness', nbins=50, title='Distribution of Image Brightness')
fig_brightness.show()

# top 5 dominant colors
color_long_df = df.melt(value_vars=['dom_color_1', 'dom_color_2', 'dom_color_3'], 
                        var_name='Color_Type', value_name='Color')
color_counts = color_long_df['Color'].value_counts().reset_index()
top_colors = color_counts.head(5)

fig_colors = px.bar(top_colors, x='Color', y='count', title=f'Top 5 Dominant Colors Across All Images',
             color='Color', text='count', color_discrete_map={color: color for color in top_colors['Color']})
fig_colors.update_traces(textposition='inside', marker_line_color='black')
fig_colors.show()


# image quality
df_quality = df.melt(value_vars=['brightness', 'contrast', 'sharpness'], 
                  var_name='Metric', value_name='Value')
fig_qual = px.box(df_quality, x='Metric', y='Value', color='Metric',
             title='Distribution of Image Quality Metrics')
fig_qual.update_layout(xaxis_title="Quality Metric", yaxis_title="Value", legend_title="Metric")
fig_qual.show()

# Brightness vs. Sharpness
fig_bs = px.scatter(df, x='brightness', y='sharpness', title='Brightness vs. Sharpness')
fig_bs.show()

# Contrast vs. Sharpness
fig_cs = px.scatter(df, x='contrast', y='sharpness', title='Contrast vs. Sharpness')
fig_cs.show()

# Brightness vs. Contrast
fig_bc = px.scatter(df, x='brightness', y='contrast', title='Brightness vs. Contrast')
fig_bc.show()

