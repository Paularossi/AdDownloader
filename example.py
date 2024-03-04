from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
import pandas as pd

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
ads_api.add_parameters(countries = 'BE', start_date = "2023-06-01", end_date = '2023-09-01', search_terms = "pizza hut")

# check the parameters
ads_api.get_parameters()

# start the ad data download
data = ads_api.start_download()

# if you want to download media right away
start_media_download(project_name = "testtt", nr_ads = 50, data = data)

# if you want to download media from an earlier project
from AdDownloader.helpers import update_access_token
data_path = 'path/to/your/data.xlsx'
new_data = pd.read_excel(data_path)
new_data = update_access_token(data = new_data, new_access_token = access_token)

start_media_download(project_name = "test", nr_ads = 20, data = new_data)

# you can find all the output in the 'output/<your-project-name>' folder


#### Example with Political ads:
plt_ads_api = adlib_api.AdLibAPI(access_token, project_name = "test2")

plt_ads_api.add_parameters(countries = 'US', start_date = "2023-02-01", end_date = "2023-03-01", ad_type = "POLITICAL_AND_ISSUE_ADS",
                   ad_active_status = "ALL", estimated_audience_size_max = 10000, languages = 'es', search_terms = "Biden")

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
import matplotlib.pyplot as plt
data_path = "output/test1/ads_data/test1_processed_data.xlsx"
data = load_data(data_path)
data.head(20)

# create graphs with EDA
fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = get_graphs(data)
fig1.show() # will open a webpage with the graph, which can also be saved locally

# perform text analysis of the ad captions
freq_dist, word_cl, textblb_sent, nltk_sent, lda_model, coherence = start_text_analysis(data)
print(f"Most common 10 keywords: {freq_dist.most_common(10)}")

# show the word cloud
plt.imshow(word_cl, interpolation='bilinear')
plt.axis("off")
plt.show()

# check the sentiment
textblb_sent.head(20) # or textblb_sent

# print the top 3 topics and the coherence score
for idx, topic in lda_model.print_topics(num_words=5):
    print("Topic: {} \nWords: {}".format(idx + 1, topic))

print('Coherence Score:', coherence)