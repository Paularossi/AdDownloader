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
ads_api.add_parameters(countries = 'BE', start_date = "2023-09-01", end_date = "2023-09-02", search_terms = "burger")

# check the parameters
ads_api.get_parameters()

# start the ad data download
data = ads_api.start_download()

# if you want to download media right away
start_media_download(project_name = "test1", nr_ads = 20, data = data)

# if you want to download media from an earlier project
data_path = 'path/to/your/data.xlsx'
new_data = pd.read_excel(data_path)

start_media_download(project_name = "test11", nr_ads = 20, data = new_data)

# you can find all the output in the 'output/your-project-name' folder

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


