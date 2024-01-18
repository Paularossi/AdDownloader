from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download

access_token = input() # your fb-access-token-here
ads = adlib_api.AdLibAPI(access_token)

# add parameters for your search
# For available parameters, visit https://developers.facebook.com/docs/graph-api/reference/ads_archive/
# ads.add_parameters(ad_delivery_date_min = "2023-01-01", ad_delivery_date_max = "2023-01-05", ad_type = "POLITICAL_AND_ISSUE_ADS",
#                    ad_reached_countries = ['US'], ad_active_status = "ALL", impression_condition = 'HAS_IMPRESSIONS_LAST_90_DAYS',
#                    search_terms = "Biden")

# either search_terms OR search_pages_ids
ads.add_parameters(countries = 'BE', start_date = "2023-09-01", end_date = "2023-09-02", search_terms = "pizza", project_name = "test1")

# check the parameters
ads.get_parameters()

# start the download
data = ads.start_download()

# if you want to download media
start_media_download(project_name = "test1", nr_ads = 20, data = data)

# you can find all the output in the 'output/your-project-name' folder
