AdLibAPI Module
===============

.. automodule:: AdDownloader.adlib_api
   :members:
   :undoc-members:
   :show-inheritance:

AdLibAPI.__init__ Method
------------------------

.. automethod:: AdLibAPI.__init__

   Example::

      >>> from AdDownloader import adlib_api
      >>> access_token = input() # your fb-access-token-here
      >>> ads_api = adlib_api.AdLibAPI(access_token)

AdLibAPI.fetch_data Method
--------------------------

.. automethod:: AdLibAPI.fetch_data
   :no-index:

   Example::
      >>> url = "https://graph.facebook.com/v18.0/ads_archive"
      >>> params = ads_api.get_parameters()
      >>> api.fetch_data(url, params)


AdLibAPI.add_parameters Method
------------------------------

.. automethod:: AdLibAPI.add_parameters
   :no-index:

   Example::
      >>> ads_api.add_parameters(countries = 'NL', start_date = "2023-09-01", end_date = "2023-09-02", search_terms = "pizza", project_name = "test1")


AdLibAPI.start_download Method
------------------------------

.. automethod:: AdLibAPI.start_download
   :no-index:

   Example::
      >>> data = ads_api.start_download()
      >>> print(data.head(1))
            id ad_delivery_start_time ad_delivery_stop_time  ... unknown_45-54 unknown_55-64 unknown_65+
      0  11111             2023-06-30            2024-01-09  ...          21.0           5.0        11.0

AdLibAPI.get_parameters Method
------------------------------

.. automethod:: AdLibAPI.get_parameters
   :no-index:

   Example::
      >>> ads_api.get_parameters()
      {'fields': 'id, ad_delivery_start_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown', 'ad_reached_countries': 'BE', 'search_page_ids': None, 'search_terms': 'pizza', 'ad_delivery_date_min': '2023-09-01', 'ad_delivery_date_max': '2023-09-02', 'limit': '300', 'access_token': 'XX'}

AdLibAPI.get_fields Method
--------------------------

.. automethod:: AdLibAPI.get_fields
   :no-index:

   Example::
      >>> ads_api.get_fields()
      'id, ad_delivery_start_time, ad_delivery_stop_time, ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles, ad_snapshot_url, 
      page_id, page_name, target_ages, target_gender, target_locations, eu_total_reach, age_country_gender_reach_breakdown'
