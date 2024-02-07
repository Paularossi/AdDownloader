.. AdDownloader documentation master file, created by
   sphinx-quickstart on Fri Jan 19 15:58:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AdDownloader's documentation!
========================================

**AdDownloader** is a Python library and Command-line Interface (CLI) that directly
calls the Meta Ad Library API given a valid Meta developer access token. It downloads general ad data
based on user-defined parameters by creating an :class:`~AdDownloader.adlib_api.AdLibAPI` object, then downloads 
the media content of the ads using :func:`~AdDownloader.media_download.start_media_download`. Additionally, for a more user-friendly interface, the entire ad download
process can be run as a CLI through the :func:`~AdDownloader.cli.run_analysis` call, offering a *simple* and *intuitive* API.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   adlib_api
   helpers
   media_download
   cli


Below, you can find two examples of how to (1) manually download ad data and media content using the AdLibAPI class and the media_download module
and (2) run the automated CLI to download ad data and media content.

**Example 1.1:** *Manual Download for ALL ads*.

.. code-block:: python

   from AdDownloader import adlib_api
   from AdDownloader.media_download import start_media_download
   from AdDownloader.cli import run_analysis
   import pandas as pd

   access_token = input() # your fb-access-token-here
   # initialize the AdLibAPI object
   ads_api = adlib_api.AdLibAPI(access_token, project_name = "test1")

   # either search_terms OR search_pages_ids
   ads_api.add_parameters(countries = 'BE', start_date = "2023-09-01", end_date = "2023-09-02", search_terms = "pizza")

   # check the parameters
   ads_api.get_parameters()

   # start the ad data download
   data = ads_api.start_download()

   # if you want to download media right away
   start_media_download(project_name = "test1", nr_ads = 20, data = data)

   # if you want to download media from an earlier project
   data_path = 'path/to/your/data.xlsx'
   new_data = pd.read_excel(data_path)

   start_media_download(project_name = "test2", nr_ads = 20, data = new_data)

   # you can find all the output in the 'output/your-project-name' folder

**Example 1.2:** *Manual Download for POLITICAL_AND_ISSUE_ADS*.

.. code-block:: python

   # the same can be done for POLITICAL_AND_ISSUE_ADS:
   plt_ads_api = adlib_api.AdLibAPI(access_token, project_name = "test2")

   plt_ads_api.add_parameters(countries = 'US', start_date = "2023-02-01", end_date = "2023-03-01", ad_type = "POLITICAL_AND_ISSUE_ADS",
                   ad_active_status = "ALL", estimated_audience_size_max = 10000, languages = 'es', search_terms = "Biden")
                   
   # check the parameters
   plt_ads_api.get_parameters()

   # start the ad data download
   plt_data = plt_ads_api.start_download()

   # start the media download
   start_media_download(project_name = "test2", nr_ads = 20, data = plt_data)


**Example 2:** *Automated CLI*.

.. code-block:: python

   from AdDownloader.cli import run_analysis
   run_analysis()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`