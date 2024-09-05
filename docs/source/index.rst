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
Lastly, AdDownloader provides additional analysis functionalities, by performing text and image analysis of the ad creative contents,
and creating useful visualizations inside an Analytics dashboard. For a detailed tutorial on how to use AdDownloader, you can check out our 
step-by-step video on YouTube `here <https://youtu.be/WK6eS2dXTbg?si=lR_cMcbyNkdlK915>`_.

If you use AdDownloader in your research or project, please refer to and cite our paper (under review). Read the pre-print `here <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4866619>`_.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   adlib_api
   helpers
   media_download
   cli
   analysis


Below, you can find two examples of how to (1) manually download ad data and media content using the AdLibAPI class and the media_download module
and (2) run the automated CLI to download ad data and media content.

**Example 1.1:** *Manual Download for ALL ads*.

.. code-block:: python

   from AdDownloader import adlib_api
   from AdDownloader.media_download import start_media_download
   import pandas as pd

   access_token = input() # your fb-access-token-here
   # initialize the AdLibAPI object
   ads_api = adlib_api.AdLibAPI(access_token, project_name = "test1")

   # either search_terms OR search_pages_ids
   ads_api.add_parameters(ad_reached_countries = 'BE', ad_delivery_date_min = "2023-09-01", ad_delivery_date_max = "2023-09-02", search_terms = "pizza")

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

   plt_ads_api.add_parameters(ad_reached_countries = 'US', ad_delivery_date_min = "2023-02-01", ad_delivery_date_max = "2023-03-01", 
                              ad_type = "POLITICAL_AND_ISSUE_ADS", ad_active_status = "ALL", estimated_audience_size_max = 10000, languages = 'es', search_terms = "Biden")
                   
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


AdDownloader Analytics
======================

The output saved by AdDownloader, which includes: Excel files with original and processed ads data, and ad images and videos, can 
further be analysed using the analytics module provided by AdDownloader. This can be achieved in two ways: (1) Run a Dash dashboard
with various EDA and statistics, or (2) Analyze your data locally.

**Example 1:** *Run Analytics Dashboard*.

.. code-block:: python

   from AdDownloader.start_app import start_gui # takes some time to load...
   start_gui()

This function will open an html page at http://127.0.0.1:8050/ once Dash is running.


**Example 2.1:** *Analyze text data manually*.

.. code-block:: python

   from AdDownloader.analysis import *
   import matplotlib.pyplot as plt
   data_path = "output/test1/ads_data/test1_processed_data.xlsx"
   data = load_data(data_path)
   data.head(20)

   # create graphs with EDA
   fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = get_graphs(data)
   fig1.show() # will open a webpage with the graph, which can also be saved locally

   # perform text analysis of the ad captions
   tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(data['ad_creative_bodies'])
   print(f"Most common 10 keywords: {freq_dist.most_common(10)}")

   # show the word cloud
   plt.imshow(word_cl, interpolation='bilinear')
   plt.axis("off")
   plt.show()

   # check the sentiment
   textblb_sent.head(20) # or textblb_sent

   # get the topics
   lda_model, coherence, topics_df = get_topics(tokens, num_topics = 5)

   # print the top 5 words for each topic and the coherence score
   for idx, topic in lda_model.print_topics(num_words=5):
      print("Topic: {} \nWords: {}".format(idx + 1, topic))

   print('Coherence Score:', coherence)

**Example 2.2:** *Analyze media data manually*.

.. code-block:: python

   images_path = f"output/test2/ads_images"
   image_files = [f for f in os.listdir(images_path) if f.endswith(('jpg', 'png', 'jpeg'))]

   ### for an individual image
   # dominant colors:
   dominant_colors, percentages = extract_dominant_colors(os.path.join(images_path, image_files[2]))
   for col, percentage in zip(dominant_colors, percentages):
      print(f"Color: {col}, Percentage: {percentage:.2f}%")
   
   # image quality
   resolution, brightness, contrast, sharpness = assess_image_quality(os.path.join(images_path, image_files[2]))
   print(f"Resolution: {resolution} pixels, Brightness: {brightness}, Contrast: {contrast}, Sharpness: {sharpness}")

   # OR - all features together
   analysis_result = analyse_image(os.path.join(images_path, image_files[2]))
   print(analysis_result)

   ### for a folder with images
   df = analyse_image_folder(images_path)
   df.head(5)


   ### CAPTIONING AND QUESTION ANSWERING WITH BLIP
   img_caption = blip_call(images_path, nr_images=20)
   img_caption.head(5)

   img_content = blip_call(images_path, task="visual_question_answering", nr_images=20, questions="Are there people in this ad?")
   img_content.head(5)



Image Download (Selenium) Setup
===============================
On some machines it might happen that a potential binary version mismatch might occur between the installed Chrome version and the required ChromeDriver. 
We recommend that users first try running the image download functionality of AdDownloader as it is. If an error occurs related to a version 
mismatch, we advise downloading the appropriate version of ChromeDriver directly from the official 
`ChromeDriver website <https://developer.chrome.com/docs/chromedriver/downloads>`_ and ensuring that it matches the version of Chrome installed on 
their machine. Once downloaded, placing the ChromeDriver executable in a directory included in the system’s PATH should help avoid version mismatches and related errors.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`