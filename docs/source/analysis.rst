Analysis Module
===============

.. module:: AdDownloader.analysis
   :synopsis: Provides different analysis functions for the AdDownloader.

   This module provides different analysis functions for the AdLibAPI object, such as text and image analysis, and visualizations.

load_data Function
------------------

.. autofunction:: load_data

   Example::
   
      >>> from AdDownloader.analysis import *
      >>> data_path = "output/<project_name>/ads_data/<project_name>_processed_data.xlsx"
      >>> data = load_data(data_path)

preprocess Function
-------------------

.. autofunction:: preprocess

   Example::
   
      >>> tokens = data["ad_creative_bodies"].apply(preprocess)
      >>> tokens.head(3)  
      0    person earli vote open soon georgia wait take ...
      1    2020 help turn year around find vote earli person
      2    person earli vote open soon georgia wait take ...

get_word_freq Function
----------------------

.. autofunction:: get_word_freq

   Example::
   
      >>> freq_dist = get_word_freq(tokens)
      >>> print(f"Most common 3 keywords: {freq_dist[0:3]}")
      Most common 3 keywords: [('vote', 3273), ('elect', 1155), ('earli', 1125)]

get_sentiment Function
----------------------

.. autofunction:: get_sentiment

   Example::
   
      >>> textblb_sent, nltk_sent = get_sentiment(data["ad_creative_bodies"])
      >>> nltk_sent.head(3)
      0     {'neg': 0.0, 'neu': 0.859, 'pos': 0.141, 'comp...
      1     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...
      2     {'neg': 0.098, 'neu': 0.644, 'pos': 0.258, 'co...
      >>> textblb_sent.head(3)
      0     0.125000
      1     0.112500
      2     0.142857

get_topics Function
-------------------

.. autofunction:: get_topics

   Example::
   
      >>> lda_model, coherence_lda, perplexity, log_likelihood, topics_df = get_topics(tokens, nr_topics=5)
      Number of unique tokens: 435
      Number of documents: 2000
      Finished topic modeling for 5 topics.
      Coherence: 0.71; Perplexity: 51.78; Log-Likelihood: -104762.56
      
      Topic 0: ['vote', 'elect', 'paramount', 'give', 'need', 'win', 'novemb', '3rd']
      Topic 1: ['vote', 'earli', 'find', 'year', 'person', 'click', 'easi', 'wait']
      ...
      >>> topics_df.head(3)
         dom_topic  perc_contr                                     topic_keywords
      0          1      0.6444  vote, earli, find, year, person, click, easi, ...
      1          1      0.6567  vote, earli, find, year, person, click, easi, ...
      2          4      0.9138  ballot, return, vote, today, click, home, demo...

get_topic_per_caption Function
------------------------------

.. autofunction:: get_topic_per_caption

   Example::
   
      >>> vectorizer = CountVectorizer(stop_words = stop_words, max_features = 1000, min_df = 5, max_df = 0.95)
      >>> vect_text = vectorizer.fit_transform(tokens) # assuming the tokens are already processed captions
      >>> tf_feature_names = vectorizer.get_feature_names_out()

      >>> lda_model = LatentDirichletAllocation(n_components=5, learning_method='online', random_state=0, max_iter=10, learning_decay=0.7, learning_offset=10).fit(vect_text) 
      >>> topics_df = get_topic_per_caption(lda_model, vect_text, tf_feature_names)
      >>> topics_df.head(3)
         dom_topic  perc_contr                                     topic_keywords
      0          1      0.6444  vote, earli, find, year, person, click, easi, ...
      1          1      0.6567  vote, earli, find, year, person, click, easi, ...
      2          4      0.9138  ballot, return, vote, today, click, home, demo...

start_text_analysis Function
----------------------------

.. autofunction:: start_text_analysis

   Example::
   
      >>> # without topic modeling
      >>> tokens, freq_dist, textblb_sent, nltk_sent = start_text_analysis(data)
      >>> # with topic modeling
      >>> tokens, freq_dist, textblb_sent, nltk_sent, lda_model, coherence_lda, perplexity, log_likelihood, topics_df = start_text_analysis(data)
      >>> # for output see all examples from above

transform_data_by_age Function
------------------------------

.. autofunction:: transform_data_by_age

   Example::
   
      >>> import pandas as pd
      >>> data_path = "output/<project_name>/ads_data/<project_name>_processed_data.xlsx"
      >>> data = pd.read_excel(data_path)
      >>> data_by_age = transform_data_by_age(data)
      >>> data_by_age.head(3)
            Reach Age Range
      0     7.0       18-24
      1     0.0       18-24
      3     23.0       65+

transform_data_by_gender Function
---------------------------------

.. autofunction:: transform_data_by_gender

   Example::
   
      >>> # assuming data was already loaded
      >>> data_by_gender = transform_data_by_gender(data)
      >>> data_by_gender.head(3)
            Reach Gender
      0      NaN  female
      1     68.0  female
      2    243.0    male

get_graphs Function
-------------------

.. autofunction:: get_graphs

   Example::
   
      >>> fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = get_graphs(data)
      >>> fig1.show() # will open a webpage with the graph, which can also be saved locally

show_topics_top_pages Function
------------------------------

.. autofunction:: show_topics_top_pages

   Example::
   
      >>> # using the output from `get_topics(tokens)`
      >>> fig = show_topics_top_pages(topics_df, data)
      >>> fig.show()

blip_call Function
------------------

.. autofunction:: blip_call

   Example::
   
      >>> images_path = "output/<project_name>/ads_images"
      >>> img_caption = blip_call(images_path, nr_images=20) # captioning
      >>> img_caption.head(3)
                     ad_id                                        img_caption
      0     689539479274809            a group of people eating pizza together
      1     352527490742823  a couple of people sitting at a table eating p...
      2     891711935895560              a man and woman eating pizza together
      >>> img_content = blip_call(images_path, task="visual_question_answering", nr_images=20, questions="Are there people in this ad?")
      >>> img_content.head(5)
                  ad_id Are there people in this ad?
      0   723805182773873                          yes
      1   871823271403675                           no
      2  6398713840181656                          yes

extract_dominant_colors Function
--------------------------------

.. autofunction:: extract_dominant_colors

   Example::
   
      >>> image_files = [f for f in os.listdir(images_path) if f.endswith(('jpg', 'png', 'jpeg'))]
      >>> dominant_colors, percentages = extract_dominant_colors(os.path.join(images_path, image_files[2]))
      >>> for col, percentage in zip(dominant_colors, percentages):
      ...     print(f"Color: {col}, Percentage: {percentage:.2f}%")
      ...
      Color: #3a2f28, Percentage: 41.99%
      Color: #dfcbac, Percentage: 32.76%
      Color: #817875, Percentage: 25.24%

assess_image_quality Function
-----------------------------

.. autofunction:: assess_image_quality

   Example::
   
      >>> resolution, brightness, contrast, sharpness = assess_image_quality(os.path.join(images_path, image_files[2]))
      >>> print(f"Resolution: {resolution} pixels, Brightness: {brightness}, Contrast: {contrast}, Sharpness: {sharpness}")
      Resolution: 188400 pixels, Brightness: 142.5308, Contrast: 71.3726, Sharpness: 3691.4007

analyse_image Function
----------------------

.. autofunction:: analyse_image

   Example::
   
      >>> analysis_result = analyse_image(os.path.join(images_path, image_files[2]))
      >>> print(analysis_result)
      {'ad_id': '1043287820216670', 'resolution': 188400, 'brightness': 142.53080148619958, 'contrast': 71.3726801705792, 'sharpness': 3691.40007606529, 
      'ncorners': 17, 'dom_color_1': '#817875', 'dom_color_1_prop': 41.943359375, 'dom_color_2': '#dfcbab', 'dom_color_2_prop': 32.8369140625, 'dom_color_3': '#3a2f28', 'dom_color_3_prop': 25.2197265625}

analyse_image_folder Function
-----------------------------

.. autofunction:: analyse_image_folder

   Example::
   
      >>> df = analyse_image_folder(images_path, nr_images=20)
      >>> df.head(3)
                  ad_id  resolution  brightness   contrast    sharpness  ncorners dom_color_1  dom_color_1_prop dom_color_2  dom_color_2_prop dom_color_3  dom_color_3_prop
      0  1039719343827470      187800  172.399936  60.601719  1585.668739        21     #ced2ce         55.395508     #a48b7d         28.369141     #464347         16.235352
      1  1043131113478341      187800  108.217066  73.420019   903.498253        18     #1b1c17         45.996094     #96603d         33.593750     #dcbea0         20.410156
      2  1043287820216670      188400  142.530801  71.372680  3691.400076        17     #3a2f28         41.992188     #817875         32.763672     #dfcbac         25.244141
