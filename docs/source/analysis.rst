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
   
      >>> captions = data["ad_creative_bodies"].dropna()
      >>> tokens = captions.apply(preprocess)
      >>> tokens.head(3)  
      0    [samedi, malin, bien, pizza, médium, prix, seu...
      1    [profiter, lundi, gourmand, achat, cheezy, cru...
      2    [reprise, dur, dur, aussi, heureusement, pizza...

get_word_freq Function
----------------------

.. autofunction:: get_word_freq

   Example::
   
      >>> freq_dist, word_cl = get_word_freq(tokens)
      >>> print(f"Most common 10 keywords: {freq_dist.most_common(3)}")
      Most common 10 keywords: [('pizza', 149), ('hut', 81), ('chain', 24)]
      >>> # show the word cloud
      >>> plt.imshow(word_cl, interpolation='bilinear')
      >>> plt.axis("off")
      >>> plt.show()

get_sentiment Function
----------------------

.. autofunction:: get_sentiment

   Example::
   
      >>> textblb_sent, nltk_sent = get_sentiment(captions)
      >>> nltk_sent.head(3)
      0    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...
      1    {'neg': 0.0, 'neu': 0.931, 'pos': 0.069, 'comp...
      2    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...
      >>> textblb_sent.head(3)
      0    -0.247619
      1     0.004167
      2     0.000000

get_topics Function
-------------------

.. autofunction:: get_topics

   Example::
   
      >>> lda_model, coherence = get_topics(tokens)
      >>> for idx, topic in lda_model.print_topics(num_words=5):
      ...     print("Topic: {} \nWords: {}".format(idx + 1, topic))
      ...
      Topic: 1
      Words: 0.051*"pizza" + 0.041*"hut" + 0.024*"menu" + 0.022*"chez" + 0.020*"place"
      ...
      >>> print('Coherence Score:', coherence)
      Coherence Score: 0.4461575424013203

get_topic_per_caption Function
------------------------------

.. autofunction:: get_topic_per_caption

   Example::
   
      >>> # assuming the tokens are already processed captions
      >>> dictionary = corpora.Dictionary(tokens)
      >>> corpus = [dictionary.doc2bow(text) for text in tokens]
      >>> lda_model = LdaModel(corpus, id2word = dictionary, num_topics = 3, passes = 20, eval_every = None)
      >>> sent_topics_df = get_topic_per_caption(lda_model, corpus)
      >>> sent_topics_df.head(5)
            dom_topic  perc_contr                                     topic_keywords                                 ad_creative_bodies
      0          1      0.5464  would, could, one, father, like, eye, back, ge...  ["Ready to conquer your blood sugar? \nDiscove...
      1          1      0.6054  would, could, one, father, like, eye, back, ge...  ["👀 Watch now to conquer Type 2 Diabetes! Lowe...
      2          0      0.9734  alpha, victor, said, daisy, like, andrea, ceci...  ['Fast Food Chains: Worst to Best, Ranked (202...


start_text_analysis Function
----------------------------

.. autofunction:: start_text_analysis

   Example::
   
      >>> freq_dist, word_cl, textblb_sent, nltk_sent, lda_model, coherence = start_text_analysis(data, topics = True)
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
   
      >>> # using the output from `lda_model, coherence, topics_df = analysis.get_topics(tokens)`
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
