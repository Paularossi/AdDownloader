Analysis Module
==============

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

start_text_analysis Function
----------------------------

.. autofunction:: start_text_analysis

   Example::
   
        >>> freq_dist, word_cl, textblb_sent, nltk_sent, lda_model, coherence = start_text_analysis(data)
        >>> # for output see all examples from above

get_graphs Function
-------------------

.. autofunction:: get_graphs

   Example::
   
        >>> fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = get_graphs(data)
        >>> fig1.show() # will open a webpage with the graph, which can also be saved locally
