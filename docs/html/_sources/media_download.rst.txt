Media Download Module
=====================

.. module:: AdDownloader.media_download
   :synopsis: Provides functionality for media content download using Selenium.

download_media Function
-----------------------

.. autofunction:: download_media

    Example::

        >>> driver.get(data['ad_snapshot_url'][0])
        >>> img_element = driver.find_element(By.XPATH, img_xpath)
        >>> media_url = img_element.get_attribute('src')
        >>> media_type = 'image'
        >>> download_media(media_url, media_type, str(data['id'][i]), folder_path_img)

accept_cookies Function
-----------------------

.. autofunction:: accept_cookies

    Example::

        >>> driver = webdriver.Chrome()
        >>> driver.get(data['ad_snapshot_url'][0]) # start from here to accept cookies
        >>> accept_cookies(driver)

start_media_download Function
-----------------------------

.. autofunction:: start_media_download

    Example::

        >>> start_media_download(project_name = "test1", nr_ads = 20, data = data)