"""This module provides the functionality of media content download of the AdDownloader."""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import requests
import os


def download_media(media_url, media_type, ad_id, media_folder):
    # once we got the url of the media, try to download it
    try:
        response = requests.get(media_url, stream=True)
        response.raise_for_status() # catch any status error

        # determine the path based on the media type - also change the folder here 
        if media_type == 'image':
            file_path = f"{media_folder}\\ad_{ad_id}_img.png"
        elif media_type == 'video':
            file_path = f"{media_folder}\\ad_{ad_id}_video.mp4"
        else:
            print("Wrong media type.")

        # save the media file
        with open(file_path, 'wb') as media_file:
            media_file.write(response.content)

        print(f"{media_type} of ad with id {ad_id} downloaded successfully to {file_path}")
     # catch any possible exceptions
    except requests.exceptions.RequestException as e:
        print(f"Error during the request: {e}")

    except IOError as e:
        print(f"IOError during file write: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def accept_cookies(driver):
    # accept the cookies if needed
    try:
        # Wait up to 10 seconds for the accept cookies element to be present
        cookies = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='cookie-policy-manage-dialog-accept-button']")))
        # cookies = driver.find_element(By.CSS_SELECTOR, "[data-testid='cookie-policy-manage-dialog-accept-button']")
        cookies.click()
        print("Cookies accepted.")
    except NoSuchElementException:
        print("Cookies already accepted.")

# media download by category
# j - index of the current category, is_last - is it the last category to download, if yes then close the session
def start_media_download(brand, is_file=False, data=[]): 
    # first read the data and open a driver
    if is_file:
        file_path = f'data\\ads\\{brand}'
        data = pd.read_excel(file_path)

    category = os.path.splitext(brand)[0].lower()
    print(f"Downloading media content for {category}.")
    nr_ads_processed = 0

    # initialize folders for the images and videos of current category
    folder_path_img = f"data\\ad_images\\{category}"
    folder_path_vid = f"data\\ad_videos\\{category}"

    # check if the folders exist
    if not os.path.exists(folder_path_img):
        os.makedirs(folder_path_img)

    if not os.path.exists(folder_path_vid):
        os.makedirs(folder_path_vid)
    
    # define some constants first
    img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[2]/a/div[1]/img'
    video_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/video'
    #multpl_img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[3]/div/div/div/div[{}]/div/div/a/div[1]/img'
    # //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
    multpl_img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[3]/div/div/div/div[{}]/div/div/div/img'

    # start the downloads here, accept cookies
    driver = webdriver.Chrome()
    driver.get(data['ad_snapshot_url'][0]) # start from here to accept cookies
    accept_cookies(driver)

    # for each ad in the dataset download the media
    for i in range(0, 100): # change here
        # get the target ad
        driver.get(data['ad_snapshot_url'][96])

        try: # first try to get the img
            img_element = driver.find_element(By.XPATH, img_xpath)
            # if it's found, get its url and download it
            media_url = img_element.get_attribute('src')
            media_type = 'image'
            download_media(media_url, media_type, str(data['id'][i]), folder_path_img)
            nr_ads_processed += 1

        except NoSuchElementException: 
            try: # otherwise try to find the video
                video_element = driver.find_element(By.XPATH, video_xpath)
                # if it's found, get its url and download it
                media_url = video_element.get_attribute('src')
                media_type = 'video'
                download_media(media_url, media_type, str(data['id'][i]), folder_path_vid)
                nr_ads_processed += 1
            
            except NoSuchElementException:
                # means there must be more than 1 image:
                # determine the number of images on the page
                image_count = len(driver.find_elements(By.XPATH, multpl_img_xpath.format('*')))
                if image_count == 0:
                    print(f"No media were downloaded for ad {data['id'][i]}.")
                    continue
                print(f'{image_count} media content found. Trying to retrieve all of them.')
                
                # iterate over the images and download each one
                for img_index in range(1, image_count + 1):
                    multpl_img_element = driver.find_element(By.XPATH, multpl_img_xpath.format(img_index))
                    media_url = multpl_img_element.get_attribute('src')
                    media_type = 'image'
                    download_media(media_url, media_type, f"{str(data['id'][i])}_{img_index}", folder_path_img)
                nr_ads_processed += 1

    print(f'Finished saving media content for {nr_ads_processed} ads for {category}.')
    
    # close the driver if it's the last category to download
    driver.quit()


# new function to download the media in separate brand folders by category
def download_media_by_brand(categ, is_file=False, data=[]): 
    # first read the data and open a driver
    if is_file:
        file_path = f'data\\ads\\{categ}'
        data = pd.read_excel(file_path)
    
    category = os.path.splitext(categ)[0].lower()
    print(f"Downloading media content for {category}.")

    # define some constants 
    img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[2]/a/div[1]/img'
    video_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/video'
    #multpl_img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[3]/div/div/div/div[{}]/div/div/a/div[1]/img'
    # //*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img
    multpl_img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[3]/div/div/div/div[{}]/div/div/div/img'

    brands_all = pd.read_excel("data/brands/brands_all.xlsx")
    
    name_parts = categ.split('.')[0].split('_')
    first_part = name_parts[0]
    second_part = name_parts[1].upper()

    cat_i = brands_all.loc[brands_all['Cat'] == first_part, second_part].reset_index(drop=True)
    
    # for each brand in the category
    # remove red bull racing sportgoods, and pepsico veurne from drinks_be
    for brand in cat_i:
        print(f"Retrieving data for {brand}")

        brand_data = data[data['page_name'].str.contains(brand, case=False)].reset_index(drop=True)
        if brand_data.empty:
            print(f'No ads were found for {brand}.')
            continue

        # initialize folders for the images and videos of current category
        folder_path_img = f"data\\ad_images\\{category}\\{brand}"
        folder_path_vid = f"data\\ad_videos\\{category}\\{brand}"

        # check if the folders exist
        if not os.path.exists(folder_path_img):
            os.makedirs(folder_path_img)

        if not os.path.exists(folder_path_vid):
            os.makedirs(folder_path_vid)

        brand_data = brand_data.sample(frac=1).reset_index(drop=True)
        nr_ads_processed = 0
       
        #brand_data[brand_data.duplicated(subset=['id'], keep=False)]
        #brand_data[brand_data.duplicated(subset=['id', 'ad_creation_time', 'ad_snapshot_url'], keep=False)]

        # start the downloads here, accept cookies
        driver = webdriver.Chrome()
        driver.get(brand_data['ad_snapshot_url'][0]) # start from here to accept cookies
        accept_cookies(driver)

        # for each ad in the dataset download the media
        for i in range(0, min(len(brand_data), 50)): # change here
            # get the target ad
            driver.get(brand_data['ad_snapshot_url'][i])

            try: # first try to get the img
                img_element = driver.find_element(By.XPATH, img_xpath)
                # if it's found, get its url and download it
                media_url = img_element.get_attribute('src')
                media_type = 'image'
                download_media(media_url, media_type, str(brand_data['id'][i]), folder_path_img)
                nr_ads_processed += 1

            except NoSuchElementException: 
                try: # otherwise try to find the video
                    video_element = driver.find_element(By.XPATH, video_xpath)
                    # if it's found, get its url and download it
                    media_url = video_element.get_attribute('src')
                    media_type = 'video'
                    download_media(media_url, media_type, str(brand_data['id'][i]), folder_path_vid)
                    nr_ads_processed += 1
                
                except NoSuchElementException:
                    # means there must be more than 1 image:
                    # determine the number of images on the page
                    image_count = len(driver.find_elements(By.XPATH, multpl_img_xpath.format('*')))
                    if image_count == 0:
                        print(f"No media were downloaded for ad {brand_data['id'][i]}.")
                        continue
                    print(f'{image_count} media content found. Trying to retrieve all of them.')
                    
                    # iterate over the images and download each one
                    for img_index in range(1, image_count + 1):
                        multpl_img_element = driver.find_element(By.XPATH, multpl_img_xpath.format(img_index))
                        media_url = multpl_img_element.get_attribute('src')
                        media_type = 'image'
                        download_media(media_url, media_type, f"{str(brand_data['id'][i])}_{img_index}", folder_path_img)
                    nr_ads_processed += 1

        print(f'Finished saving media content for {nr_ads_processed} ads for {brand} from {category}.')
        print(f'{cat_i.index[cat_i == brand].tolist()[0]+1} out of {len(cat_i)} brands completed.')
        # close the driver if it's the last category to download
        driver.quit()
