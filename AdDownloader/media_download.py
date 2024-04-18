"""This module provides the functionality of media content download of the AdDownloader using Selenium."""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import requests
import os
import cv2
from AdDownloader.helpers import configure_logging, close_logger


def download_media(media_url, media_type, ad_id, media_folder):
    """
    Download media content for an ad given its ID.

    :param media_url: The url address for accessing the media content.
    :type media_url: str
    :param media_type: The type of the media content to download, can be 'image' or 'videos'.
    :type media_type: str
    :param ad_id: The ID of the ad for which media content is downloaded.
    :type ad_id: str
    :param media_folder: The path to the folder where media content will be saved.
    :type media_folder: str
    """
    
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
    """
    Accept the cookies in a running Chrome webdriver. Only needs to be done once, when openning the webdriver.

    :param driver: A running Chrome webdriver.
    :type driver: webdriver.Chrome
    """
    # accept the cookies if needed
    try:
        # wait up to 10 seconds for the accept cookies element to be present
        cookies = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='cookie-policy-manage-dialog-accept-button']")))
        cookies.click()
        print("Cookies accepted.")
    except NoSuchElementException:
        print("Cookies already accepted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def start_media_download(project_name, nr_ads, data=[]):
    """
    Start media content download for a given project and desired number of ads. 
    The ads media are saved in the output folder with the project_name.

    :param project_name: The name of the current project.
    :type project_name: str
    :param nr_ads: The desired number of ads for which media content should be downloaded.
    :type nr_ads: int
    :param data: A dataframe containing an `ad_snapshot_url` column.
    :type data: pandas.DataFrame
    """

    # configure logger
    logger = configure_logging(project_name)

    # check if data was provided
    if data is None or len(data) == 0:
        logger.error("No data was provided for media download. Please try again.")
        return(print("No data was provided for media download. Please try again."))


    # check if the nr of ads to download is within the length of the data
    if nr_ads > len(data):
        nr_ads = len(data)
    print(f"Downloading media content for project {project_name}.")
    logger.info(f'Downloading media content for project {project_name}.')
    nr_ads_processed = 0
    nr_ads_failed = 0

    # initialize folders for the images and videos of current category
    folder_path_img = f"output\\{project_name}\\ads_images"
    folder_path_vid = f"output\\{project_name}\\ads_videos"

    # check if the folders exist
    if not os.path.exists(folder_path_img):
        os.makedirs(folder_path_img)

    if not os.path.exists(folder_path_vid):
        os.makedirs(folder_path_vid)
    
    # define some constants for the xpaths
    img_xpath_1 = '//*[@id="content"]/div/div/div/div/div/div/div[2]/a/div[1]/img'
    img_xpath_2 = '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img'
    video_xpath_1 = '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/video'
    video_xpath_2 = '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/video'
    multpl_img_xpath = '//*[@id="content"]/div/div/div/div/div/div/div[3]/div/div/div/div[{}]/div/div/div/img'

    # sample the nr_ads
    data = data.sample(nr_ads)
    data = data.reset_index(drop=True)

    # start the downloads here, accept cookies
    driver = webdriver.Chrome()

    driver.get(data['ad_snapshot_url'][0]) # start from here to accept cookies
    accept_cookies(driver)    
    
    # for each ad in the dataset download the media
    for i in range(0, nr_ads): #TODO: randomize the ads to download
        # get the target ad
        success = False
        driver.get(data['ad_snapshot_url'][i])

        try: # first try to get the img using first xpath
            img_element = driver.find_element(By.XPATH, img_xpath_1)
            # if it's found, get its url and download it
            media_url = img_element.get_attribute('src')
            media_type = 'image'
            download_media(media_url, media_type, str(data['id'][i]), folder_path_img)
            nr_ads_processed += 1
            success = True

        except NoSuchElementException: 
            try: # otherwise try the second xpath
                img_element = driver.find_element(By.XPATH, img_xpath_2)
                # if it's found, get its url and download it
                media_url = img_element.get_attribute('src')
                media_type = 'image'
                download_media(media_url, media_type, str(data['id'][i]), folder_path_img)
                nr_ads_processed += 1
                success = True

            except NoSuchElementException: 
                pass

        try: # if it's not an image, try to find the video with first xpath
            video_element = driver.find_element(By.XPATH, video_xpath_2)
            # if it's found, get its url and download it
            media_url = video_element.get_attribute('src')
            media_type = 'video'
            download_media(media_url, media_type, str(data['id'][i]), folder_path_vid)
            nr_ads_processed += 1
            success = True
        
        except NoSuchElementException:
            try: # otherwise try the second xpath
                video_element = driver.find_element(By.XPATH, video_xpath_1)
                # if it's found, get its url and download it
                media_url = video_element.get_attribute('src')
                media_type = 'video'
                download_media(media_url, media_type, str(data['id'][i]), folder_path_vid)
                nr_ads_processed += 1
                success = True
            
            except NoSuchElementException:
                pass

        try: # check if there is more than one image
            # determine the number of images on the page
            image_count = len(driver.find_elements(By.XPATH, multpl_img_xpath.format('*')))
            if image_count > 0:
                print(f'{image_count} media content found. Trying to retrieve all of them.')
                
                # iterate over the images and download each one
                for img_index in range(1, image_count + 1):
                    multpl_img_element = driver.find_element(By.XPATH, multpl_img_xpath.format(img_index))
                    media_url = multpl_img_element.get_attribute('src')
                    media_type = 'image'
                    download_media(media_url, media_type, f"{str(data['id'][i])}_{img_index}", folder_path_img)
                nr_ads_processed += 1
                success = True
        
        except NoSuchElementException:
            pass

        if not success:
            nr_ads_failed += 1
            print(f"No media were downloaded for ad {data['id'][i]}.")
            logger.error(f"No media were downloaded for ad {data['id'][i]}")
        
        if (i+1)/nr_ads == 0.25:
            print("===== 25% done =====")
        elif (i+1)/nr_ads == 0.5:
            print("===== 50% done =====")
        elif (i+1)/nr_ads == 0.75:
            print("===== 75% done =====")


    print(f'Finished saving media content for {nr_ads_processed} ads for project {project_name}.')
    logger.info(f'Finished saving media content for {nr_ads_processed} ads for project {project_name}.')
    logger.info(f'Media failed to download for {nr_ads_failed} ads. Success rate: {nr_ads_processed / nr_ads}')

    # close the driver once it's done downloading
    driver.quit()

    # close the logger
    close_logger(logger)


def extract_frames(video, project_name, interval = None, num_frames = None):
    """
    Extract a number of frames from ad videos

    :param video: The name of the video for which frames should be extracted.
    :type video: str
    :param project_name: The name of the current project.
    :type project_name: str
    :param interval: The interval between the (in seconds), optional. Should be specified instead of `num_frames`.
    :type interval: int
    :param num_frames: The number of frames to extract, distributed evenly, optional. Should be specified instead of the `interval`.
    :type num_frames: int
    """
    video_path = f"output/{project_name}/ads_videos/{video}"
    # create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # get the ad id
    ad_id = os.path.basename(video_path).split('_')[1]
    frame_dir = f"output/{project_name}/video_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    if interval is not None:
        print(f"Processing {video_path} | FPS: {fps} | Total Frames: {frame_count} | Duration: {duration}s")
        # read the video and save frames every interval
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # check if the current frame number is the one we want to save
            if frame_number % (interval * fps) == 0:
                frame_path = f"{frame_dir}/ad_{ad_id}_frame{frame_number}.png"
                cv2.imwrite(frame_path, frame)
                print(f"Saved {frame_path}")

            frame_number += 1
    elif num_frames is not None:
        frames_to_capture = [(x * frame_count) // (num_frames + 1) for x in range(1, num_frames + 1)]
        print(f"Processing {video_path} | FPS: {fps} | Total Frames: {frame_count} | Frames to capture: {frames_to_capture}")
        for frame_number in frames_to_capture:
            # set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame_path = f"{frame_dir}/ad_{ad_id}_frame{frame_number}.png"
                cv2.imwrite(frame_path, frame)
                print(f"Saved {frame_path}")

    # release the VideoCapture object
    cap.release()