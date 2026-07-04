"""This module provides the functionality of media content download of the AdDownloader using Selenium."""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
import requests
import os
import random
import time
import cv2
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from AdDownloader.helpers import configure_logging, close_logger

chrome_opts = Options()
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--enable-unsafe-swiftshader")
chrome_opts.add_argument("--log-level=4") # suppress logs
chrome_opts.add_argument("--disable-notifications")
# a fixed, "desktop-sized" window keeps Facebook from serving a narrower responsive layout (which uses a different DOM structure and breaks the xpaths below)
chrome_opts.add_argument("--window-size=1920,1080")
chrome_opts.add_argument("--start-maximized")
# force English so the removed-ad text detection below matches reliably regardless of the account/browser locale
chrome_opts.add_argument("--lang=en-US")
chrome_opts.add_experimental_option("prefs", {"intl.accept_languages": "en-US,en"})

# xpaths Facebook has used for the media elements on an ad snapshot page, tried in order
# Meta redesigns this page periodically changing the div nesting and breaking these 
# adding a new xpath here is optional (the fallback still catches it)
IMG_XPATHS = [
    '//*[@id="content"]/div/div/div/div/div/div/div/div[2]/a/div[1]/img',
    '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/img',
    '//*[@id="content"]/div/div/div/div/div/div/div/div/div[2]/div/a/div[1]/img',
    '//*[@id="content"]/div/div/div/div/div/div/div/div/div[2]/a/div[1]/img',
    '//*[@id="content"]/div/div/div/div/div/div/div/div/div[2]/div[2]/img'
]

VIDEO_XPATHS = [
    '//*[@id="content"]/div/div/div/div/div/div/div[2]/div[2]/video',
    '//*[@id="content"]/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/video',
    '//*[@id="content"]/div/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/div/video',
    '//*[@id="content"]/div/div/div/div/div/div/div/div/div[2]/div/div[2]/div/div/div/div/div/video',
]

MULTI_IMG_XPATH = '//*[@id="content"]/div/div/div/div/div/div/div/div[3]/div/div[2]/div/div/div[{}]/div/div/a/div[1]/img'

# substrings (case-insensitive) Meta shows on a snapshot page when the ad creative itself was taken down for a policy violation, as opposed to an ad that never had an image/video.
# This list is a best-effort starting point - if you find ads being mislabelled, inspect the actual page text with detect_removed_ad() disabled and add the missing phrase here.
REMOVED_AD_PHRASES = [
    "ad has been removed",
    "ad has been taken down",
    "ad is no longer available",
    "ad isn't available",
    "ad is not available",
    "this content isn't available",
    "doesn't comply with",
    "violat", # matches "violates"/"violation" of advertising standards/policies
    "advertising standards",
]


def find_media_element(driver, xpaths):
    """
    Try a list of candidate xpaths in order and return the first matching element.

    :param driver: A running Chrome webdriver, already navigated to the target page.
    :type driver: webdriver.Chrome
    :param xpaths: Candidate xpaths to try, in order.
    :type xpaths: list[str]
    :returns: The first matching WebElement, or None if none of the xpaths matched.
    :rtype: WebElement | None
    """
    for xpath in xpaths:
        try:
            return driver.find_element(By.XPATH, xpath)
        except NoSuchElementException:
            continue
    return None


# JS run in-browser by find_fallback_media() below: gathers every <img>/<video> anywhere inside
# #content in one pass, in one round-trip, rather than the multiple Selenium calls it'd take to
# fetch and measure each element individually through find_elements() + .size.
_FALLBACK_MEDIA_JS = """
    return Array.from(document.querySelectorAll('#content img, #content video')).map(el => ({
        tag: el.tagName.toLowerCase(),
        src: el.currentSrc || el.src,
        width: el.offsetWidth,
        height: el.offsetHeight
    })).filter(el => el.src && el.width >= arguments[0] && el.height >= arguments[0]);
"""


def find_fallback_media(driver, min_dimension=100):
    """
    Structure-agnostic fallback for when none of the known positional xpaths above matched:
    instead of guessing another exact div path, look for any <img>/<video> anywhere inside
    #content that's at least `min_dimension` px on each side (filtering out small icons like
    the page avatar). This costs one query total (a single execute_script call) and, unlike the
    xpath lists, keeps working across page redesigns without needing a manual update - it's only
    used as a last resort, so it adds no overhead for ads that already match a known xpath.

    :param driver: A running Chrome webdriver, already navigated to the target page.
    :type driver: webdriver.Chrome
    :param min_dimension: Minimum width/height (px) for an element to count as real media rather
        than an icon/avatar.
    :type min_dimension: int
    :returns: A tuple (images, video). images is a list of {'src', 'width', 'height'} dicts for
        every large enough <img> found, largest first (empty if none). video is the largest
        matching <video> dict, or None.
    :rtype: tuple(list[dict], dict | None)
    """
    try:
        candidates = driver.execute_script(_FALLBACK_MEDIA_JS, min_dimension)
    except Exception:
        return [], None

    images = sorted((c for c in candidates if c['tag'] == 'img'), key=lambda c: c['width'] * c['height'], reverse=True)
    videos = sorted((c for c in candidates if c['tag'] == 'video'), key=lambda c: c['width'] * c['height'], reverse=True)
    return images, (videos[0] if videos else None)


def detect_removed_ad(driver):
    """
    Check whether the current ad snapshot page indicates the ad creative was removed by Meta
    for not following advertising policies (as opposed to the ad simply being text-only).

    :param driver: A running Chrome webdriver, already navigated to the target page.
    :type driver: webdriver.Chrome
    :returns: True if a known "removed" message was found on the page.
    :rtype: bool
    """
    try:
        page_text = driver.page_source.lower()
    except Exception:
        return False
    return any(phrase in page_text for phrase in REMOVED_AD_PHRASES)


def download_media(media_url, media_type, ad_id, media_folder):
    """
    Download media content for an ad given its ID.

    :param media_url: The url address for accessing the media content.
    :type media_url: str
    :param media_type: The type of the media content to download, can be 'image' or 'video'.
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
            file_path = f"{media_folder}/ad_{ad_id}_img.png"
        elif media_type == 'video':
            file_path = f"{media_folder}/ad_{ad_id}_video.mp4"
        else:
            print("Wrong media type.")
            return

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
        # wait up to 20 seconds for the accept cookies element to be present
        cookies = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="facebook"]/body/div[3]/div[2]/div/div/div/div/div[3]/div[2]/div/div[2]/div[1]'))
        )
        cookies.click()
        print("Cookies accepted.")
    except NoSuchElementException:
        print("Cookies already accepted.")
    except TimeoutException:
        print("Cookie dialog not found; maybe already dismissed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def start_media_download(project_name, nr_ads, data=None, random_state=None):
    """
    Start media content download for a given project and desired number of ads.
    The ads media are saved in the output folder with the project_name.

    :param project_name: The name of the current project.
    :type project_name: str
    :param nr_ads: The desired number of ads for which media content should be downloaded.
    :type nr_ads: int
    :param data: A dataframe containing an `ad_snapshot_url` column.
    :type data: pandas.DataFrame
    :param random_state: Seed used to sample `nr_ads` ads out of `data`, for reproducibility. Default is
        None, in which case a seed is generated automatically and printed/logged so the exact same
        sample of ads can be reproduced later by passing it back in.
    :type random_state: int, optional
    """

    # configure logger
    logger = configure_logging(project_name)

    # check if data was provided
    if data is None or len(data) == 0:
        logger.error("No data was provided for media download. Please try again.")
        return(print("No data was provided for media download. Please try again."))

    # check if the nr of ads to download is within the length of the data
    if nr_ads > len(data):
        print(f'More ad media requested than available in the data. Downloading the maximum number ({len(data)}).')
        logger.warning(f'More ads requested than available in the data. Downloading the maximum number ({len(data)}).')
        nr_ads = len(data)
        
    print(f"Downloading media content for project {project_name}.")
    logger.info(f'Downloading media content for project {project_name}.')
    nr_ads_processed = 0
    nr_ads_failed = 0

    # initialize folders for the images and videos of current category
    folder_path_img = f"output/{project_name}/ads_images"
    folder_path_vid = f"output/{project_name}/ads_videos"

    # check if the folders exist
    if not os.path.exists(folder_path_img):
        os.makedirs(folder_path_img)

    if not os.path.exists(folder_path_vid):
        os.makedirs(folder_path_vid)
    
    # sample the nr_ads - use a fixed seed so the exact same sample can be reproduced later,
    # e.g. for a paper's replication package; if none was given, generate and report one
    if random_state is None:
        random_state = random.randint(0, 2**32 - 1)
    print(f"Sampling {nr_ads} ads using random_state={random_state}. Pass this value as `random_state` to reproduce this exact sample.")
    logger.info(f'Sampling {nr_ads} ads using random_state={random_state}.')

    data = data.sample(nr_ads, random_state=random_state)
    data = data.reset_index(drop=True)
    media_statuses = [] # one entry per ad: {"id": ..., "media_status": ...}

    # start the downloads here, accept cookies
    driver = webdriver.Chrome(
        service = Service(ChromeDriverManager().install()),
        options = chrome_opts,
    )

    # wrap the whole run in try/finally so the driver always quits, even if a page crashes or raises an unexpected exception
    try:
        driver.get(data['ad_snapshot_url'][0]) # start from here to accept cookies
        accept_cookies(driver)

        progress_columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*progress_columns) as progress:
            task = progress.add_task(f"Downloading media for {project_name}", total=nr_ads)

            # for each ad in the dataset download the media
            for i in range(0, nr_ads):
                ad_id = str(data['id'][i]) # get the target ad
                success = False
                media_status = "no_media_detected" # overwritten below once we know more

                driver.get(data['ad_snapshot_url'][i])
                # the ad snapshot page renders its media asynchronously via JS and the image xpaths
                # can fire before the <img> exists yet, so wait a bit before trying to find the image
                time.sleep(1.5)

                # try to find a single image
                img_element = find_media_element(driver, IMG_XPATHS)
                if img_element is not None:
                    media_url = img_element.get_attribute('src')
                    download_media(media_url, 'image', ad_id, folder_path_img)
                    success = True
                    media_status = 'image'

                # try to find a single video (independent of whether an image was found, since a small number of ads carry both)
                video_element = find_media_element(driver, VIDEO_XPATHS)
                if video_element is not None:
                    media_url = video_element.get_attribute('src')
                    download_media(media_url, 'video', ad_id, folder_path_vid)
                    media_status = 'image_and_video' if success else 'video'
                    success = True

                # check if there is more than one image (carousel-style ad)
                image_count = len(driver.find_elements(By.XPATH, MULTI_IMG_XPATH.format('*')))
                if image_count > 0:
                    print(f'{image_count} media content found. Trying to retrieve all of them.')
                    for img_index in range(1, image_count + 1):
                        multpl_img_element = driver.find_element(By.XPATH, MULTI_IMG_XPATH.format(img_index))
                        media_url = multpl_img_element.get_attribute('src')
                        download_media(media_url, 'image', f"{ad_id}_{img_index}", folder_path_img)
                    success = True
                    media_status = 'multiple_images'

                if not success:
                    # none of the known positional xpaths matched (e.g. a page layout we haven't
                    # seen yet) - try a generic, size-filtered search before giving up
                    fallback_images, fallback_video = find_fallback_media(driver)
                    if fallback_images:
                        for idx, img in enumerate(fallback_images, start=1):
                            suffix = ad_id if len(fallback_images) == 1 else f"{ad_id}_{idx}"
                            download_media(img['src'], 'image', suffix, folder_path_img)
                        success = True
                        media_status = 'image' if len(fallback_images) == 1 else 'multiple_images'
                    if fallback_video is not None:
                        download_media(fallback_video['src'], 'video', ad_id, folder_path_vid)
                        media_status = 'image_and_video' if success else 'video'
                        success = True

                if not success:
                    # still nothing found - figure out why before giving up
                    if detect_removed_ad(driver):
                        media_status = 'removed_policy_violation'
                        logger.info(f"Ad {ad_id} media was removed by Meta for policy violation.")
                    else:
                        media_status = 'no_media_detected'
                        nr_ads_failed += 1
                        print(f"No media were downloaded for ad {ad_id}.")
                        logger.error(f"No media were downloaded for ad {ad_id}")
                else:
                    nr_ads_processed += 1

                media_statuses.append({"id": data['id'][i], "media_status": media_status})
                progress.advance(task)

        print(f'Finished saving media content for {nr_ads_processed} ads for project {project_name}.')
        logger.info(f'Finished saving media content for {nr_ads_processed} ads for project {project_name}.')
        logger.info(f'Media failed to download for {nr_ads_failed} ads. Success rate: {nr_ads_processed / nr_ads}')

    finally:
        # close the driver once it's done downloading, even on failure
        driver.quit()

    try:
        # save a media_status column for this batch, so ads with no image/video can be told apart:
        # 'removed_policy_violation' (Meta took the creative down), 'no_media_detected' (neither the
        # known xpaths nor the generic fallback found anything - likely a genuinely text-only ad),
        # or one of 'image' / 'video' / 'image_and_video' / 'multiple_images' for successful downloads.
        status_path = f"output/{project_name}/ads_data"
        if not os.path.exists(status_path):
            os.makedirs(status_path)
        status_df = pd.DataFrame(media_statuses)
        status_df['random_state'] = random_state # so the sample used for this run can be reproduced later
        status_df.to_excel(f"{status_path}/{project_name}_media_status.xlsx", index=False)
        removed_count = (status_df['media_status'] == 'removed_policy_violation').sum()
        print(f"{removed_count} of {nr_ads} ads had media removed by Meta for policy violations (see {project_name}_media_status.xlsx).")
        logger.info(f'Saved media status for {nr_ads} ads to {project_name}_media_status.xlsx ({removed_count} removed for policy violation).')
    finally:
        # close the logger, even if saving the media status excel above failed
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