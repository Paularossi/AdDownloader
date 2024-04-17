import os
from AdDownloader.media_download import extract_frames
from AdDownloader.analysis import *
import matplotlib.pyplot as plt
import ast
import math


# extract one frame from each video
video_path = 'output/uselections/ads_videos'
videos = os.listdir(video_path)

for video in videos:
    extract_frames(video = video, project_name = "uselections", num_frames = 1)


# analyze the ad text
data_path = "output/uselections/ads_data/uselections-sample.xlsx" # need to use the processed data
data = load_data(data_path)           
data['id'] = data['id'].astype(str)
tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(data['ad_creative_bodies'])
print(f"Most common 10 keywords: {freq_dist.most_common(10)}")

plt.imshow(word_cl, interpolation='bilinear')
plt.axis("off")
plt.show()

# check the sentiment
nltk_sent.head(20) # or textblb_sent
data["textblb_sent"] = textblb_sent
data["nltk_sent"] = nltk_sent

# topics
lda_model, coherence, topics_df = get_topics(tokens, nr_topics=3) # change nr of topics
topics_df.head(20)

fig = show_topics_top_pages(topics_df, data)
fig.show()

# print the topics and the coherence score
for idx, topic in lda_model.print_topics(num_words=8):
    print("Topic: {} \nWords: {}".format(idx, topic))
print('Coherence Score:', coherence)
data = pd.concat([data, topics_df], axis=1)


# image analysis
images_path = f"output/uselections/ads_images"
images_path2 = f"output/uselections/video_frames"

img_df = analyse_image_folder(images_path)
img_df2 = analyse_image_folder(images_path2)


data_img_new = pd.merge(data, img_df, left_on='id', right_on='ad_id', how='inner')
data_img_new2 = pd.merge(data, img_df2, left_on='id', right_on='ad_id', how='inner')

final_data = pd.concat([data_img_new, data_img_new2], axis=0)
final_data.to_excel("output/uselections/topic_img_data.xlsx", index=False)

# continue with blip...
img_caption = blip_call(images_path)
img_caption2 = blip_call(images_path2)

data_capt = pd.merge(final_data, img_caption, left_on='id', right_on='ad_id', how='inner')
data_capt2 = pd.merge(final_data, img_caption2, left_on='id', right_on='ad_id', how='inner')


final_captions = pd.concat([data_capt, data_capt2], axis=0)

tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(final_captions['img_caption'])
print(f"Most common 10 keywords: {freq_dist.most_common(10)}")
final_captions["textblb_sent_caption"] = textblb_sent
final_captions["nltk_sent_caption"] = nltk_sent

lda_model, coherence, topics_df = get_topics(tokens, nr_topics=5)
for idx, topic in lda_model.print_topics(num_words=8):
    print("Topic: {} \nWords: {}".format(idx, topic))
print('Coherence Score:', coherence)


final_captions["dom_topic_caption"] = topics_df["dom_topic"]
final_captions["perc_contr_caption"] = topics_df["perc_contr"]
final_captions["topic_keywords_caption"] = topics_df["topic_keywords"]

final_captions.to_excel("output/uselections/final-data-all-features.xlsx", index=False)



import time
# question answering
question = "Is there an African-American person in the image? Is there an Asian person? Is there a White person?" # ask each ethnicity hispanic, african-american, white, asian, man vs woman, mask
question1 = "Which of the following emotions is predominant in the image: happiness, sadness, anger, disgust?"
question2 = "Which of the following ethinicities you see in the image: Hispanic, African-American, Asian, White?"
question3 = "Do you see men, women or no people in the image?"
# 10 specific symbols , e.g. men or women, masks, 
# is the picture happy? sad? etc.
questions = """Do you see men, women or no people in the image? Is there an African-American person in the image? Is there an Asian person? Is there a White person?
            Which of the following emotions is predominant in the image: happiness, sadness, anger, disgust? Do you see surgical masks?"""
img_path_test = "output/uselections/test_images"
img_content = blip_call(img_path_test, task="visual_question_answering", questions=questions)


# raw_image = Image.open(os.path.join(images_path, 'ad_647543005953352_img.png')).convert('RGB')
# inputs_quest = processor(raw_image, questions, return_tensors="pt")

# out_quest = model_answering.generate(**inputs_quest, max_length = 30)
# #print(processor.decode(out_quest[0], skip_special_tokens=True))
# processor.decode(out_quest[0], skip_special_tokens=True)

start_time = time.time()
img_content = blip_call(images_path, task="visual_question_answering", questions=questions)
img_content2 = blip_call(images_path2, task="visual_question_answering", questions=questions)
end_time = time.time()
print(f"Finished q. answering in {end_time - start_time} seconds.")
# 174 mins 15 sec
# 114 min 19 sec
img_content2.to_excel("output/uselections/question_answering2.xlsx", index=False)
#img_content2 = pd.read_excel("output/uselections/question_answering2.xlsx")
#### run the lines above and merge with "output/uselections/final-data-all-features.xlsx"


# first, just using the party, then using general features, and lastly using our features (captioning, etc)
new_columns = ["ad_id", "people", "afr-amer", "asian", "white", "emotion", "masks"]
img_content2.columns = new_columns # change the column names for readability

# some checks first:
# adjust the 'people' column based on 'afr-amer', 'asian', and 'white' values
img_content2['people'] = img_content2.apply(
    lambda row: 'no' if row['afr-amer'] == 'no' and row['asian'] == 'no' and row['white'] == 'no' else row['people'],
    axis=1)

# adjust the ethnicities columns based on whether there are people
mask = img_content['people'] == 'no'
img_content.loc[mask, ['afr-amer', 'asian', 'white']] = 'no'

# recode the binary columns
columns_to_map = ['afr-amer', 'asian', 'white', 'masks']
img_content2[columns_to_map] = img_content2[columns_to_map].replace({'yes': 1, 'no': 0})

img_content2["emotion"].unique() # ['anger', 'happy', 'sadness', 'happiness', 'sad', 'none', 'zero']
img_content["emotion"] = img_content["emotion"].replace({'zero': 'none', 'happy': 'happiness', 'sad': 'sadness'})
img_content[img_content["emotion"] == 'none']

# recode the emotion
dummy_vars = pd.get_dummies(img_content["emotion"], prefix="is")
columns = dummy_vars.columns.tolist()
new_columns = [col for col in columns if col != 'is_none'] + ['is_none']
dummy_vars = dummy_vars[new_columns] # reindex to have is_none last
img_content = pd.concat([img_content, dummy_vars], axis=1)

dummy_vars2 = pd.get_dummies(img_content2["emotion"], prefix="is")
dummy_vars2["is_none"] = False
img_content2 = pd.concat([img_content2, dummy_vars2], axis=1)


# recode the people
img_content["people"].unique() # ['anger', 'happy', 'sadness', 'happiness', 'sad', 'none', 'zero']
img_content["people"] = img_content["people"].replace({'men': 'man', 'women': 'woman', 'no': 'no_people'})

dummy_vars = pd.get_dummies(img_content["people"], prefix="is")
img_content = pd.concat([img_content, dummy_vars], axis=1)

dummy_vars2 = pd.get_dummies(img_content2["people"], prefix="is")
img_content2 = pd.concat([img_content2, dummy_vars2], axis=1)

# merge all together
img_content['ad_id'] = img_content['ad_id'].astype(str)
img_content2['ad_id'] = img_content2['ad_id'].astype(str)
data_quest = pd.merge(final_captions, img_content, left_on='id', right_on='ad_id', how='inner')
data_quest2 = pd.merge(final_captions, img_content2, left_on='id', right_on='ad_id', how='inner')
final_questions = pd.concat([data_quest, data_quest2], axis=0, ignore_index=True)

final_questions['spend'] = final_questions['spend'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
final_questions['spend_avg'] = final_questions['spend'].apply(lambda x: math.ceil((int(x['lower_bound']) + int(x.get('upper_bound', x['lower_bound']))) / 2))
final_questions.to_excel("output/uselections/us-elections-final.xlsx", index=False)

# continue here...

final_questions.columns







# in case not matched ids:
id1 = img_content2["ad_id"]
id2 = data_quest2["id"]
id1[~id1.isin(id2)]
img_content2.loc[94,"ad_id"] = "1526222434253284"