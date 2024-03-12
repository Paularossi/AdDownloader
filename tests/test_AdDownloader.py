# tests/test_AdDownloader.py

from AdDownloader.analysis import *
from AdDownloader.media_download import start_media_download
from AdDownloader.helpers import update_access_token
import time


def text_time_by_sample_size(data):
    text_times = pd.DataFrame(columns = ['sample_size', 'text_analysis_time', 'topic_analysis_time'])

    for i in range(500, len(data), 500):
        print(f'===== Starting experiments for sample size {i} =====')
        df = data.sample(i)

        start_time = time.time()
        tokens, freq_dist, word_cl, textblb_sent, nltk_sent = start_text_analysis(df)
        end_time = time.time()
        time_text = end_time - start_time

        start_time = time.time()
        lda_model, coherence, sent_topics_df = get_topics(tokens, df["ad_creative_bodies"])
        end_time = time.time()
        topic_time = end_time - start_time

        new_row = {'sample_size': i, 'text_analysis_time' : time_text, 'topic_analysis_time': topic_time}
        text_times = pd.concat([text_times, pd.DataFrame([new_row], index=[0])], ignore_index=True)

    text_times["minutes"] = text_times["topic_analysis_time"].apply(lambda x: x // 60)
    text_times["seconds"] = text_times["topic_analysis_time"].apply(lambda x: x % 60)

    text_times.to_excel("tests/experiments/text_analysis_time_by_sample_size.xlsx", index=False)

    return text_times


def get_best_nr_of_topics(data):
    df = data.sample(5000)
    topics_nr = pd.DataFrame(columns = ['nr_topics', 'coherence', 'topic_analysis_time'])

    captions = df["ad_creative_bodies"].dropna()
    tokens = captions.apply(preprocess)

    for i in range(2, 10, 1):
        print(f'===== Starting experiments for {i} topics =====')
        start_time = time.time()
        lda_model, coherence, sent_topics_df = get_topics(tokens, captions)
        end_time = time.time()
        topic_time = end_time - start_time

        new_row = {'nr_topics': i, 'coherence' : coherence, 'topic_analysis_time': topic_time}
        topics_nr = pd.concat([topics_nr, pd.DataFrame([new_row], index=[0])], ignore_index=True)

    topics_nr["minutes"] = topics_nr["topic_analysis_time"].apply(lambda x: x // 60)
    topics_nr["seconds"] = topics_nr["topic_analysis_time"].apply(lambda x: x % 60)

    topics_nr.to_excel("tests/experiments/get_best_nr_of_topics_all_eng_5000.xlsx", index=False)

    return topics_nr


def img_download_by_sample_size(data):
    media_download_time = pd.DataFrame(columns = ['sample_size', 'img_download_time'])

    for i in range(50, 510, 50):
        print(f'===== Starting experiments for downloading media content for {i} ads =====')
        start_time = time.time()
        start_media_download(project_name = "testbig2", nr_ads = i, data = data.sample(600))
        end_time = time.time()
        download_time = end_time - start_time

        new_row = {'sample_size': i, 'img_download_time': download_time}
        media_download_time = pd.concat([media_download_time, pd.DataFrame([new_row], index=[0])], ignore_index=True)

    media_download_time["minutes"] = media_download_time["img_download_time"].apply(lambda x: x // 60)
    media_download_time["seconds"] = media_download_time["img_download_time"].apply(lambda x: x % 60)

    media_download_time.to_excel("tests/experiments/media_download_times_pol.xlsx", index=False)

    return media_download_time

    


data_path = "output/testbig3/ads_data/testbig3_processed_data.xlsx"
data = load_data(data_path)
data = data.dropna(subset = ["ad_creative_bodies"])

access_token = input()
data = update_access_token(data, access_token)

text_times = text_time_by_sample_size(data)
print(text_times)

topics_nr_coherence = get_best_nr_of_topics(data)
print(topics_nr_coherence)



##### ANALYZE THE DIFFERENCES BETWEEN SAMPLE SIZES #####
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

### Media download
d1 = pd.read_excel("tests/experiments/media_download_times_all.xlsx")
d2 = pd.read_excel("tests/experiments/media_download_times_pol.xlsx")

d1['ad_type'] = 'ALL'
d2['ad_type'] = 'POL'

merged_df = pd.concat([d1, d2], ignore_index=True)
merged_df = merged_df.dropna()

# avg download time by ad type
plt.figure(figsize=(10, 6))
sns.barplot(x='ad_type', y='img_download_time', data=merged_df, estimator=sum, ci=None)
plt.title('Average Media Download Time by Ad Type')
plt.ylabel('Average Media Download Time (seconds)')
plt.xlabel('Ad Type')
plt.show()

# download time distribution by ad type
plt.figure(figsize=(10, 6))
sns.boxplot(x='ad_type', y='img_download_time', data=merged_df)
plt.title('Media Download Time Distribution by Ad Type')
plt.ylabel('Media Download Time (seconds)')
plt.xlabel('Ad Type')
plt.show()

# download times by size per ad type
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='sample_size', y='img_download_time', hue='ad_type', palette=['blue', 'red'])
plt.title('Download Time for Different Ad Types by Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Download Time (seconds)')
plt.legend(title='Ad Type')
plt.show()

# create a simple model predicting the media download time
from sklearn.metrics import mean_squared_error # scikit-learn==1.4.1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

merged_df['ad_type'] = merged_df['ad_type'].map({'ALL': 0, 'POL': 1})
X = merged_df[['sample_size', 'ad_type']]
y = merged_df['img_download_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error when predicting media download time: {mse}')


### Text and topic analysis
d1 = pd.read_excel("tests/experiments/text_analysis_time_by_sample_size_all.xlsx", )
d2 = pd.read_excel("tests/experiments/text_analysis_time_by_sample_size_pol.xlsx")

d1['ad_type'] = 'ALL'
d2['ad_type'] = 'POL'

merged_df = pd.concat([d1, d2], ignore_index=True, join='inner')
merged_df = merged_df.dropna()

# text analysis time by size per ad type
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='sample_size', y='text_analysis_time', hue='ad_type', palette='viridis')
plt.title('Text Analysis Time by Sample Size and Ad Type')
plt.xlabel('Sample Size')
plt.ylabel('Text Analysis Time (seconds)')
plt.xticks(rotation=45)
plt.legend(title='Ad Type')
plt.show()

# topic analysis time by size per ad type
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='sample_size', y='topic_analysis_time', hue='ad_type', palette='viridis')
plt.title('Topic Analysis Time by Sample Size and Ad Type')
plt.xlabel('Sample Size')
plt.ylabel('Topic Analysis Time (seconds)')
plt.xticks(rotation=45)
plt.legend(title='Ad Type')
plt.show()

# predict the text analysis time based on sample size and ad type
merged_df['ad_type'] = merged_df['ad_type'].map({'ALL': 0, 'POL': 1})
X = merged_df[['sample_size', 'ad_type']]
y = merged_df['text_analysis_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error when predicting text analysis time: {mse}')

# calculate the average time for 500 ads
merged_df['avg_text_time_per_ad'] = merged_df['text_analysis_time'] / merged_df['sample_size']
avg_text_time_1000_ads = merged_df['avg_text_time_per_ad'].mean() * 1000


# predict the topic analysis time based on sample size and ad type
y = merged_df['topic_analysis_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error when predicting topic analysis time: {mse}')

# calculate the average time for 500 ads
merged_df['avg_topic_time_per_ad'] = merged_df['topic_analysis_time'] / merged_df['sample_size']
avg_topic_time_1000_ads = merged_df['avg_topic_time_per_ad'].mean() * 1000

### Number of topics (also coherence)
d1 = pd.read_excel("tests/experiments/get_best_nr_of_topics_all_3000.xlsx")
d2 = pd.read_excel("tests/experiments/get_best_nr_of_topics_pol_3000.xlsx")

d3 = pd.read_excel("tests/experiments/get_best_nr_of_topics_all_eng_5000.xlsx")
d4 = pd.read_excel("tests/experiments/get_best_nr_of_topics_pol_5000.xlsx")

d1['ad_type'] = 'ALL'
d3['ad_type'] = 'ALL'
d2['ad_type'] = 'POL'
d4['ad_type'] = 'POL'

d1['sample_size'] = 3000
d2['sample_size'] = 3000

d3['sample_size'] = 5000
d4['sample_size'] = 5000


merged_df = pd.concat([d1, d2, d3, d4], ignore_index=True, join='inner')
merged_df = merged_df.dropna()

# coherence by ad type and sample size
g = sns.FacetGrid(merged_df, col="ad_type", row="sample_size", margin_titles=True, height=3, aspect=1.3)
g.map(sns.scatterplot, "nr_topics", "coherence")
g.figure.suptitle('Coherence by Number of Topics, Ad Type, and Sample Size', fontsize=16)
g.figure.subplots_adjust(top=0.9) 
plt.show()

# topic analysis time by ad type and sample size
g = sns.FacetGrid(merged_df, col="ad_type", row="sample_size", margin_titles=True, height=3, aspect=1.3)
g.map(sns.scatterplot, "nr_topics", "topic_analysis_time")
g.figure.suptitle('Topic Analysis Time by Number of Topics, Ad Type, and Sample Size', fontsize=16)
g.figure.subplots_adjust(top=0.9)  
plt.show()





# plt.figure(figsize=(12, 6))
# g = sns.lineplot(data=merged_df, x='nr_topics', y='coherence', hue='ad_type', style='sample_size', markers=True, dashes=False)
# g.set_title('Coherence by Number of Topics, Ad Type, and Sample Size')
# g.set_xlabel('Number of Topics')
# g.set_ylabel('Coherence')
# plt.legend(title='Ad Type & Sample Size', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 6))
# g = sns.lineplot(data=merged_df, x='nr_topics', y='topic_analysis_time', hue='ad_type', style='sample_size', markers=True, dashes=False)
# g.set_title('Topic Analysis Time by Number of Topics, Ad Type, and Sample Size')
# g.set_xlabel('Number of Topics')
# g.set_ylabel('Topic Analysis Time (seconds)')
# plt.legend(title='Ad Type & Sample Size', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
