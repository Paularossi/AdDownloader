import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from AdDownloader.helpers import flatten_age_country_gender

st.title('Ad Data analysis')

DATE_MIN = 'ad_delivery_start_time'
DATE_MAX = 'ad_delivery_stop_time'

# list of demographic columns
DEMOGRAPHIC_COLS = ['female_13-17', 'female_18-24', 'female_25-34', 'female_35-44', 'female_45-54', 'female_55-64', 'female_65+',
                        'male_13-17', 'male_18-24', 'male_25-34', 'male_35-44', 'male_45-54', 'male_55-64', 'male_65+',
                        'unknown_13-17', 'unknown_18-24', 'unknown_25-34', 'unknown_35-44', 'unknown_45-54', 'unknown_55-64', 'unknown_65+']
GENDERS = ['female', 'male', 'unknown']
AGE_RANGES = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

# separate demographic columns into genders
female_columns = [col for col in DEMOGRAPHIC_COLS if 'female' in col]
male_columns = [col for col in DEMOGRAPHIC_COLS if 'male' in col and not 'female' in col]
unknown_columns = [col for col in DEMOGRAPHIC_COLS if 'unknown' in col]

# separate demographic columns into age ranges
age_13_17_columns = [col for col in DEMOGRAPHIC_COLS if '13-17' in col]
age_18_24_columns = [col for col in DEMOGRAPHIC_COLS if '18-24' in col]
age_25_34_columns = [col for col in DEMOGRAPHIC_COLS if '25-34' in col]
age_35_44_columns = [col for col in DEMOGRAPHIC_COLS if '35-44' in col]
age_45_54_columns = [col for col in DEMOGRAPHIC_COLS if '45-54' in col]
age_55_64_columns = [col for col in DEMOGRAPHIC_COLS if '55-64' in col]
age_65_columns = [col for col in DEMOGRAPHIC_COLS if '65+' in col]


@st.cache_data
def load_data(data_path, nrows=100):
    data = pd.read_excel(data_path, nrows=nrows)
    data[DATE_MIN] = pd.to_datetime(data[DATE_MIN])
    data[DATE_MAX] = pd.to_datetime(data[DATE_MAX])

    # check if the age_country_gender_reach_breakdown column has been processed
    if not (any(col in data.columns for col in DEMOGRAPHIC_COLS)):
        #data['flattened_data'] = data['age_country_gender_reach_breakdown'].apply(flatten_age_country_gender, target_country=country)
        # create a new DataFrame from the flattened data
        #flattened_df = pd.DataFrame(data['flattened_data'].sum()) # DONT FORGET TO CHANGE HERE
        print("`age_country_gender_reach_breakdown` column needs to be processed.")
    return data


# transpose the data to have age ranges on the x-axis
def transform_data_by_age(data):
    age_columns = []

    # check if 'age_13_17_columns' exist before including them
    if 'male_13-17' in data.columns:
        age_columns.append(data[age_13_17_columns].values.flatten())

    age_columns.extend([
        data[age_18_24_columns].values.flatten(),
        data[age_25_34_columns].values.flatten(),
        data[age_35_44_columns].values.flatten(),
        data[age_45_54_columns].values.flatten(),
        data[age_55_64_columns].values.flatten(),
        data[age_65_columns].values.flatten()
    ])

    return age_columns


data_load_state = st.text('Loading data...')
data_path = f"output/teststream/ads_data/processed_data.xlsx"
data = load_data(data_path)
data_load_state.text("Done! (using st.cache_data)")

data_by_age = transform_data_by_age(data)

# transpose the data to have genders on the x-axis
data_by_gender = [data[female_columns].values.flatten(), data[male_columns].values.flatten(), data[unknown_columns].values.flatten()]
long_format_df = pd.DataFrame({
    'Reach': [value for sublist in data_by_gender for value in sublist],  # Flatten the list
    'Gender': [label for label, sublist in zip(GENDERS, data_by_gender) for _ in sublist]  # Repeat labels accordingly
})

fig = px.box(long_format_df, y='Reach', x='Gender', color='Gender',
             title=f'Reach Across Genders for all ads for project x',
             labels={'Reach': 'Reach', 'Gender': 'Gender'},
             orientation='v',  # Vertical boxplot
             category_orders={"Gender": GENDERS}  # Order the categories if needed
)

# Customize the layout for better presentation
fig.update_layout(
    xaxis_title="Gender",
    yaxis_title="Reach",
    legend_title="Gender",
    plot_bgcolor="white",
    boxmode='group'  # Group together boxes of the different genders for easier comparison
)

# Optionally, customize the boxplot colors
fig.update_traces(marker=dict(line=dict(width=2)),
                  selector=dict(type='box'))

# Show the plot
fig.show()

# PLOT IDEAS FOR THE DASHBOARD:
# 1. Gender and Age Group Reach - barcharts, heatmap (for showing all together)
# 2. Reach by Location - map? or barchart
# 3. Ad Creative Analysis - word cloud, most common words/topics
# 4. Time Series Analysis - line chart over time using ad_delivery_start_time and ad_delivery_stop_time
# 5. Ads Activity Duration - histogram, show the distribution
# 6. Page Activity - barchart, could also show like a number of unique page ids
# 7. Ad Engagement by Target Gender/Age - compare with the original target gender and age 