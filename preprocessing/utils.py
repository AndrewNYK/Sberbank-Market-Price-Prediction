import numpy as np
import pandas as pd

top_40_cols = [
    'full_sq',
    'sub_area',
    'state',
    'num_room',
    'year',
    'floor',
    'build_year',
    'max_floor',
    'cafe_count_5000_price_2500',
    'kitch_sq',
    'life_sq',
    'cafe_count_2000',
    'sport_count_3000',
    'ttk_km',
    'public_healthcare_km',
    'cafe_count_3000',
    'product_type',
    'product_type_Investment',
    'cafe_count_5000_price_high',
    'cafe_count_3000_price_2500',
    'office_sqm_5000',
    'green_zone_km',
    'zd_vokzaly_avto_km',
    'cafe_count_3000_price_1500',
    'detention_facility_km',
    'railroad_km',
    'industrial_km',
    'additional_education_km',
    'metro_min_avto',
    'metro_min_walk',
    'month',
    'metro_km_avto',
    'swim_pool_km',
    'kindergarten_km',
    'thermal_power_plant_km',
    'workplaces_km',
    'preschool_km',
    'public_transport_station_km',
    'power_transmission_line_km',
    'cafe_count_2000_price_1000',
    'cafe_count_5000_na_price'
]

categorical_cols = ['floor', 'max_floor', 'state', 'product_type', 'sub_area', 'num_room', 'year', 'month']

def preprocess_train(df):
    # drop rows where life_sq and kitch_sq higher than full_sq
    df = df.drop(df[(df['full_sq'] <= df['life_sq'])].index)
    df = df.drop(df[(df['full_sq'] <= df['kitch_sq'])].index)

    # for max_floor, we could fill NaN with the median max_floor of properties in the same sub_area
    sub_area_medians = df.groupby('sub_area')['max_floor'].median().reset_index()
    # sub_area_medians['max_floor'] = np.ceil(sub_area_medians['max_floor'])
    df = df.merge(sub_area_medians, on='sub_area', suffixes=('', '_median'), how='left')
    df['max_floor'].fillna(df['max_floor_median'], inplace=True)
    df.drop(columns='max_floor_median', inplace=True)

    # and then for floor, we just fill NaN with the max_floor
    df['floor'].fillna(df['max_floor'], inplace=True)

    # finally we replace the max_floor with the floor, if there are any value of floor greater than max_floor(e.g row 63)
    df['max_floor'] = df.apply(lambda row: row['floor'] if row['floor'] > row['max_floor'] else row['max_floor'], axis=1)

    # we do the same for build_year, fill NaN with the median build_year of properties in the same sub_area
    sub_area_medians = df.groupby('sub_area')['build_year'].median().reset_index()
    # sub_area_medians['build_year'] = np.ceil(sub_area_medians['build_year'])
    df = df.merge(sub_area_medians, on='sub_area', suffixes=('', '_median'), how='left')
    df['build_year'].fillna(df['build_year_median'], inplace=True)
    df.drop(columns='build_year_median', inplace=True)

    # for num_room, we shall split the data into different ranges of full_sq value and calculate the average num_room for each range
    # then, we replace the num_room NaN values depending on which range the row's full_sq belongs to
    # Define the ranges for 'full_sq' bins
    bins = [0, 30, 52, 80, float('inf')]  # these values are eyeballed
    print(bins)

    # Use pd.cut to create bins for 'full_sq'
    df['full_sq_bins'] = pd.cut(df['full_sq'], bins=bins)

    # Calculate the average 'num_room' for each 'full_sq' range
    # num_room_averages = df.groupby('full_sq_bins')['num_room'].transform('mean')
    num_room_averages = df.groupby('full_sq_bins')['num_room'].transform(lambda x: np.ceil(x.mean()))
    df['num_room'].fillna(num_room_averages, inplace=True)
    df.drop(columns='full_sq_bins', inplace=True)

    # for kitch_sq, we shall group by sub_area and calculate the average kitch_sq/life_sq proportion, then replace NaN values with the proporiton multiplied by life_sq
    # Calculate the 'kitch_sq/life_sq' for each row
    df['kitch_sq_per_life_sq'] = df['kitch_sq'] / df['life_sq']

    # Calculate the average 'kitch_sq/life_sq' for each 'sub_area'
    sub_area_avg = df.groupby('sub_area')['kitch_sq_per_life_sq'].mean()

    # Apply the function to fill NaN values in 'kitch_sq'
    df['kitch_sq'] = df.apply(fill_kitch_sq, args=(sub_area_avg) ,axis=1)

    # Drop the 'kitch_sq_per_life_sq' column if you no longer need it
    df.drop(columns='kitch_sq_per_life_sq', inplace=True)

    # we do the same for state, fill NaN with the median state of properties in the same sub_area
    sub_area_medians = df.groupby('sub_area')['state'].median().reset_index()
    # sub_area_medians['build_year'] = np.ceil(sub_area_medians['build_year'])
    df = df.merge(sub_area_medians, on='sub_area', suffixes=('', '_median'), how='left')
    df['state'].fillna(df['state_median'], inplace=True)
    df.drop(columns='state_median', inplace=True)

    df_drop_categorical = df.drop(columns=categorical_cols, axis=1)
    # Remove rows with outliers
    df_no_outliers = remove_outliers_iqr(df_drop_categorical)
    # scaler = StandardScaler()
    # df_no_outliers_scaled = scaler.fit_transform(df_no_outliers)

    for c in categorical_cols:
        df_no_outliers[c] = df[c]

    for c in categorical_cols:
        df_no_outliers[c] = df_no_outliers[c].astype('category')

    df_no_outliers['price_doc'] = df["price_doc"] * .969 + 10

    return df_no_outliers

# Define a function to remove rows with outliers based on the IQR
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Define a function to fill NaN values in 'kitch_sq' based on 'sub_area'
def fill_kitch_sq(row, sub_area_avg):
    sub_area = row['sub_area']
    if pd.notna(row['kitch_sq']):
        return row['kitch_sq']
    if sub_area in sub_area_avg:
        return np.ceil(row['life_sq'] * sub_area_avg[sub_area])
    return row['kitch_sq']