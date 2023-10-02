from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import MEstimateEncoder, TargetEncoder

import numpy as np
import pandas as pd

categorical_cols = [
'id',
# 'timestamp',
# 'month',
# 'year',
'floor',
'material',
'max_floor',
# 'build_year',
'state',
'product_type',
'sub_area',
'ID_railroad_station_walk',
'ID_railroad_station_avto',
'water_1line',
'ID_big_road1',
'big_road1_1line',
'ID_big_road2',
'railroad_1line',
'ID_railroad_terminal',
'ID_bus_terminal',
'ecology',
'nuclear_reactor_raion',
'culture_objects_top_25',
'thermal_power_plant_raion',
'incineration_raion',
'oil_chemistry_raion',
'radiation_raion',
'railroad_terminal_raion',
'big_market_raion',
'nuclear_reactor_raion',
'detention_facility_raion',
'ID_metro',
]

# one hot
one_hot_cols = [
    'product_type',
]

# Target Encoding
target_encoding_cols = [
    'sub_area',
]

# label encoding
ordinal_cols = [
'sub_area',
'product_type',



'ecology',
'floor',
'culture_objects_top_25',
'thermal_power_plant_raion',
'incineration_raion',
'oil_chemistry_raion',
'radiation_raion',
'railroad_terminal_raion',
'big_market_raion',
'nuclear_reactor_raion',
'detention_facility_raion',
'water_1line',
'big_road1_1line',
'railroad_1line',
'month',
# 'year',
'material',
'max_floor',
# 'build_year',
'state',
]


useless_cols = [
'id',
# 'timestamp',
# 'floor',
# 'material',
# 'max_floor',
# 'build_year',
# 'state',
# 'product_type',
# 'sub_area',
'ID_railroad_station_walk',
'ID_railroad_station_avto',
# 'water_1line',
'ID_big_road1',
# 'big_road1_1line',
'ID_big_road2',
# 'railroad_1line',
'ID_railroad_terminal',
'ID_bus_terminal',
# 'ecology',
# 'nuclear_reactor_raion',
# 'culture_objects_top_25',
# 'thermal_power_plant_raion',
# 'incineration_raion',
# 'oil_chemistry_raion',
# 'radiation_raion',
# 'railroad_terminal_raion',
# 'big_market_raion',
# 'nuclear_reactor_raion',
# 'detention_facility_raion',
'ID_metro',

]


numeric_cols =[
'build_year'
'full_sq',
'life_sq',
'num_room',
'kitch_sq',
'area_m',
'raion_popul',
'green_zone_part',
'indust_part',
'children_preschool',
'preschool_quota',
'preschool_education_centers_raion',
'children_school',
'school_quota',
'school_education_centers_raion',
'school_education_centers_top_20_raion',
'hospital_beds_raion',
'healthcare_centers_raion',
'university_top_20_raion',
'sport_objects_raion',
'additional_education_raion',
'culture_objects_top_25_raion',
'shopping_centers_raion',
'office_raion',
'full_all',
'male_f',
'female_f',
'young_all',
'young_male',
'young_female',
'work_all',
'work_male',
'work_female',
'ekder_all',
'ekder_male',
'ekder_female',
'0_6_all',
'0_6_male',
'0_6_female',
'7_14_all',
'7_14_male',
'7_14_female',
'0_17_all',
'0_17_male',
'0_17_female',
'16_29_all',
'16_29_male',
'16_29_female',
'0_13_all',
'0_13_male',
'0_13_female',
'raion_build_count_with_material_info',
'build_count_block',
'build_count_wood',
'build_count_frame',
'build_count_brick',
'build_count_monolith',
'build_count_panel',
'build_count_foam',
'build_count_slag',
'build_count_mix',
'raion_build_count_with_builddate_info',
'build_count_before_1920',
'build_count_1921-1945',
'build_count_1946-1970',
'build_count_1971-1995',
'build_count_after_1995',
'metro_min_avto',
'metro_km_avto',
'metro_min_walk',
'metro_km_walk',
'kindergarten_km',
'school_km',
'park_km',
'green_zone_km',
'industrial_km',
'water_treatment_km',
'cemetery_km',
'incineration_km',
'railroad_station_walk_km',
'railroad_station_walk_min',
'railroad_station_avto_km',
'railroad_station_avto_min',
'public_transport_station_km',
'public_transport_station_min_walk',
'water_km',
'mkad_km',
'ttk_km',
'sadovoe_km',
'bulvar_ring_km',
'kremlin_km',
'big_road1_km',
'big_road2_km',
'railroad_km',
'zd_vokzaly_avto_km',
'bus_terminal_avto_km',
'oil_chemistry_km',
'nuclear_reactor_km',
'radiation_km',
'power_transmission_line_km',
'thermal_power_plant_km',
'ts_km',
'big_market_km',
'market_shop_km',
'fitness_km',
'swim_pool_km',
'ice_rink_km',
'stadium_km',
'basketball_km',
'hospice_morgue_km',
'detention_facility_km',
'public_healthcare_km',
'university_km',
'workplaces_km',
'shopping_centers_km',
'office_km',
'additional_education_km',
'preschool_km',
'big_church_km',
'church_synagogue_km',
'mosque_km',
'theater_km',
'museum_km',
'exhibition_km',
'catering_km',
'green_part_500',
'prom_part_500',
'office_count_500',
'office_sqm_500',
'trc_count_500',
'trc_sqm_500',
'cafe_count_500',
'cafe_sum_500_min_price_avg',
'cafe_sum_500_max_price_avg',
'cafe_avg_price_500',
'cafe_count_500_na_price',
'cafe_count_500_price_500',
'cafe_count_500_price_1000',
'cafe_count_500_price_1500',
'cafe_count_500_price_2500',
'cafe_count_500_price_4000',
'cafe_count_500_price_high',
'big_church_count_500',
'church_count_500',
'mosque_count_500',
'leisure_count_500',
'sport_count_500',
'market_count_500',
'green_part_1000',
'prom_part_1000',
'office_count_1000',
'office_sqm_1000',
'trc_count_1000',
'trc_sqm_1000',
'cafe_count_1000',
'cafe_sum_1000_min_price_avg',
'cafe_sum_1000_max_price_avg',
'cafe_avg_price_1000',
'cafe_count_1000_na_price',
'cafe_count_1000_price_500',
'cafe_count_1000_price_1000',
'cafe_count_1000_price_1500',
'cafe_count_1000_price_2500',
'cafe_count_1000_price_4000',
'cafe_count_1000_price_high',
'big_church_count_1000',
'church_count_1000',
'mosque_count_1000',
'leisure_count_1000',
'sport_count_1000',
'market_count_1000',
'green_part_1500',
'prom_part_1500',
'office_count_1500',
'office_sqm_1500',
'trc_count_1500',
'trc_sqm_1500',
'cafe_count_1500',
'cafe_sum_1500_min_price_avg',
'cafe_sum_1500_max_price_avg',
'cafe_avg_price_1500',
'cafe_count_1500_na_price',
'cafe_count_1500_price_500',
'cafe_count_1500_price_1000',
'cafe_count_1500_price_1500',
'cafe_count_1500_price_2500',
'cafe_count_1500_price_4000',
'cafe_count_1500_price_high',
'big_church_count_1500',
'church_count_1500',
'mosque_count_1500',
'leisure_count_1500',
'sport_count_1500',
'market_count_1500',
'green_part_2000',
'prom_part_2000',
'office_count_2000',
'office_sqm_2000',
'trc_count_2000',
'trc_sqm_2000',
'cafe_count_2000',
'cafe_sum_2000_min_price_avg',
'cafe_sum_2000_max_price_avg',
'cafe_avg_price_2000',
'cafe_count_2000_na_price',
'cafe_count_2000_price_500',
'cafe_count_2000_price_1000',
'cafe_count_2000_price_1500',
'cafe_count_2000_price_2500',
'cafe_count_2000_price_4000',
'cafe_count_2000_price_high',
'big_church_count_2000',
'church_count_2000',
'mosque_count_2000',
'leisure_count_2000',
'sport_count_2000',
'market_count_2000',
'green_part_3000',
'prom_part_3000',
'office_count_3000',
'office_sqm_3000',
'trc_count_3000',
'trc_sqm_3000',
'cafe_count_3000',
'cafe_sum_3000_min_price_avg',
'cafe_sum_3000_max_price_avg',
'cafe_avg_price_3000',
'cafe_count_3000_na_price',
'cafe_count_3000_price_500',
'cafe_count_3000_price_1000',
'cafe_count_3000_price_1500',
'cafe_count_3000_price_2500',
'cafe_count_3000_price_4000',
'cafe_count_3000_price_high',
'big_church_count_3000',
'church_count_3000',
'mosque_count_3000',
'leisure_count_3000',
'sport_count_3000',
'market_count_3000',
'green_part_5000',
'prom_part_5000',
'office_count_5000',
'office_sqm_5000',
'trc_count_5000',
'trc_sqm_5000',
'cafe_count_5000',
'cafe_sum_5000_min_price_avg',
'cafe_sum_5000_max_price_avg',
'cafe_avg_price_5000',
'cafe_count_5000_na_price',
'cafe_count_5000_price_500',
'cafe_count_5000_price_1000',
'cafe_count_5000_price_1500',
'cafe_count_5000_price_2500',
'cafe_count_5000_price_4000',
'cafe_count_5000_price_high',
'big_church_count_5000',
'church_count_5000',
'mosque_count_5000',
'leisure_count_5000',
'sport_count_5000',
'market_count_5000',
'price_doc',    
]


xgb_params = {
    'n_estimators': 500,
    'device': 'cuda',
    # 'random_state': 24,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmsle',
    'enable_categorical': True,
    'early_stopping_rounds': 50,

    'colsample_bytree': 0.8,
    'eta': 0.05,
    'gamma': 8.084029345968737,
    'max_depth': 6,
    'min_child_weight': 2.0,
    'reg_alpha': 93.0, 
    'reg_lambda': 0.8685796539747039
}


def get_na_cols(df):
    return df.columns[df.isna().any()].tolist()

def preprocess_1(df_):
    df = df_.copy()
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    df.drop(['timestamp'], axis=1, inplace=True)

    for c in categorical_cols:
        df[c] = df[c].astype('category')

    # Replacing NAN with mean
    df.drop(useless_cols, axis=1, inplace=True)
    
    cols_with_na = get_na_cols(df)

    for c in cols_with_na:
        if c not in useless_cols and df[c].dtype.name != 'category':
            df[c] = df[c].fillna((df[c].mean()))

    return df

def cat_encode(df_, target_encoder, ordinal_encoder):
    df = df_.copy()
    # Drop price doc if found
    try:
        df.drop(['price_doc'], axis=1, inplace=True)
    except:
        print("price_doc not found")

    # One Hot Encoding
    # df = pd.get_dummies(df, columns=one_hot_cols, dtype='int64')

    
    # Ordinal
    df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])

    # Target
    # df['sub_area_te'] = target_encoder.transform(df)['sub_area']

    # Label Encoding
    # df['sub_area'] = target_encoder.transform(df['sub_area'])

    # df.drop(['sub_area'], axis=1, inplace=True)

    return df    

def process_train(train_df):
    tdf = train_df.drop(train_df[train_df['build_year'] > 2019].index)

 
    processed_1_train_df = preprocess_1(tdf)
    df = processed_1_train_df.copy()


    # One Hot Encoding
    # one_hot = OneHotEncoder(handle_unknown='ignore')
    # one_hot.fit(df[one_hot_cols])
    # df[one_hot_cols] = one_hot.transform(df[one_hot_cols])
    # df = pd.get_dummies(df, columns=one_hot_cols, dtype='int64')

    # Ordinal
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    oe.fit(df[ordinal_cols])
    df[ordinal_cols] = oe.transform(df[ordinal_cols])

    # TESTING
    # df.dropna(inplace=True)
    # df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    y = df['price_doc']
    X = df.drop(['price_doc'], axis=1)
    
    # Target
    # te_encoder = TargetEncoder(cols=target_encoding_cols, min_samples_leaf=5, smoothing=8)
    # te_encoder.fit(X, y)

    # X['sub_area_te'] = te_encoder.transform(X)['sub_area']
    # X.drop(['sub_area'], axis=1, inplace=True)


    df_new = pd.concat([X, y], axis=1)

    # Handle Na
    # df_new.dropna(inplace=True)
    print("*"*100,'\n', df_new.shape)
    df_new[['year', 'build_year']] = df_new[['year', 'build_year']].astype('int64')

    return df_new, oe, None #te_encoder

def process(df_, te, oe):
    df = df_.copy()
    df = preprocess_1(df_)

    
    # Encode categories
    df = cat_encode(df, te, oe)
    
    return df

