from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import MEstimateEncoder, TargetEncoder

import numpy as np
import pandas as pd

# categorical_cols = [
# 'id',
# # 'timestamp',
# # 'month',
# # 'year',
# 'floor',
# 'material',
# 'max_floor',
# # 'build_year',
# 'state',
# 'product_type',
# 'sub_area',
# 'ID_railroad_station_walk',
# 'ID_railroad_station_avto',
# 'water_1line',
# 'ID_big_road1',
# 'big_road1_1line',
# 'ID_big_road2',
# 'railroad_1line',
# 'ID_railroad_terminal',
# 'ID_bus_terminal',
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
# 'ID_metro',
# ]

categorical_cols = [
    "product_type",
    "material",
    "sub_area",
    "culture_objects_top_25",
    "thermal_power_plant_raion",
    "incineration_raion",
    "oil_chemistry_raion",
    "radiation_raion",
    "railroad_terminal_raion",
    "big_market_raion",
    "nuclear_reactor_raion",
    "detention_facility_raion",
    "water_1line",
    "big_road1_1line",
    "railroad_1line",
    "ecology",
]

# ordinal
ordinal_cols = [
    "ecology",
    "sub_area",
    "product_type",
    "culture_objects_top_25",
    "thermal_power_plant_raion",
    "incineration_raion",
    "oil_chemistry_raion",
    "radiation_raion",
    "railroad_terminal_raion",
    "big_market_raion",
    "nuclear_reactor_raion",
    "detention_facility_raion",
    "water_1line",
    "big_road1_1line",
    "railroad_1line",
    "material",
]

# Target Encoding
target_encoding_cols = [
    "sub_area",
]

# one hot
one_hot_cols = [
    "ecology",
    "sub_area",
    "product_type",
    "culture_objects_top_25",
    "thermal_power_plant_raion",
    "incineration_raion",
    "oil_chemistry_raion",
    "radiation_raion",
    "railroad_terminal_raion",
    "big_market_raion",
    "nuclear_reactor_raion",
    "detention_facility_raion",
    "water_1line",
    "big_road1_1line",
    "railroad_1line",
    "material",
]


useless_cols = [
    "id",
    'timestamp',
    # 'floor',
    # 'material',
    # 'max_floor',
    # 'build_year',
    # 'state',
    # 'product_type',
    # 'sub_area',
    "ID_railroad_station_walk",
    "ID_railroad_station_avto",
    # 'water_1line',
    "ID_big_road1",
    # 'big_road1_1line',
    "ID_big_road2",
    # 'railroad_1line',
    "ID_railroad_terminal",
    "ID_bus_terminal",
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
    "ID_metro",
]


numeric_cols = [
    "build_year",
    "full_sq",
    "life_sq",
    "num_room",
    "kitch_sq",
    "area_m",
    "raion_popul",
    "green_zone_part",
    "indust_part",
    "children_preschool",
    "preschool_quota",
    "preschool_education_centers_raion",
    "children_school",
    "school_quota",
    "school_education_centers_raion",
    "school_education_centers_top_20_raion",
    "hospital_beds_raion",
    "healthcare_centers_raion",
    "university_top_20_raion",
    "sport_objects_raion",
    "additional_education_raion",
    "culture_objects_top_25_raion",
    "shopping_centers_raion",
    "office_raion",
    "full_all",
    "male_f",
    "female_f",
    "young_all",
    "young_male",
    "young_female",
    "work_all",
    "work_male",
    "work_female",
    "ekder_all",
    "ekder_male",
    "ekder_female",
    "0_6_all",
    "0_6_male",
    "0_6_female",
    "7_14_all",
    "7_14_male",
    "7_14_female",
    "0_17_all",
    "0_17_male",
    "0_17_female",
    "16_29_all",
    "16_29_male",
    "16_29_female",
    "0_13_all",
    "0_13_male",
    "0_13_female",
    "raion_build_count_with_material_info",
    "build_count_block",
    "build_count_wood",
    "build_count_frame",
    "build_count_brick",
    "build_count_monolith",
    "build_count_panel",
    "build_count_foam",
    "build_count_slag",
    "build_count_mix",
    "raion_build_count_with_builddate_info",
    "build_count_before_1920",
    "build_count_1921-1945",
    "build_count_1946-1970",
    "build_count_1971-1995",
    "build_count_after_1995",
    "metro_min_avto",
    "metro_km_avto",
    "metro_min_walk",
    "metro_km_walk",
    "kindergarten_km",
    "school_km",
    "park_km",
    "green_zone_km",
    "industrial_km",
    "water_treatment_km",
    "cemetery_km",
    "incineration_km",
    "railroad_station_walk_km",
    "railroad_station_walk_min",
    "railroad_station_avto_km",
    "railroad_station_avto_min",
    "public_transport_station_km",
    "public_transport_station_min_walk",
    "water_km",
    "mkad_km",
    "ttk_km",
    "sadovoe_km",
    "bulvar_ring_km",
    "kremlin_km",
    "big_road1_km",
    "big_road2_km",
    "railroad_km",
    "zd_vokzaly_avto_km",
    "bus_terminal_avto_km",
    "oil_chemistry_km",
    "nuclear_reactor_km",
    "radiation_km",
    "power_transmission_line_km",
    "thermal_power_plant_km",
    "ts_km",
    "big_market_km",
    "market_shop_km",
    "fitness_km",
    "swim_pool_km",
    "ice_rink_km",
    "stadium_km",
    "basketball_km",
    "hospice_morgue_km",
    "detention_facility_km",
    "public_healthcare_km",
    "university_km",
    "workplaces_km",
    "shopping_centers_km",
    "office_km",
    "additional_education_km",
    "preschool_km",
    "big_church_km",
    "church_synagogue_km",
    "mosque_km",
    "theater_km",
    "museum_km",
    "exhibition_km",
    "catering_km",
    "green_part_500",
    "prom_part_500",
    "office_count_500",
    "office_sqm_500",
    "trc_count_500",
    "trc_sqm_500",
    "cafe_count_500",
    "cafe_sum_500_min_price_avg",
    "cafe_sum_500_max_price_avg",
    "cafe_avg_price_500",
    "cafe_count_500_na_price",
    "cafe_count_500_price_500",
    "cafe_count_500_price_1000",
    "cafe_count_500_price_1500",
    "cafe_count_500_price_2500",
    "cafe_count_500_price_4000",
    "cafe_count_500_price_high",
    "big_church_count_500",
    "church_count_500",
    "mosque_count_500",
    "leisure_count_500",
    "sport_count_500",
    "market_count_500",
    "green_part_1000",
    "prom_part_1000",
    "office_count_1000",
    "office_sqm_1000",
    "trc_count_1000",
    "trc_sqm_1000",
    "cafe_count_1000",
    "cafe_sum_1000_min_price_avg",
    "cafe_sum_1000_max_price_avg",
    "cafe_avg_price_1000",
    "cafe_count_1000_na_price",
    "cafe_count_1000_price_500",
    "cafe_count_1000_price_1000",
    "cafe_count_1000_price_1500",
    "cafe_count_1000_price_2500",
    "cafe_count_1000_price_4000",
    "cafe_count_1000_price_high",
    "big_church_count_1000",
    "church_count_1000",
    "mosque_count_1000",
    "leisure_count_1000",
    "sport_count_1000",
    "market_count_1000",
    "green_part_1500",
    "prom_part_1500",
    "office_count_1500",
    "office_sqm_1500",
    "trc_count_1500",
    "trc_sqm_1500",
    "cafe_count_1500",
    "cafe_sum_1500_min_price_avg",
    "cafe_sum_1500_max_price_avg",
    "cafe_avg_price_1500",
    "cafe_count_1500_na_price",
    "cafe_count_1500_price_500",
    "cafe_count_1500_price_1000",
    "cafe_count_1500_price_1500",
    "cafe_count_1500_price_2500",
    "cafe_count_1500_price_4000",
    "cafe_count_1500_price_high",
    "big_church_count_1500",
    "church_count_1500",
    "mosque_count_1500",
    "leisure_count_1500",
    "sport_count_1500",
    "market_count_1500",
    "green_part_2000",
    "prom_part_2000",
    "office_count_2000",
    "office_sqm_2000",
    "trc_count_2000",
    "trc_sqm_2000",
    "cafe_count_2000",
    "cafe_sum_2000_min_price_avg",
    "cafe_sum_2000_max_price_avg",
    "cafe_avg_price_2000",
    "cafe_count_2000_na_price",
    "cafe_count_2000_price_500",
    "cafe_count_2000_price_1000",
    "cafe_count_2000_price_1500",
    "cafe_count_2000_price_2500",
    "cafe_count_2000_price_4000",
    "cafe_count_2000_price_high",
    "big_church_count_2000",
    "church_count_2000",
    "mosque_count_2000",
    "leisure_count_2000",
    "sport_count_2000",
    "market_count_2000",
    "green_part_3000",
    "prom_part_3000",
    "office_count_3000",
    "office_sqm_3000",
    "trc_count_3000",
    "trc_sqm_3000",
    "cafe_count_3000",
    "cafe_sum_3000_min_price_avg",
    "cafe_sum_3000_max_price_avg",
    "cafe_avg_price_3000",
    "cafe_count_3000_na_price",
    "cafe_count_3000_price_500",
    "cafe_count_3000_price_1000",
    "cafe_count_3000_price_1500",
    "cafe_count_3000_price_2500",
    "cafe_count_3000_price_4000",
    "cafe_count_3000_price_high",
    "big_church_count_3000",
    "church_count_3000",
    "mosque_count_3000",
    "leisure_count_3000",
    "sport_count_3000",
    "market_count_3000",
    "green_part_5000",
    "prom_part_5000",
    "office_count_5000",
    "office_sqm_5000",
    "trc_count_5000",
    "trc_sqm_5000",
    "cafe_count_5000",
    "cafe_sum_5000_min_price_avg",
    "cafe_sum_5000_max_price_avg",
    "cafe_avg_price_5000",
    "cafe_count_5000_na_price",
    "cafe_count_5000_price_500",
    "cafe_count_5000_price_1000",
    "cafe_count_5000_price_1500",
    "cafe_count_5000_price_2500",
    "cafe_count_5000_price_4000",
    "cafe_count_5000_price_high",
    "big_church_count_5000",
    "church_count_5000",
    "mosque_count_5000",
    "leisure_count_5000",
    "sport_count_5000",
    "market_count_5000",
    "price_doc",
]


def get_na_cols(df):
    return df.columns[df.isna().any()].tolist()


def process_timestamp(**kwargs):
    df = kwargs["df"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # df.drop(["timestamp"], axis=1, inplace=True)

    return df


def convert_to_cat_and_drop_useless_cols(**kwargs):
    df = kwargs["df"]
    for c in categorical_cols:
        df[c] = df[c].astype("category")
    
    df.drop(useless_cols, axis=1, inplace=True)

    return df

def replace_nan_with_mean(**kwargs):
    df = kwargs["df"]
    train_df = kwargs.get("train_df", None)

    cols_with_na = get_na_cols(df)

    for c in cols_with_na:
        if (
            c not in useless_cols
            and df[c].dtype.name != "category"
            # and c not in ["floor", "max_floor"]
        ):
            if train_df is None:
                df[c] = df[c].fillna((df[c].mean()))
            else:
                df[c] = df[c].fillna((train_df[c].mean()))

    return df

def update_price_doc(**kwargs):
    df = kwargs['df']
    # df['price_doc'] = df['price_doc'] * .969 + 10
    # Investment
    df['price_doc'] = df['price_doc'] * 1.05
    # Owner Occupier
    # df['price_doc'] = df['price_doc'] * .9

    return df

def update_price_doc_with_price_index(**kwargs):
    # df = process_timestamp(**kwargs)
    df = kwargs['df']
    price_indexes = pd.read_csv('price_index.csv', index_col=0)
    price_indexes_dic = price_indexes.to_dict()
    for year, vals in price_indexes_dic.items():
        for sub_area, price_index in vals.items():
            df.loc[(df['sub_area'] == sub_area) & (df['year'] == int(year)), 'price_index'] = price_index
    
    df['price_doc'] = df['price_doc'] * df['price_index']

    # Remove columns
    df.drop(["month","year","price_index"], axis=1, inplace=True)

    return df


def cat_encode():
    pass


def process_train(
    train_df, one_hot_encoder=None, ordinal_encoder=None, target_encoder=None
):
    # Remove outlier build year
    df = train_df.drop(train_df[train_df["build_year"] > 2019].index)

    pipeline = [
        # process_timestamp,
        convert_to_cat_and_drop_useless_cols,
        replace_nan_with_mean,
    ]
    for f in pipeline:
        df = f(df=df)

    # One Hot Encoding
    if one_hot_encoder is None:
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        one_hot_encoder.fit(df[one_hot_cols])

    _one_hot_transform(df, one_hot_encoder)

    y = df["price_doc"]
    X = df.drop(["price_doc"], axis=1)


    df_new = pd.concat([X, y], axis=1)

    # df_new[["year", "build_year"]] = df_new[["year", "build_year"]].astype("int64")
    df_new[["build_year"]] = df_new[["build_year"]].astype("int64")

    # Handle Na
    # df_new.dropna(inplace=True)

    # Reset index
    df_new.reset_index(drop=True, inplace=True)

    return df_new, ordinal_encoder, target_encoder, one_hot_encoder


def process_test(
    df_,
    train_processed,
    ordinal_encoder=None,
    target_encoder=None,
    one_hot_encoder=None,
    recombine_id=False,
):
    df = df_.copy()
    pipeline = [
        # process_timestamp,
        convert_to_cat_and_drop_useless_cols,
        replace_nan_with_mean,
    ]
    for f in pipeline:
        df = f(df=df, train_df=train_processed)

    # One Hot Encoding
    if one_hot_encoder is not None:
        _one_hot_transform(df, one_hot_encoder)
    # Ordinal
    if ordinal_encoder is not None:
        df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])

    # Recombine id
    if recombine_id:
        df = pd.concat([df_["id"], df], axis=1)

    return df


def _one_hot_transform(df, one_hot_encoder):
    encoded_cols = list(one_hot_encoder.get_feature_names_out(one_hot_cols))
    df[encoded_cols] = one_hot_encoder.transform(df[one_hot_cols])
    df.drop(one_hot_cols, axis=1, inplace=True)


# K folds Functions
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import shap

# Inv Seed = 100
# Own Seed = 42
def cal_mean_errors(train_df, models, features=None, seed=42):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmsles = []
    most_important_features = []

    # get one hot encoder
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoder.fit(train_df[one_hot_cols])

    # get ordinal encoder
    # ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    # ordinal_encoder.fit(train_df[ordinal_cols])

    # processed_df, _, _, _ = process_train(train_df, one_hot_encoder=one_hot_encoder)
    train_df = train_df.drop(train_df[train_df["build_year"] > 2019].id)
    train_df.reset_index(drop=True, inplace=True)

    for fold, (train_idx, test_idx) in tqdm(enumerate(cv.split(train_df))):
        rmsles_ = []
        most_important_features_ = []

        # Splitting data
        tmp_df = train_df.iloc[train_idx]
        processed_df, ordinal_encoder, target_encoder, one_hot_encoder = process_train(
            tmp_df, one_hot_encoder=one_hot_encoder
        )  # , ordinal_encoder=ordinal_encoder)
        X_train = processed_df.drop(["price_doc"], axis=1)
        y_train = processed_df["price_doc"]

        X_test = train_df.iloc[test_idx]
        y_test = X_test["price_doc"]
        X_test = X_test.drop(["price_doc"], axis=1)
        X_test = process_test(
            X_test,
            X_train,
            one_hot_encoder=one_hot_encoder,
            ordinal_encoder=ordinal_encoder,
        )

        # Select features is specified
        if features is not None:
            X_train = X_train[features]
            X_test = X_test[features]

        for model in models:
            evaluation = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=evaluation, verbose=False)
            pred = model.predict(X_test)
            rmsles_.append(mean_squared_error(y_test, pred, squared=False))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Average across samples
            most_important_features_.append(np.abs(shap_values).mean(axis=0))

        rmsles.append(rmsles_)
        most_important_features.append(most_important_features_)

    # Average across folds
    average_feature_importance = np.array(most_important_features).mean(axis=0)
    important_features = []

    for average_val in average_feature_importance:
        feature_names = X_test.columns
        shap_importance = pd.DataFrame(
            list(zip(feature_names, average_val)),
            columns=["col_name", "feature_importance_vals"],
        )
        shap_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )

        important_features.append(shap_importance)

    # Mean rmsles
    rmsles = np.array(rmsles).mean(axis=0)

    return rmsles, important_features
