import os
from urllib.parse import urlparse
from fastapi import FastAPI
import joblib
import pandas as pd
import mysql.connector
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler

from dotenv import load_dotenv
load_dotenv() 

def establish_db_connection():
    print("Establishing connection...")
    print(os.getenv("DATABASE_URL"))
    print(os.getenv("SSL_CA_LOCATION"))
    
    database_url = os.getenv("DATABASE_URL")

    # Parse the database URL
    parsed_url = urlparse(database_url)
    username = parsed_url.username
    password = parsed_url.password
    hostname = parsed_url.hostname
    database_name = parsed_url.path.lstrip("/")

    # SSL configuration - based on your 'sslaccept=strict' parameter
    ssl_config = {
        "use_pure": True,
        "ssl_verify_cert": True,
        'ssl_ca': os.getenv("SSL_CA_LOCATION"),
    }

    # Establishing the connection
    conn = mysql.connector.connect(
        host=hostname,
        database=database_name,
        user=username,
        passwd=password,
        **ssl_config,
    )
    return conn

def create_df():

    conn = establish_db_connection()

    # Check if the connection is established
    print(conn.is_connected())

    # # Create a cursor object
    # cursor = conn.cursor()

    # Executing a query to select all contents from the Activity table  # TODO: Make this get all user/activity combinations (not only those with a rating/some interaction)  (need all values for label encoders)
    query = """
    SELECT
        u.id AS UserId,
        a.id AS ActivityId,
        a.name AS ActivityName,
        a.latitude AS Latitude,
        a.longitude AS Longitude,
        r.rating AS RatingScore,
        CASE
            WHEN `l`.id IS NOT NULL AND `l`.liked = TRUE THEN 1
            WHEN `l`.id IS NOT NULL AND `l`.liked = FALSE THEN -1
            ELSE 0
        END AS LikeStatus,
        CASE
            WHEN sa.A IS NOT NULL AND sa.B IS NOT NULL THEN 1
            ELSE 0
        END AS SaveStatus,
        GROUP_CONCAT(c.name SEPARATOR ', ') AS Categories
    FROM
        User u
    CROSS JOIN
        Activity a
    LEFT JOIN
        Rating r ON u.id = r.userId AND a.id = r.activityId
    LEFT JOIN
        `Like` `l` ON u.id = `l`.userId AND a.id = `l`.activityId
    LEFT JOIN
        _SavedActivities sa ON u.id = sa.A AND a.id = sa.B
    LEFT JOIN
        _ActivityToCategory atc ON a.id = atc.A
    LEFT JOIN
        Category c ON atc.B = c.id
    GROUP BY
        u.id, a.id, a.name, a.latitude, a.longitude, r.rating, `l`.id, sa.A, sa.B

        """

    df = pd.read_sql(query, conn)
    conn.close()
    
    categories = df['Categories'].str.get_dummies(sep=', ')
    df = pd.concat([df, categories], axis=1)

    # Save the columns created by get_dummies
    with open('files/category_columns.txt', 'w') as file:
        file.write(','.join(categories.columns))

    def calculate_score(row):
        score = 0
        # RatingScore conditions
        if not pd.isna(row['RatingScore']):
            if row['RatingScore'] <= 1:
                score -= 10
            elif row['RatingScore'] <= 2:
                score -= 7
            elif row['RatingScore'] <= 3:
                score -= 3
            elif row['RatingScore'] <= 4:
                score += 6
            elif row['RatingScore'] <= 5:
                score += 10

        # LikeStatus conditions
        if row['LikeStatus'] == 1:
            score += 4
        elif row['LikeStatus'] == -1:
            score -= 4

        # SaveStatus conditions
        if row['SaveStatus'] == 1:
            score += 7

        return score

    # Apply the function to each row
    df['Score'] = df.apply(calculate_score, axis=1)

    df.drop(['RatingScore', 'LikeStatus', 'SaveStatus', 'Categories', 'ActivityName'], axis=1, inplace=True)
    
    lat_long_columns = ['Latitude', 'Longitude']
    score_column = ['Score']

    # Create separate scalers for lat-long and score
    scaler_lat_long = StandardScaler()
    scaler_score = StandardScaler()

    # Fit and transform the data using the respective scalers
    df[lat_long_columns] = scaler_lat_long.fit_transform(df[lat_long_columns])
    df[score_column] = scaler_score.fit_transform(df[score_column])

    # Save the scalers
    joblib.dump(scaler_lat_long, 'files/scaler_lat_long.joblib')
    joblib.dump(scaler_score, 'files/scaler_score.joblib')
    
    label_encoder_user = LabelEncoder()
    label_encoder_activity = LabelEncoder()

    # Fit and transform the data using the respective encoders
    df['UserId'] = label_encoder_user.fit_transform(df['UserId'])
    df['ActivityId'] = label_encoder_activity.fit_transform(df['ActivityId'])

    # Save the label encoders
    joblib.dump(label_encoder_user, 'files/label_encoder_user.joblib')
    joblib.dump(label_encoder_activity, 'files/label_encoder_activity.joblib')

    return df


def create_df_predict(user_id, activity_ids_str):
    
    scaler_lat_long = joblib.load("files/scaler_lat_long.joblib")
    label_encoder_user = joblib.load("files/label_encoder_user.joblib")
    label_encoder_activity = joblib.load("files/label_encoder_activity.joblib")
    
    with open("files/category_columns.txt", "r") as file:
        category_columns = file.read().split(",")
    
    conn = establish_db_connection()

    # Modify the SQL query
    query = f"""
    SELECT
        u.id AS UserId,
        a.id AS ActivityId,
        a.name AS ActivityName,
        a.latitude AS Latitude,
        a.longitude AS Longitude,
        r.rating AS RatingScore,
        CASE
            WHEN `l`.id IS NOT NULL AND `l`.liked = TRUE THEN 1
            WHEN `l`.id IS NOT NULL AND `l`.liked = FALSE THEN -1
            ELSE 0
        END AS LikeStatus,
        CASE
            WHEN sa.A IS NOT NULL AND sa.B IS NOT NULL THEN 1
            ELSE 0
        END AS SaveStatus,
        GROUP_CONCAT(c.name SEPARATOR ', ') AS Categories
    FROM
        User u
    CROSS JOIN
        Activity a
    LEFT JOIN
        Rating r ON u.id = r.userId AND a.id = r.activityId
    LEFT JOIN
        `Like` `l` ON u.id = `l`.userId AND a.id = `l`.activityId
    LEFT JOIN
        _SavedActivities sa ON u.id = sa.A AND a.id = sa.B
    LEFT JOIN
        _ActivityToCategory atc ON a.id = atc.A
    LEFT JOIN
        Category c ON atc.B = c.id
    WHERE
    u.id = {user_id} AND a.id IN ({activity_ids_str})
    GROUP BY
        u.id, a.id, a.name, a.latitude, a.longitude, r.rating, `l`.id, sa.A, sa.B
    """

    # Execute the query
    df = pd.read_sql(query, conn)

    # Apply get_dummies and ensure the same columns as in training
    input_categories = df["Categories"].str.get_dummies(sep=", ")
    for col in category_columns:
        if col not in input_categories:
            input_categories[col] = 0
    df = pd.concat([df, input_categories[category_columns]], axis=1)

    df.drop(
        ["RatingScore", "LikeStatus", "SaveStatus", "Categories", "ActivityName"],
        axis=1,
        inplace=True,
    )

    columns_to_scale = ["Latitude", "Longitude"]

    df[columns_to_scale] = scaler_lat_long.transform(df[columns_to_scale])
    df["UserId"] = label_encoder_user.transform(df["UserId"])
    df["ActivityId"] = label_encoder_activity.transform(df["ActivityId"])
    
    return df