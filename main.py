import json
from fastapi import FastAPI, Request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import uvicorn
import xgboost
from mangum import Mangum

from utils import create_df, create_df_predict

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/retrain")
async def train_model():
    df = create_df()

    X = df.drop("Score", axis=1)
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgboost.XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    joblib.dump(model, "files/model.joblib")

    return {"message": "Model trained and saved successfully!"}


@app.post("/predict")
async def predict(request: Request):

    data = await request.json()

    print(data)

    user_id = f"'{data['userId']}'"
    activity_ids = data["activityIds"]
    activity_ids_str = ", ".join([f"'{id}'" for id in activity_ids])

    df = create_df_predict(user_id, activity_ids_str)

    scaler_score = joblib.load("files/scaler_score.joblib")
    label_encoder_activity = joblib.load("files/label_encoder_activity.joblib")
    model = joblib.load("files/model.joblib")

    predictions = model.predict(df)
    predictions = scaler_score.inverse_transform(predictions.reshape(-1, 1))

    activity_ids = df["ActivityId"]
    original_activity_ids = label_encoder_activity.inverse_transform(activity_ids)

    results_df = pd.DataFrame(
        {"ActivityId": original_activity_ids, "Score": predictions.flatten()}
    )

    prediction_output = dict(
        zip(results_df["ActivityId"].astype(str), results_df["Score"])
    )

    return json.dumps(prediction_output)


handler = Mangum(app)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
