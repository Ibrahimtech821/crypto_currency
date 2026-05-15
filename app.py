import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report



st.set_page_config(
    page_title="Crypto Movement Prediction",
    layout="wide"
)

st.title("Cross-Asset Cryptocurrency Movement Prediction")

st.write("""
This app uses the existing crypto dataset to train a model, then predicts the movement class
from values entered manually using sliders.
""")




df = pd.read_csv("final_crypto.csv")
df = df[~df["type"].isin(["BTC", "DOGE"])].copy()

st.subheader("Dataset Preview")
st.dataframe(df.head())




drop_cols = [
    'label',
    'date',
    'type',
    'down_days_7',
    'taker_buy_volume',
    'z_score',
    'taker_buy_ratio',
    'taker_sell_ratio',
    'negative_momentum',
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df["label"]




le = LabelEncoder()
y_encoded = le.fit_transform(y)




st.sidebar.header("Prediction Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "KNN Distance", "XGBoost"]
)

selected_coin = st.sidebar.selectbox(
    "Choose Coin",
    sorted(df["type"].unique())
)

train_mode = st.sidebar.selectbox(
    "Training Mode",
    [
        "Train on all coins",
        "Train on all coins except selected coin"
    ]
)

predict_button = st.sidebar.button("Predict Movement")




if train_mode == "Train on all coins":
    train_df = df.copy()

else:
    train_df = df[df["type"] != selected_coin].copy()

X_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df["label"]

y_train_encoded = le.transform(y_train)


# =========================
# USER INPUT USING SLIDERS
# =========================

st.subheader("Enter Feature Values")

st.write("""
Use the sliders to enter the market values for prediction.
The slider ranges are taken automatically from your dataset.
""")

user_input = {}

cols = st.columns(3)

for i, feature in enumerate(X.columns):
    col = cols[i % 3]

    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())

    if min_val == max_val:
        user_input[feature] = mean_val
    else:
        user_input[feature] = col.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

input_df = pd.DataFrame([user_input])


# =========================
# MODEL TRAINING + PREDICTION
# =========================

if predict_button:

    if model_choice == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=700,
            max_depth=13,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            criterion="entropy",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train_encoded)

        prediction_encoded = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)


    elif model_choice == "KNN Distance":

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        input_scaled = scaler.transform(input_df)

        model = KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train_encoded)

        prediction_encoded = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)


    elif model_choice == "XGBoost":

       
        model = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

        model.fit(X_train, y_train_encoded)

        prediction_encoded = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)


  

    
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    st.subheader("Final Prediction")
    st.success(f"The predicted movement is: {prediction_label}")

    
    proba_df = pd.DataFrame({
        "Class": le.classes_,
        "Probability": prediction_proba[0]
    })

    st.dataframe(proba_df)

    st.bar_chart(proba_df.set_index("Class"))




st.subheader("Model Information")

st.write("Selected model:", model_choice)
st.write("Selected coin:", selected_coin)
st.write("Training mode:", train_mode)

if train_mode == "Train on all coins except selected coin":
    test_df = df[df["type"] == selected_coin].copy()

    if len(test_df) > 0:

        X_test = test_df.drop(columns=drop_cols, errors="ignore")
        y_test = le.transform(test_df["label"])

        if st.button("Evaluate on Selected Coin"):

            if model_choice == "Random Forest":

                eval_model = RandomForestClassifier(
                    n_estimators=700,
                    max_depth=13,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    criterion="entropy",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )

                eval_model.fit(X_train, y_train_encoded)
                preds = eval_model.predict(X_test)

            elif model_choice == "KNN Distance":

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                eval_model = KNeighborsClassifier(
                    n_neighbors=9,
                    weights="distance",
                    metric="euclidean",
                    n_jobs=-1
                )

                eval_model.fit(X_train_scaled, y_train_encoded)
                preds = eval_model.predict(X_test_scaled)

            elif model_choice == "XGBoost":

                eval_model = XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=len(le.classes_),
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1
                )

                eval_model.fit(X_train, y_train_encoded)
                preds = eval_model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")

            st.metric("Accuracy on Selected Coin", round(acc, 4))
            st.metric("Macro F1 on Selected Coin", round(macro_f1, 4))

            report = classification_report(
                y_test,
                preds,
                target_names=le.classes_,
                output_dict=True,
                zero_division=0
            )

            st.dataframe(pd.DataFrame(report).transpose())