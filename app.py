import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Cross-Asset Crypto Movement Prediction",
    layout="wide"
)

st.title("Cross-Asset Cryptocurrency Movement Classification")


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("final_crypto.csv")

# Remove BTC and DOGE
df = df[~df["type"].isin(["BTC", "DOGE"])].copy()


# =========================
# BASIC SETTINGS
# =========================

drop_cols = [
    "label",
    "date",
    "type",
    "down_days_7",
    "taker_buy_volume",
    "z_score",
    "taker_buy_ratio",
    "taker_sell_ratio",
    "negative_momentum",
]

# Use only numeric features
feature_cols = (
    df.drop(columns=drop_cols, errors="ignore")
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

le = LabelEncoder()
le.fit(df["label"])


# =========================
# HELPER FUNCTIONS
# =========================

def get_random_forest():
    return RandomForestClassifier(
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


def get_knn():
    return KNeighborsClassifier(
        n_neighbors=9,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )


def get_xgboost():
    return XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )


# =========================
# SIDEBAR PAGES
# =========================

page = st.sidebar.radio(
    "Choose Page",
    ["Evaluation Page", "Prediction Page"]
)


# ============================================================
# PAGE 1: EVALUATION PAGE
# ============================================================

if page == "Evaluation Page":

    st.header("Evaluation Page")

    st.write("""
    This page tests cross-asset generalization.

    The model trains on all coins except the selected coin, then evaluates on the selected coin.
    BTC and DOGE are removed from the dataset.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    available_coins = sorted(df["type"].unique())

    eval_coin = st.sidebar.selectbox(
        "Choose coin to evaluate on",
        available_coins
    )

    model_choice = st.sidebar.selectbox(
        "Choose model",
        ["Random Forest", "KNN Distance", "XGBoost"]
    )

    run_eval = st.sidebar.button("Run Evaluation")

    if run_eval:

        train_df = df[df["type"] != eval_coin].copy()
        test_df = df[df["type"] == eval_coin].copy()

        X_train = train_df[feature_cols]
        y_train = le.transform(train_df["label"])

        X_test = test_df[feature_cols]
        y_test = le.transform(test_df["label"])

        st.subheader("Evaluation Setup")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Training Rows", len(train_df))

        with col2:
            st.metric("Testing Rows", len(test_df))

        with col3:
            st.metric("Test Coin", eval_coin)

        st.write("Training coins:", sorted(train_df["type"].unique()))
        st.write("Selected model:", model_choice)

        if model_choice == "Random Forest":

            model = get_random_forest()
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

        elif model_choice == "KNN Distance":

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = get_knn()
            model.fit(X_train_scaled, y_train)

            train_preds = model.predict(X_train_scaled)
            test_preds = model.predict(X_test_scaled)

        elif model_choice == "XGBoost":

            model = get_xgboost()
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        macro_f1 = f1_score(y_test, test_preds, average="macro")
        weighted_f1 = f1_score(y_test, test_preds, average="weighted")

        st.subheader("Evaluation Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Train Accuracy", round(train_acc, 4))

        with col2:
            st.metric("Test Accuracy", round(test_acc, 4))

        with col3:
            st.metric("Macro F1", round(macro_f1, 4))

        with col4:
            st.metric("Weighted F1", round(weighted_f1, 4))

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            test_preds,
            labels=np.arange(len(le.classes_)),
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            test_preds,
            labels=np.arange(len(le.classes_))
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_choice} Evaluation on {eval_coin}")

        st.pyplot(fig)


# ============================================================
# PAGE 2: PREDICTION PAGE
# ============================================================

elif page == "Prediction Page":

    st.header("Prediction Page")

    st.write("""
    This page predicts one movement result using manual slider values.

    Only ADA and AVAX are available here.
    The prediction model is Random Forest only.
    """)

    prediction_df = df[df["type"].isin(["ADA", "AVAX"])].copy()

    selected_coin = st.sidebar.selectbox(
        "Choose coin",
        ["ADA", "AVAX"]
    )

    predict_button = st.sidebar.button("Predict Movement")

    coin_df = prediction_df[prediction_df["type"] == selected_coin].copy()

    st.subheader("Selected Coin Data Preview")
    st.dataframe(coin_df.head())

    X_train = prediction_df[feature_cols]
    y_train = le.transform(prediction_df["label"])

    st.subheader("Enter Feature Values")

    st.write("""
    Use the sliders to enter the market values.
    The slider ranges are based on the selected coin history.
    """)

    user_input = {}

    cols = st.columns(3)

    for i, feature in enumerate(feature_cols):
        col = cols[i % 3]

        min_val = float(coin_df[feature].min())
        max_val = float(coin_df[feature].max())
        mean_val = float(coin_df[feature].mean())

        if min_val == max_val:
            user_input[feature] = mean_val
            col.write(f"{feature}: {mean_val}")
        else:
            user_input[feature] = col.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    input_df = pd.DataFrame([user_input])

    if predict_button:

        model = get_random_forest()

        model.fit(X_train, y_train)

        prediction_encoded = model.predict(input_df)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        prediction_proba = model.predict_proba(input_df)

        st.subheader("Final Prediction")

        st.success(f"The predicted movement for {selected_coin} is: {prediction_label}")

        st.subheader("Prediction Probability")

        proba_df = pd.DataFrame({
            "Class": le.classes_,
            "Probability": prediction_proba[0]
        })

        st.dataframe(proba_df)

        st.bar_chart(proba_df.set_index("Class"))

        st.subheader("Random Forest Feature Importance")

        feat_imp = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).sort_values()

        fig, ax = plt.subplots(figsize=(8, 6))
        feat_imp.plot(kind="barh", ax=ax)

        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")

        st.pyplot(fig)